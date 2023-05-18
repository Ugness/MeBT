# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import math
import copy
import random
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import repeat, rearrange

from .utils import shift_dim, accuracy, comp_getattr, ForkedPdb
from utils import instantiate_from_config
from .modules.gpt import GPT, complement_idx
from .modules.encoders import Labelator, SOSProvider, Identity
from .mask_sampler import MaskGen

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform(vid_lengths, t):
    return np.ones_like(vid_lengths, dtype=float)

def gaussian(vid_lengths, t, b, c):
    x = (-(t - (vid_lengths - 1) * b) ** 2) / (2*(b*c)**2)
    return np.exp(x)

def gaussian100000_2(vid_lengths, t):
    b = 100000
    c = 2
    T = (vid_lengths - 1) * b
    x = (-(t - (vid_lengths - 1) * b) ** 2) / (2*(b*c)**2)
    return np.exp(x)

def gaussian2(vid_lengths, t):
    b = 30000
    c = 2
    T = (vid_lengths - 1) * b
    x = (-(t - (vid_lengths - 1) * b) ** 2) / (2*(b*c)**2)
    return np.exp(x)

def longest(vid_lengths, t):
    x = np.zeros_like(vid_lengths, dtype=float)
    x[-1] = 1.
    return x

def linear(t):
    return 1.-t

def constant(t):
    return 1.

def cosine(t):
    return np.cos(t * np.pi / 2.)

class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 mask_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.config = transformer_config
        self.class_cond_dim = self.config.class_cond_dim
        self.be_unconditional = self.config.unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.first_stage_vocab_size = self.config.vocab_size
        self.cond_stage_key = cond_stage_key
        self.vtokens = self.config.vtokens
        self.n_embd = self.config.n_embd
        self.vis_epoch = self.config.vis_epoch
        if not hasattr(self.config, 'avg_loss'):
            self.config.avg_loss = 0.0
        self.config.avg_loss = float(self.config.avg_loss)
        if not hasattr(self.config, 'embd_pdrop'):
            self.config.embd_pdrop=0.0
        if not hasattr(self.config, 'resid_pdrop'):
            self.config.resid_pdrop=0.0
        if not hasattr(self.config, 'attn_pdrop'):
            self.config.attn_pdrop=0.0
        if hasattr(self.config, 'sample_every_n_latent_frames'):
            self.sample_every_n_latent_frames = self.config.sample_every_n_latent_frames
        else:
            self.sample_every_n_latent_frames = 0

        if hasattr(self.config, 'label_smoothing'):
            self.label_smoothing = self.config.label_smoothing
        else:
            self.label_smoothing = 0.0

        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(self.config)

        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size
        self.transformer = GPT(gpt_vocab_size, self.config.block_size, n_layer=self.config.n_layer, n_head=self.config.n_head, 
                                n_embd=self.config.n_embd, vtokens_pos=self.config.vtokens_pos, n_unmasked=self.config.n_unmasked,
                                attn_pdrop=self.config.attn_pdrop, embd_pdrop=self.config.embd_pdrop, resid_pdrop=self.config.resid_pdrop,
                                mode=self.config.mode)

        self.mask_sampler = instantiate_from_config(config=mask_config)

        if not hasattr(self.config, 'beta_params'):
            self.range = [0., 1.] if not hasattr(mask_config.params, 't_range') else mask_config.params.t_range
            self.beta = False
        else:
            self.beta_params = self.config.beta_params
            self.beta_iter = float(self.config.beta_iter)
            self.beta = True
        T = self.mask_sampler.shape[0]
        self.t_lengths = np.array(list(range(T))) + 1

        if not hasattr(self.config, 't_prior'):
            self.config.t_prior = 'longest'
        self.t_prior = eval(self.config.t_prior)
        self.tok_emb = nn.Embedding(gpt_vocab_size, self.config.n_embd)
        self.tok_emb.weight.data.normal_(mean=0.0, std=0.02)

        self.mask_emb = nn.Parameter(torch.zeros(1, 1, self.config.n_embd))
        self.mask_emb.data.normal_(mean=0.0, std=0.02)

        if not hasattr(self.config, 'sos_emb'):
            self.config.sos_emb = 1
        if self.config.sos_emb > 0:
            self.sos_emb = nn.Parameter(torch.zeros(1, self.config.sos_emb, self.config.n_embd))
            self.sos_emb.data.normal_(mean=0.0, std=0.02)

        self.num_pos = np.prod(self.mask_sampler.shape[1:])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, self.config.n_embd))
        self.pos_emb.data.normal_(mean=0.0, std=0.02)
        self.n_head = self.config.n_head
        self.first_mode = self.config.mode[0]
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        self.save_hyperparameters()

    @staticmethod
    def _get_bias(block_size, shape, n_heads):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        slopes = torch.tensor(get_slopes(n_heads)).view(1, -1, 1, 1)
        T = block_size // np.prod(shape[1:])
        base_mx = -torch.abs(torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1))
        base_mx = base_mx.view(1, 1, T, T)
        bias = base_mx * slopes
        bias.requires_grad = False
        return bias

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.copy().keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        from .download import load_vqgan
        if not self.vtokens:
            self.first_stage_model = load_vqgan(config.params.ckpt_path)
            for p in self.first_stage_model.parameters():
                p.requires_grad = False
            self.first_stage_model.codebook._need_init = False
            self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes
        else:
            self.first_stage_model = None
            self.first_stage_vocab_size = 16384
            # self.first_stage_vocab_size = self.args.first_stage_vocab_size

        '''
        model = instantiate_from_config(config)
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.train = disabled_train
        self.first_stage_model = model
        '''

    def init_cond_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
            self.cond_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond_stage_key)

    def forward(self, x, c, t=None, indices=None, vid_t=None, debug=False):
        # one step to produce the logits
        assert indices is not None
        _, x_indices = self.encode_to_z(x)

        B = x_indices.shape[0]
        N = int(np.prod(x_indices.shape[1:]))
        C = self.config.sos_emb

        if t is None:
            if self.training or debug:
                if not self.beta:
                    t = torch.tensor(random.random())
                    t = self.range[0] + t * (self.range[1] - self.range[0])
                else:
                    if self.global_step > self.beta_iter:
                        alpha, beta = 1., 1.
                    else:
                        alpha_, beta_ = self.beta_params
                        alpha = alpha_ - (alpha_ - 1.) * (self.global_step / self.beta_iter)
                        beta = beta_ - (beta_ - 1.) * (self.global_step / self.beta_iter)
                    t = torch.distributions.beta.Beta(alpha, beta).sample()
            else:
                t = torch.tensor(random.random())
        else:
            t = torch.tensor(t)

        if vid_t is None:
            prior_t = self.t_prior(self.t_lengths, self.global_step)
            vid_t = self.t_lengths
        else:
            assert len(vid_t) == 1
            prior_t = np.ones_like(vid_t, dtype=float)
        # checked

        context_indices, target_indices, seq_len = self.mask_sampler.divide_indices(indices, t, vid_t, prior_t, debug)

        # checked
        # Gather z_indices and targets
        z_contexts = torch.gather(x_indices, 1, context_indices)
        z_targets = torch.gather(x_indices, 1, target_indices)
        N = seq_len
        NC = z_contexts.shape[1]
        NT_weight = float(N - NC)
        NT = z_targets.shape[1]

        context_embeddings = self.tok_emb(z_contexts) # each index maps to a (learnable) vector
        target_embeddings = self.mask_emb.repeat(B, NT, 1)

        # Add PE
        pe = self.pos_emb.repeat(B, 1, 1)
        context_addr = context_indices.unsqueeze(-1).repeat(1, 1, self.n_embd)
        target_addr = target_indices.unsqueeze(-1).repeat(1, 1, self.n_embd)
        context_pe = torch.gather(pe, 1, context_addr)
        target_pe = torch.gather(pe, 1, target_addr)
        contexts = context_pe + context_embeddings
        targets = target_pe + target_embeddings
        if C > 0:
            sos_emb = self.sos_emb.repeat(B, 1, 1)
        else:
            sos_emb = torch.zeros(B, 0, self.config.n_embd).to(x.device)
        mask_emb = self.mask_emb.repeat(B, 1, 1)

        # indices: B, N
        # self.attn_bias: 1, n_head, NQ, NK
        attn_bias_cond = 0.

        logits, _ = self.transformer(sos_emb, contexts, targets, mask_emb, attn_bias_cond)

        # checked
        return logits, z_targets, NT_weight, seq_len

    def reconstruct_mask(self, x_indices, context_indices, target_indices, debug=False):
        # one step to produce the logits

        B = x_indices.shape[0]
        N = int(np.prod(x_indices.shape[1:]))
        C = self.config.sos_emb

        x_indices = x_indices.reshape(B, N)

        # Gather z_indices and targets
        z_contexts = torch.gather(x_indices, 1, context_indices)
        z_targets = torch.gather(x_indices, 1, target_indices)
        NT = z_targets.shape[1]

        context_embeddings = self.tok_emb(z_contexts) # each index maps to a (learnable) vector
        target_embeddings = self.mask_emb.repeat(B, NT, 1)

        # Add PE
        pe = self.pos_emb.repeat(B, 1, 1)
        context_addr = context_indices.unsqueeze(-1).repeat(1, 1, self.n_embd)
        target_addr = target_indices.unsqueeze(-1).repeat(1, 1, self.n_embd)
        context_pe = torch.gather(pe, 1, context_addr)
        target_pe = torch.gather(pe, 1, target_addr)
        contexts = context_pe + context_embeddings
        targets = target_pe + target_embeddings
        if C > 0:
            sos_emb = self.sos_emb.repeat(B, 1, 1)
        else:
            sos_emb = torch.zeros(B, 0, self.config.n_embd).to(x_indices.device)
        mask_emb = self.mask_emb.repeat(B, 1, 1)

        # indices: B, N
        # self.attn_bias: 1, n_head, NQ, NK
        attn_bias_cond = 0.
        logits, idx = self.transformer(sos_emb, contexts, targets, mask_emb, attn_bias_cond, debug)

        return logits, idx

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def on_train_epoch_start(self):
        # Adjust K
        epo = self.current_epoch

    def on_validation_epoch_start(self):
        if (self.current_epoch+1) % self.vis_epoch == 0:
            orig_schedule = copy.deepcopy(self.mask_sampler.schedule)
            self.mask_sampler.schedule = 'cosine'
            shape = (4, *self.mask_sampler.shape)
            x = torch.zeros(shape, dtype=torch.long, device=self.device)
            x = self.sample(x, None, 1.0, None, None, 32, None, None, context_temperature=6.0, skips=False)[0]
            code_map = x.reshape(*shape)
            img_x = []
            for i in range(code_map.shape[0]):
                img_x.append(self.first_stage_model.decode(code_map[i:i+1]))
            img_x = torch.cat(img_x, 0).clamp(-0.5, 0.5) + 0.5
            img_x = img_x.permute(0, 2, 1, 3, 4)
            self.logger.experiment.add_video('sample', img_x, self.current_epoch, fps=20)
            self.logger.experiment.flush()
            self.mask_sampler.schedule = orig_schedule
            
    @torch.no_grad()
    def sample(self, x, c,
            temperature=1.0,
            top_k=None,
            top_p=None,
            n_steps=8,
            context_indices=None,
            target_indices=None,
            strategy='maskgit',
            context_temperature=4.5,
            phase_history=None,
            refine_steps=1,
            forget_pivot=False,
            skips=[False, False, False],
            debug=False,
            ctemp_schedule='linear',
            edit=False
            ):
        B = x.shape[0]
        N = int(np.prod(x.shape[1:]))
        if edit:
            edit_N = target_indices.shape[1]
        else:
            edit_N = N
        x = x.reshape(B, N)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if debug:
            history=[]
            context_history=[]
        if strategy in ['maskgit', 'random', 'mlm', 'bootstrap']:
            if context_indices is None:
                context_indices = torch.empty(B, 0).long().to(x.device)
                target_indices = torch.stack([torch.arange(N) for _ in range(B)]).to(x.device)
            else:
                context_indices = context_indices.clone()
                target_indices = target_indices.clone()

            timesteps = np.linspace(0, 1, n_steps+1)
            partial_sample = x
            if debug:
                history.append(partial_sample.clone())
                indices=[]
                partial_probs = -torch.ones(B, N, 16384).to(x.device)
            for t_next in timesteps[1:]:
                t = torch.full((B,), fill_value=t_next, device=x.device)
                n_masked_toks = torch.ceil(self.mask_sampler.schedule_fn(t) * edit_N)
                # if the context is bigger than expected, go to the next step
                if (n_masked_toks > target_indices.shape[-1]).sum() == B:
                    continue
                logits, idx = self.reconstruct_mask(partial_sample, context_indices, target_indices, debug)
                if idx is not None and idx[0] is not None:
                    final_idx = context_indices.gather(1, idx[0])
                    indices.append((context_indices, final_idx))
                xs_d, probs_d = sample_from_logits(logits, temperature, top_k, top_p, return_probs=True)

                scores = probs_d.gather(dim=-1, index=xs_d.unsqueeze(-1)).squeeze(-1)

                target_indices = target_indices.view(B, -1)

                coo_idx = torch.stack([torch.arange(B) for _ in range(target_indices.shape[-1])], dim=1).to(x.device)
                coo_idx = torch.stack([coo_idx, target_indices], dim=-1).view(-1, 2)

                coo_mask = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        torch.ones(len(coo_idx), device=x.device),
                        size=(B,N))
                dense_mask = coo_mask.to_dense().bool()
                coo_values = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        xs_d.view(-1),
                        size=(B,N))
                dense_values = coo_values.to_dense()
                
                if debug:
                    coo_probs = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        probs_d.view(-1,probs_d.shape[-1]),
                        size=(B,N,probs_d.shape[-1])
                    )
                    dense_probs = coo_probs.to_dense()
                    partial_probs = torch.where(
                        dense_mask.unsqueeze(-1).expand_as(partial_probs),
                        dense_probs,
                        partial_probs)

                partial_sample = torch.where(dense_mask, dense_values, partial_sample)
                actual_temperature = context_temperature * eval(ctemp_schedule)(t_next)
                if debug:
                    history.append(partial_sample.clone())
                    context_history.append(context_indices)
                context_indices, target_indices = self.mask_sampler.generate_next_mask(context_indices, target_indices, scores, t_next, strategy=strategy, context_temperature=actual_temperature, n_masked_toks=n_masked_toks)
            if debug:
                return partial_sample.view(B, -1), context_indices, target_indices, history, context_history, partial_probs
            return partial_sample.view(B, -1), context_indices, target_indices
    
    @torch.no_grad()
    def entp_sample(self, x, c,
            temperature=1.0,
            top_k=None,
            top_p=None,
            n_steps=8,
            context_indices=None,
            target_indices=None,
            strategy='maskgit',
            context_temperature=4.5,
            phase_history=None,
            refine_steps=1,
            forget_pivot=False,
            skips=[False, False, False],
            debug=False,
            ctemp_schedule='linear',
            ):
        B = x.shape[0]
        N = int(np.prod(x.shape[1:]))
        x = x.reshape(B, N)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if debug:
            history=[]
            context_history=[]
        if strategy in ['ar']:
            raise NotImplementedError

        if strategy in ['maskgit', 'random', 'mlm', 'bootstrap']:
            if context_indices is None:
                context_indices = torch.empty(B, 0).long().to(x.device)
                target_indices = torch.stack([torch.arange(N) for _ in range(B)]).to(x.device)
            else:
                context_indices = context_indices.clone()
                target_indices = target_indices.clone()

            timesteps = np.linspace(0, 1, n_steps+1)
            partial_sample = x
            if debug:
                history.append(partial_sample.clone())
                indices=[]
                partial_probs = -torch.ones(B, N, 16384).to(x.device)
            for t_next in timesteps[1:]:
                t = torch.full((B,), fill_value=t_next, device=x.device)
                n_masked_toks = torch.ceil(self.mask_sampler.schedule_fn(t) * N)
                # if the context is bigger than expected, go to the next step
                if (n_masked_toks > target_indices.shape[-1]).sum() == B:
                    continue
                logits, idx = self.reconstruct_mask(partial_sample, context_indices, target_indices, debug)
                if idx is not None and idx[0] is not None:
                    final_idx = context_indices.gather(1, idx[0])
                    indices.append((context_indices, final_idx))
                xs_d, probs_d = sample_from_logits(logits, temperature, top_k, top_p, return_probs=True)

                scores = -(-probs_d + torch.log(probs_d + 1e-8)).sum(-1)
                scores = scores.max(-1, keepdim=True)[0] - scores

                target_indices = target_indices.view(B, -1)

                coo_idx = torch.stack([torch.arange(B) for _ in range(target_indices.shape[-1])], dim=1).to(x.device)
                coo_idx = torch.stack([coo_idx, target_indices], dim=-1).view(-1, 2)

                coo_mask = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        torch.ones(len(coo_idx), device=x.device),
                        size=(B,N))
                dense_mask = coo_mask.to_dense().bool()
                coo_values = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        xs_d.view(-1),
                        size=(B,N))
                dense_values = coo_values.to_dense()
                
                if debug:
                    coo_probs = torch.sparse_coo_tensor(
                        coo_idx.t(),
                        probs_d.view(-1,probs_d.shape[-1]),
                        size=(B,N,probs_d.shape[-1])
                    )
                    dense_probs = coo_probs.to_dense()
                    partial_probs = torch.where(
                        dense_mask.unsqueeze(-1).expand_as(partial_probs),
                        dense_probs,
                        partial_probs)

                partial_sample = torch.where(dense_mask, dense_values, partial_sample)
                actual_temperature = context_temperature * eval(ctemp_schedule)(t_next)
                if debug:
                    history.append(partial_sample.clone())
                    context_history.append(context_indices)
                context_indices, target_indices = self.mask_sampler.generate_next_mask_entp(context_indices, target_indices, scores, t_next, strategy=strategy, context_temperature=0.0)
            if debug:
                return partial_sample.view(B, -1), context_indices, target_indices, history, context_history, partial_probs
            return partial_sample.view(B, -1), context_indices, target_indices

    @torch.no_grad()
    def draft(self, x, c,
            temperature=1.0,
            top_k=None,
            top_p=None,
            n_steps=8,
            debug=False,
            context_indices=None,
            target_indices=None
            ):
        B = x.shape[0]
        N = int(np.prod(x.shape[1:]))
        x = x.reshape(B, N)
        block_size = self.transformer.get_block_size()
        if context_indices is None:
            context_indices = torch.empty(B, 0).long().to(x.device)
            target_indices = torch.stack([torch.arange(N) for _ in range(B)]).to(x.device)
        # get draft_mask
        context_indices, target_indices = self.mask_sampler.create_gibbs_draft_mask(context_indices, target_indices, n_steps, x.device)
        assert not self.transformer.training
        partial_sample = x
        for c, t in zip(context_indices, target_indices):
            logits, _ = self.reconstruct_mask(partial_sample, c, t, debug)
            xs_d, probs_d = sample_from_logits(logits, temperature, top_k, top_p, return_probs=True)
            scores = probs_d.gather(dim=-1, index=xs_d.unsqueeze(-1)).squeeze(-1)
            t = t.view(B, -1)

            coo_idx = torch.stack([torch.arange(B) for _ in range(t.shape[-1])], dim=1).to(x.device)
            coo_idx = torch.stack([coo_idx, t], dim=-1).view(-1, 2)

            coo_mask = torch.sparse_coo_tensor(
                    coo_idx.t(),
                    torch.ones(len(coo_idx), device=x.device),
                    size=(B,N))
            dense_mask = coo_mask.to_dense().bool()
            coo_values = torch.sparse_coo_tensor(
                    coo_idx.t(),
                    xs_d.view(-1),
                    size=(B,N))
            dense_values = coo_values.to_dense()

            partial_sample = torch.where(dense_mask, dense_values, partial_sample)
        return partial_sample.view(B, -1)

    @torch.no_grad()
    def revise(self, x, c,
            temperature=1.0,
            top_k=None,
            top_p=None,
            n_steps=8,
            debug=False,
            context_indices=None,
            target_indices=None
            ):
        B = x.shape[0]
        N = int(np.prod(x.shape[1:]))
        x = x.reshape(B, N)
        block_size = self.transformer.get_block_size()
        if context_indices is None:
            context_indices = torch.empty(B, 0).long().to(x.device)
            target_indices = torch.stack([torch.arange(N) for _ in range(B)]).to(x.device)

        context_indices, target_indices = self.mask_sampler.create_gibbs_revise_mask(context_indices, target_indices, n_steps, x.device)
        assert not self.transformer.training
        partial_sample = x
        for c, t in zip(context_indices, target_indices):
            logits, _ = self.reconstruct_mask(partial_sample, c, t, debug)
            xs_d, probs_d = sample_from_logits(logits, temperature, top_k, top_p, return_probs=True)
            scores = probs_d.gather(dim=-1, index=xs_d.unsqueeze(-1)).squeeze(-1)
            t = t.view(B, -1)

            coo_idx = torch.stack([torch.arange(B) for _ in range(t.shape[-1])], dim=1).to(x.device)
            coo_idx = torch.stack([coo_idx, t], dim=-1).view(-1, 2)

            coo_mask = torch.sparse_coo_tensor(
                    coo_idx.t(),
                    torch.ones(len(coo_idx), device=x.device),
                    size=(B,N))
            dense_mask = coo_mask.to_dense().bool()
            coo_values = torch.sparse_coo_tensor(
                    coo_idx.t(),
                    xs_d.view(-1),
                    size=(B,N))
            dense_values = coo_values.to_dense()

            partial_sample = torch.where(dense_mask, dense_values, partial_sample)
        return partial_sample.view(B, -1)

    @torch.no_grad()
    def draft_and_revise(self, x, c,
            n_draft=8,
            draft_t=1.0,
            draft_k=None,
            draft_p=None,
            n_revise=8,
            revise_t=1.0,
            revise_k=None,
            revise_p=None,
            M=2,
            skip_draft = False,
            debug=False,
            context_indices=None,
            target_indices=None,
            edit=False,
            ):
        B = x.shape[0]
        N = int(np.prod(x.shape[1:]))
        x = x.reshape(B, N)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        # draft
        if not skip_draft:
            x = self.draft(x, c, draft_t, draft_k, draft_p, n_draft, debug, context_indices, target_indices)
        # revise M times
        if edit:
            context_indices = None
            target_indices = None
        for _ in range(M):
            x = self.revise(x, c, revise_t, revise_k, revise_p, n_revise, debug, context_indices, target_indices)
        return x.view(B, -1)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = self.learning_rate * lr_scale
        elif self.cosine_lr:
            step = float(self.trainer.global_step - self.warmup_steps)
            rad = step / float(self.trainer.max_steps - self.warmup_steps)
            assert rad >= 0
            lr_scale = 0.5 * (1 + np.cos(rad * np.pi))
            for pg in optimizer.param_groups:
                pg['lr'] = self.learning_rate * lr_scale
        else:
            lr_scale = 1.
        self.log("learning_rate", self.learning_rate * lr_scale, logger=True, on_step=True, sync_dist=True)

        optimizer.step(closure=opt_closure)

    @torch.no_grad()
    def encode_to_z(self, x):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                x = x[:, :, ::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            targets = targets.reshape(targets.shape[0], -1)
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, indices = self.cond_stage_model.encode(c, include_embeddings=True)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c = self.get_xc(batch)
        indices = self.get_input('indices', batch)
        logits, target, NT_weight, seq_len = self(x, c, indices=indices)
        ratio = NT_weight / float(seq_len)

        B, N, C = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction='sum', label_smoothing=self.label_smoothing)
        ce_loss = loss.detach()
        # Monitor CE Loss
        weight = ratio ** self.config.avg_loss
        loss = loss / (B * seq_len * weight)
        acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
        return acc1, acc5, loss, ratio

    def training_step(self, batch, batch_idx):
        acc1, acc5, loss, ratio = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        acc1, acc5, loss, ratio = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        idx = min([(int(ratio * 100) // 20) * 20, 80])
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        emb_dict = {pn: p for pn, p in self.named_parameters() if '_emb' in pn}
        del emb_dict['pos_emb']
        pos_emb_dict = {pn: p for pn, p in self.named_parameters() if 'pos_emb' in pn}

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": emb_dict.values(), "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {"params": pos_emb_dict.values(), "weight_decay": 0.0},

        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--stft_vqvae', type=str, help='path to vqgan ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        # VideoGPT hyperparmeters
        parser.add_argument('--vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)
        parser.add_argument('--block_size', type=int, default=256)
        parser.add_argument('--n_layer', type=int, default=48)
        parser.add_argument('--n_head', type=int, default=24)
        parser.add_argument('--n_embd', type=int, default=1536)
        parser.add_argument('--n_unmasked', type=int, default=0)
        parser.add_argument('--sample_every_n_latent_frames', type=int, default=0)
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
        parser.add_argument('--iid', action='store_true')
        parser.add_argument('--schedule', type=str, default='cosine')
        parser.add_argument('--max_token', type=int, default=1024)
        parser.add_argument('--method', type=str, default=None)

        return parser

def gumbel_sort(prob):
    # https://github.com/pytorch/pytorch/blob/dc81ba1f9f699bc7366f9e96aca0fb2fb09aba2c/aten/src/ATen/native/Distributions.cpp
    '''
    Get:
    - prob (*, C)
    Return:
    - indices (*, C)
    '''
    prob = prob / prob.sum(-1, keepdim=True)
    mask = prob > 0

    prob = prob / torch.empty_like(prob).exponential_()
    prob = prob * mask.float()
    val, indices = prob.sort(dim=-1, descending=True)
    
    return indices

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, return_probs=False):
    """Take a 2-dim tensor, apply softmax along each row, and sample from
    each multinomial distribution defined by the rows.

    Args:
        logits: 2-dim tensor of shape (n_samples, logit_dim)
        temperature (float): softmax temperature
        top_k (Optional[int]): if given, sample only using `top_k` logits
        top_p (Optional[float]): if given, sample only using `top_p` logits
        return_probs (bool): if True, return probs

    Returns:
        samples: 1-dim integer tensor of shape (n_samples,)
        probs (optional): 2-dim tensor of shape (n_samples, logit_dim)
    """

    logits = logits.to(dtype=torch.float32)
    logits = logits / (temperature + 1e-8)

    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')

    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    try:
        samples = gumbel_sort(probs)[..., 0]
    except RuntimeError:
        print(probs)
        print(logits)
        print('isinf, ', torch.sum(torch.isinf(probs)))
        print('isnan, ', torch.sum(torch.isnan(probs)))
        print('is negative', torch.sum(probs < 0))
        raise

    if return_probs:
        return samples, probs.detach().clone()
    else:
        return samples

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out


def top_p_probs(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return norm_probs

