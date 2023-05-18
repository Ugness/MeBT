"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import top_k_top_p_filtering

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    if n_idx == 0:
        return a
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, query, key, attn_bias, context_size=0, mode='none'):
        _attn_bias = attn_bias

        B, NQ, C = query.shape
        NK = key.shape[1]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, NK, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(query).view(B, NQ, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(key).view(B, NK, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, NQ, hs) x (B, nh, hs, NK) -> (B, nh, NQ, NK)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + _attn_bias

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, NQ, NK) x (B, nh, NK, hs) -> (B, nh, NQ, hs)
        y = y.transpose(1, 2).contiguous().view(B, NQ, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, None, None, None

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, mode):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        self.mode = mode

    def forward(self, sos_emb, contexts, targets, mask_emb=None, attn_bias=None):
        B, NS, C = sos_emb.size()
        _, NC, _ = contexts.size()
        _, NT, _ = targets.size()

        if self.mode=='latent_self':
            query = sos_emb
            key = sos_emb.clone()
        elif self.mode=='latent_enc':
            query = sos_emb
            key = contexts
        elif self.mode=='latent_dec':
            query = targets
            key = sos_emb
        elif self.mode=='lt2l':
            query = sos_emb
            key = torch.cat([sos_emb, targets], 1)
        elif self.mode=='maskgit':
            query = torch.cat([contexts, targets], 1)
            key = query.clone()
        else: assert 0
        query = self.ln1(query)
        key = self.ln1(key)
        attn, attn_score, idx, index = self.attn(query, key, attn_bias, NC, self.mode)

        x = query + attn
        x = x + self.mlp(self.ln2(x))

        if self.mode in ['latent_enc', 'latent_self', 'lt2l']:
            sos_emb = x
        elif self.mode=='latent_dec':
            targets = x
        elif self.mode in ['maskgit']:
            contexts, targets = x[:, :NC], x[:, NC:]
        else:
            assert 0
        return sos_emb, contexts, targets, attn_bias, idx


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, vtokens_pos=False,
                 mode=[]):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, mode=mode)
        if len(config.mode) < n_layer:
            config.mode += ['maskgit' for _ in range(n_layer-len(config.mode))]
        # input embedding stem
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        assert config.n_layer == len(config.mode)
        self.blocks = nn.Sequential(*[Block(config, mode) for mode in config.mode])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sos_emb, contexts, targets, mask_emb, attn_bias=None, debug=False):
        # forward the GPT model
        if attn_bias is None:
            attn_bias = 0.
        sos_emb = self.drop(sos_emb)
        contexts = self.drop(contexts)
        targets = self.drop(targets)
        mask_emb = self.drop(mask_emb)
        if debug: indices=[]
        for block in self.blocks:
            sos_emb, contexts, targets, attn_bias, idx = block(sos_emb, contexts, targets, mask_emb, attn_bias)
            if debug and idx is not None:
                indices.append(idx)
        x = self.ln_f(targets)
        logits = self.head(x)

        if debug:
            return logits, idx

        return logits, None

