import numpy as np
import torch
from torch import nn as nn
from einops import rearrange, repeat
from .modules.gpt import complement_idx
import random


class MaskGen(nn.Module):
    """Utility module to generate mask according to a prescribed noise schedule."""
    def __init__(self, iid=False, schedule='cosine', max_token=256, method=None, shape=(4, 16, 16), t_range=(0., 1.), budget=1024):
        super().__init__()

        if schedule not in self._available_schedules:
            raise ValueError(f'Unsupported schedule: {schedule}')

        if method is not None:
            self.method = method
        else:
            self.method = 'iid' if iid else 'mlm'

        if self.method not in self._available_methods:
            raise ValueError(f'Unsupported method: {self.method}')

        self.schedule = schedule
        self.device = None
        self.shape = shape
        self.seq_len = np.prod(shape)
        self.max_token = max_token
        self.dense = True
        self.range = t_range if t_range is not None else (0., 1.)
        self.budget = budget

    @staticmethod
    def cosine(t):
        return torch.cos(0.5 * np.pi * t)
    
    @staticmethod
    def cosine_plus(t):
        return 0.5*(1+torch.cos(np.pi * t))


    @staticmethod
    def linear(t):
        return 1.0 - t

    @staticmethod
    def quadratic(t):
        return (1.0 - t) ** 2.0

    @staticmethod
    def square(t):
        return 1.0 - t ** 2.0

    @staticmethod
    def cube(t):
        return 1.0 - t ** 3.0

    @staticmethod
    def sqrt(t):
        return 1.0 - t ** 0.5

    @staticmethod
    def convex(t):
        return (1.0 - t) ** 3.0

    _available_schedules = ['cosine', 'linear', 'quadratic', 'sqrt', 'square', 'cube', 'ar', 'cosine_plus', 'convex']
    _available_methods = ['iid', 'mlm', 'partial_mlm', 'block', 'ar', 'phase', 'grid', 'frame', 'interpolate', 'stochastic_phase', 'stochastic_grid', 'fdm', 'softfdm', 'clip']
    # _available_methods = ['local']

    @property
    def schedule_fn(self):
        return getattr(self, self.schedule)

    def divide_indices(self, indices, t, vid_t, prior_t, debug=False):
        """Generate a mask w.r.t. the given `shape` and `ratios`.
        Mask is generated independently for each sample in the batch.
        Here `ratios` is the list of per sample proportions of masked location.
        """

        mask_ratio = self.schedule_fn(t)

        if self.training or debug:
            # Video Slicing
            max_T = self.shape[0]
            num_pos = np.prod(self.shape[1:])
            prior_t = prior_t / prior_t.sum()
            T = np.random.choice(vid_t, p=prior_t)
            if max_T != T:
                start_t = 0 if max_T==T else np.random.randint(0, max_T-T+1)
                end_t = start_t + T

                start_idx = start_t * num_pos
                end_idx = end_t * num_pos
                seq_len = T*num_pos
                sliced_indices = torch.zeros(indices.shape[0], seq_len, dtype=torch.int64)
                for i, idx in enumerate(indices):
                    sliced_indices[i] = idx[(idx >= start_idx) * (idx < end_idx)]
                indices = sliced_indices.to(indices.device)

        seq_len = np.prod(indices.shape[1:])
        n_masked_toks = torch.ceil(mask_ratio * seq_len).to(dtype=torch.long)
        n_contexts = seq_len - n_masked_toks

        if self.training or debug:
            budget = self.budget
        else:
            budget = seq_len

        # cut the budget
        n_targets = min(budget, seq_len - n_contexts)

        context_indices = indices[:, :n_contexts]
        target_indices = indices[:, -n_targets:]
        return context_indices, target_indices, seq_len

    def sample_mlm_mask(self, shape, ratios, max_token=None):
        """Generate a mask w.r.t. the given `shape` and `ratios`.
        Mask is generated independently for each sample in the batch.
        Here `ratios` is the list of per sample proportions of masked location.
        """
        assert ratios.shape[0] == shape[0]

        seq_len = np.prod(shape[1:])
        n_masked_toks = torch.ceil(ratios[0] * seq_len).to(dtype=torch.long)
        n_contexts = seq_len - n_masked_toks

        if self.training:
            budget = self.budget
        else:
            budget = seq_len

        # cut the budget
        n_targets = min(budget, seq_len - n_contexts)

        context_indices = torch.zeros(shape[0], n_contexts, dtype=torch.long, device=ratios.device)
        target_indices = torch.zeros(shape[0], n_targets, dtype=torch.long, device=ratios.device)
        assert n_contexts + n_targets <= seq_len
        for sample_idx, ratio in enumerate(ratios):
            indices = torch.randperm(seq_len)
            context_indices[sample_idx] = indices[:n_contexts]
            target_indices[sample_idx] = indices[-n_targets:]
        return context_indices, target_indices

    def sample_mask(self, shape, ratios, max_token, debug=False, context_ratios=None, method=None):
        if max_token is None:
            max_token = self.max_token
        if method in ['iid']:
            raise NotImplementedError
            indices, masks, attn_masks = self.sample_iid_mask(shape, ratios, max_token)
        elif method in ['ar']:
            raise NotImplementedError
            indices, masks, attn_masks = self.sample_ar_mask(shape, ratios, max_token)
        elif method in ['mlm', 'maskgit', 'random']:
            context_indices, target_indices = self.sample_mlm_mask(shape, ratios, max_token)

        return context_indices, target_indices

    def forward(self, shape, t=None, max_token=None, device=None, debug=False, method=None):
        B = shape[0]

        if method is None:
            method = self.method
        if t is None:
            t = torch.rand(B, device=device)
            if self.training:
                t = self.range[0] + t * (self.range[1] - self.range[0])

        if isinstance(t, float):
            t = torch.full((B,), fill_value=t, device=device)

        mask_ratios = self.schedule_fn(t)
        context_ratios=None
        context_indices, target_indices = self.sample_mask(shape, mask_ratios, max_token, debug=debug, context_ratios=context_ratios, method=method)

        return context_indices, target_indices

    @staticmethod
    def gumbel_top_k(prob, context_temperature=1.0):
        prob = prob / prob.sum(-1, keepdim=True)

        q = torch.empty_like(prob).exponential_()
        prob = prob / (q ** context_temperature)

        val, indices = prob.sort(dim=-1, descending=True)

        return indices

    def generate_next_mask(self, context_indices, target_indices, score, t, strategy='maskgit', context_temperature=4.5, n_masked_toks=None, debug=False):
        """Generate next mask given previous mask, score, and next time-step t."""
        """prev_mask, indices, scores should be in the same order"""
        # assert strategy in ['maskgit', 'random', 'ar']
        if score is not None and strategy != 'ar':
            assert target_indices.shape == score.shape
        shape = target_indices.shape
        B = shape[0]
        B, NC = context_indices.shape
        _, NT = target_indices.shape
        # seq_len = np.prod(shape[1:])

        if strategy in ['maskgit', 'random', 'mlm', 'bootstrap']:
            # this function returns additional contexts based on the score mx.
            # indices are preserved.
            if isinstance(t, float):
                t = torch.full((B,), fill_value=t, device=score.device)
            if strategy in ['random', 'bootstrap']:
                score = torch.randn_like(score)
                context_temperature = 0.0

            seq_len = NC + NT

            if n_masked_toks is None:
                mask_ratios = self.schedule_fn(t)
                n_masked_toks = torch.ceil(mask_ratios[0] * seq_len).to(dtype=torch.long)
            else:
                n_masked_toks = n_masked_toks[0].long()

            if strategy == 'bootstrap':
                n_masked_toks = NT-1
            n_contexts = seq_len - n_masked_toks
            next_context = torch.zeros(B, n_contexts, dtype=torch.long, device=score.device)
            if n_contexts <= NC:
                if debug:
                    return context_indices, target_indices, None
                return context_indices, target_indices
            else:
                n_new_toks = n_contexts-NC
                next_context[:, :NC] = context_indices.clone()
                score_locs = self.gumbel_top_k(score, context_temperature)
                # Pass high scores
                masking_locs = score_locs[:, n_new_toks:]
                context_locs = score_locs[:, :n_new_toks]
                next_context[:, NC:] = torch.gather(target_indices, -1, context_locs)
                next_target = torch.gather(target_indices, -1, masking_locs)
            if debug:
                return next_context, next_target, context_locs
            return next_context, next_target

        elif strategy == 'ar':
            next_context = torch.cat([context_indices, target_indices[:, :1]], 1)
            next_target = target_indices[:, 1:]
            context_locs = torch.zeros_like(target_indices[:, :1])
            if debug:
                return next_context, next_target, context_locs
            else:
                return next_context, next_target
    
    def generate_next_mask_entp(self, context_indices, target_indices, score, t, strategy='maskgit', context_temperature=4.5, n_masked_toks=None, debug=False):
        """Generate next mask given previous mask, score, and next time-step t."""
        """prev_mask, indices, scores should be in the same order"""
        # assert strategy in ['maskgit', 'random', 'ar']
        if score is not None and strategy != 'ar':
            assert target_indices.shape == score.shape
        shape = target_indices.shape
        B = shape[0]
        B, NC = context_indices.shape
        _, NT = target_indices.shape
        # seq_len = np.prod(shape[1:])

        if strategy in ['maskgit', 'random', 'mlm', 'bootstrap']:
            # this function returns additional contexts based on the score mx.
            # indices are preserved.
            if isinstance(t, float):
                t = torch.full((B,), fill_value=t, device=score.device)
            if strategy in ['random']:
                score = torch.randn_like(score)
                context_temperature = 0.0

            seq_len = NC + NT

            if n_masked_toks is None:
                mask_ratios = self.schedule_fn(t)
                n_masked_toks = torch.ceil(mask_ratios[0] * seq_len).to(dtype=torch.long)

            if strategy == 'bootstrap':
                n_masked_toks = NT-1
            n_contexts = seq_len - n_masked_toks
            next_context = torch.zeros(B, n_contexts, dtype=torch.long, device=score.device)
            if n_contexts <= NC:
                if debug:
                    return context_indices, target_indices, None
                return context_indices, target_indices
            else:
                n_new_toks = n_contexts-NC
                next_context[:, :NC] = context_indices.clone()
                score_locs = self.gumbel_top_k(score, context_temperature)
                # Pass high scores
                masking_locs = score_locs[:, n_new_toks:]
                context_locs = score_locs[:, :n_new_toks]
                next_context[:, NC:] = torch.gather(target_indices, -1, context_locs)
                next_target = torch.gather(target_indices, -1, masking_locs)
            if debug:
                return next_context, next_target, context_locs
            return next_context, next_target

        elif strategy == 'ar':
            next_context = torch.cat([context_indices, target_indices[:, :1]], 1)
            next_target = target_indices[:, 1:]
            context_locs = torch.zeros_like(target_indices[:, :1])
            if debug:
                return next_context, next_target, context_locs
            else:
                return next_context, next_target
    
    def cut_entp(self, entps, threshold):
        noise = torch.randn_like(noise)
        entps = torch.where(entps < threshold, entps, entps.max()+noise)
        val, indices = entps.sort(dim=-1, descending=False)
        if entps < threshold:
            # select all
            pass
        else:
            # randperm, select 1
            pass
        return context_indices, target_indices

    @staticmethod
    def create_gibbs_revise_mask(context_indices, target_indices, num_unit_gibbs_steps, device):
        B = target_indices.shape[0]
        N = np.prod(target_indices.shape[1:])
        n_steps = num_unit_gibbs_steps  # for brevity in local

        def is_power_of_two(num):
            return num and not (num & (num - 1))

        # assert is_power_of_two(N)
        # assert is_power_of_two(n_steps) and n_steps > 1  # n_steps == 1 is not meaningful!
        assert N % n_steps == 0

        n_masked_elem = N // n_steps
        rand_indices = torch.stack([torch.randperm(N) for _ in range(B)]).to(device)
        target_indices = torch.gather(target_indices, 1, rand_indices)
        indices = repeat(target_indices, 'b n -> s b n', s=n_steps)
        additional_indices = torch.stack([torch.cat([context_indices, idx[:, (i+1)*n_masked_elem:], idx[:, :i*n_masked_elem]], 1) for i, idx in enumerate(indices)])
        target_indices = torch.stack([idx[:, i*n_masked_elem:(i+1)*n_masked_elem] for i, idx in enumerate(indices)])
        return additional_indices, target_indices

    @staticmethod
    def create_gibbs_draft_mask(context_indices, target_indices, num_unit_gibbs_steps, device):
        B = target_indices.shape[0]
        N = np.prod(target_indices.shape[1:])
        n_steps = num_unit_gibbs_steps  # for brevity in local

        def is_power_of_two(num):
            return num and not (num & (num - 1))

        # assert is_power_of_two(N)
        assert N % n_steps == 0

        n_masked_elem = N // n_steps
        rand_indices = torch.stack([torch.randperm(N) for _ in range(B)]).to(device)
        target_indices = torch.gather(target_indices, 1, rand_indices)
        indices = repeat(target_indices, 'b n -> s b n', s=n_steps)
        additional_indices = [torch.cat([context_indices, idx[:, :i*n_masked_elem]], 1) for i, idx in enumerate(indices)]
        target_indices = [idx[:, i*n_masked_elem:] for i, idx in enumerate(indices)]
        return additional_indices, target_indices
