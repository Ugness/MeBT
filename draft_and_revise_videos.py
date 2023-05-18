# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import tqdm
import time
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from mebt import VideoData, Net2NetTransformer, load_vqgan, load_transformer
from mebt.utils import save_video_grid
from mebt.data import preprocess
from mebt.utils import shift_dim
from omegaconf import OmegaConf
import math
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
from glob import glob
import random
from einops import repeat, rearrange

@torch.no_grad()
def sample(model, batch_size, total_length, step_size, context_size,
        n_draft, draft_t, draft_k, draft_p, n_revise, revise_t, revise_k, revise_p, M,
        draft=None
        ):
    assert total_length == step_size
    T, H, W = model.mask_sampler.shape[-3:]
    ratio = 0.25
    step_size = int(step_size * ratio)
    context_size = int(context_size * ratio)
    shape = (batch_size, step_size, H, W)
    c_indices = repeat(torch.tensor([0]), '1 -> b 1', b=batch_size).to(model.device)
    log = dict()
    log['samples'] = []
    code_map = []

    skip_draft = draft is not None
    if skip_draft:
        x = torch.tensor(draft, dtype=torch.long, device=model.device)
    else:
        x = torch.zeros(shape, dtype=torch.long, device=model.device)
    x = model.draft_and_revise(x, None, n_draft, draft_t, draft_k, draft_p, n_revise, revise_t, revise_k, revise_p, M, skip_draft)
    vq_x = x.reshape(shape)
    code_map.append(vq_x)

    code_map = torch.cat(code_map, 1)
    if code_map.shape[1] == 1:
        code_map = code_map.expand(-1, 4, H, W)
    try:
        img_x = model.first_stage_model.decode(code_map)
    except RuntimeError:
        img_x = []
        for i in range(code_map.shape[0]):
            img_x.append(model.first_stage_model.decode(code_map[i:i+1]))
        img_x = torch.cat(img_x, 0)
    log['code_maps'] = code_map
    log["class_label"] = c_indices
    log["samples"] = torch.clamp(img_x, -0.5, 0.5) + 0.5
    log['samples'] = log['samples'][:, :, :total_length, :, :]
    return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--base', nargs='*', metavar="base_config.yaml")
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--gpt_ckpt', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--n_draft', type=int, default=8)
    parser.add_argument('--draft_t', type=float, default=1.0)
    parser.add_argument('--draft_p', type=float, default=None)
    parser.add_argument('--draft_k', type=int, default=None)
    parser.add_argument('--n_revise', type=int, default=8)
    parser.add_argument('--revise_t', type=float, default=1.0)
    parser.add_argument('--revise_p', type=float, default=None)
    parser.add_argument('--revise_k', type=int, default=None)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--np_draft', type=str, default=None)

    parser.add_argument('--save', type=str, default='./results/mebt')
    parser.add_argument('--total_length', type=int, default=16)
    parser.add_argument('--context_size', type=int, default=12)
    parser.add_argument('--step_size', type=int, default=16)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='mshapes', choices=['ucf101', 'stl', 'taichi', 'mshapes'])
    parser.add_argument('--format', type=str, default='gif', choices=['webp', 'mp4', 'gif', 'avi'])
    parser.add_argument('--save_videos', action='store_true')
    parser.add_argument('--save_n', type=int, default=5)
    parser.add_argument('--save_codemap', action='store_true')
    parser.add_argument('--no_np', action='store_true')
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args, unknown = parser.parse_known_args()

    if args.default_root_dir is None:
        args.default_root_dir = ''

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    resolution = config.data.resolution if config.data.image_folder else args.resolution

    ver = args.exp_name
    args.save = f'results/{ver}'
    if args.gpt_ckpt=='':
        if not args.latest:
            args.gpt_ckpt = glob(f'logs/{ver}/lightning_logs/version_0/checkpoints/best_checkpoint.ckpt')[0]
        else:
            ckpts = glob(f'logs/{ver}/lightning_logs/version_0/checkpoints/*/loss=*.ckpt')
            iters = [int(ckpt.split('step=')[-1].split('-train')[0]) for ckpt in ckpts]
            max_iter = max(iters)
            args.gpt_ckpt = glob(f'logs/{ver}/lightning_logs/version_0/checkpoints/*step={max_iter}-train/loss=*.ckpt')[0]
            args.save += '_latest'
    print(args.gpt_ckpt)

    if args.np_draft is not None:
        draft = np.load(args.np_draft)
        postfix = ''
        if 'n_steps' in args.np_draft:
            args.n_draft = int(args.np_draft.split('VID_n_steps')[-1].split('_')[0])
        else:
            args.n_draft = 0
        if 'maskgit_cosine' in args.np_draft:
            ctemp = float(args.np_draft.split('ctemp')[-1].split('_')[0][:3])
            postfix += f'_ctemp{ctemp}'

        args.draft_t = 0.0
        args.draft_p = None
        args.draft_k = None
    else:
        draft = None
    os.makedirs(args.save, exist_ok=True)

    config.model.params.class_cond_dim = None
    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=None).cuda().eval()

    save_dir = f'{args.save}/videos_{args.total_length}/{args.dataset}/VID_dnr_nd{args.n_draft}_dt{args.draft_t}_nr{args.n_revise}_rt{args.revise_t}_M{args.M}' + postfix
    save_np = f'{args.save}/numpy_files_{args.total_length}/{args.dataset}/VID_dnr_nd{args.n_draft}_dt{args.draft_t}_nr{args.n_revise}_rt{args.revise_t}_M{args.M}' + postfix

    if args.draft_p is not None:
        save_dir += '_dp{args.draft_p}'
        save_np += '_dp{args.draft_p}'
    if args.draft_k is not None:
        save_dir += '_dk{args.draft_k}'
        save_np += '_dk{args.draft_k}'
    if args.revise_p is not None:
        save_dir += '_rp{args.revise_p}'
        save_np += '_rp{args.revise_p}'
    if args.revise_k is not None:
        save_dir += '_rk{args.revise_k}'
        save_np += '_rk{args.revise_k}'

    save_dir += f'_run{args.run}'
    save_np += f'_run{args.run}'

    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_code = []
    n_row = int(np.sqrt(args.batch_size))
    n_batch = args.n_sample//args.batch_size + min(1, args.n_sample % args.batch_size)
    with torch.no_grad():
        for sample_id in tqdm.tqdm(range(n_batch)):
            draft_batch = None if draft is None else draft[sample_id * args.batch_size: (sample_id+1) * args.batch_size]
            logs = sample(gpt, args.batch_size, total_length=args.total_length,
                    step_size=args.step_size, context_size=args.context_size,
                    n_draft=args.n_draft, draft_t=args.draft_t, draft_k=args.draft_k, draft_p=args.draft_p,
                    n_revise=args.n_revise, revise_t=args.revise_t, revise_k=args.revise_k, revise_p=args.revise_p, M=args.M,
                    draft=draft_batch,
                    )
            if args.save_videos:
                if sample_id < args.save_n:
                    save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d.%s'%(sample_id, args.format)), n_row)
            all_data.append(logs['samples'].cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128
            all_code.append(logs['code_maps'].cpu().data.numpy())

    if args.save_codemap:
        print('saving code_map numpy file to %s...'%save_np)
        os.makedirs(os.path.dirname(save_np), exist_ok=True)
        all_code_np = np.concatenate(all_code, 0)
        np.save(save_np+'_codemap', all_code_np)

    if args.np_draft is not None:
        with open(save_np+'.txt', 'w') as f:
            f.write(args.np_draft)

    if not args.no_np:
        print('saving numpy file to %s...'%save_np)
        os.makedirs(os.path.dirname(save_np), exist_ok=True)
        all_data_np = np.array(all_data)
        all_data_np = np.transpose(all_data_np.reshape(-1, 3, args.total_length, resolution, resolution), (0, 2, 3, 4, 1))
        n_total = all_data_np.shape[0]
        all_data_np = (all_data_np*255).astype(np.uint8)[np.random.permutation(n_total)[:args.n_sample]]
        np.save(save_np, all_data_np)
