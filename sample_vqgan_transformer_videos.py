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
def bidirect_sample(model, batch_size, total_length, step_size, context_size,
        temperature=1.0, top_k=None, top_p=None,
        frame_n_steps=8, vid_n_steps=8, frame_c_temp=4.5, vid_c_temp=4.5,
        no_phase=False, ctemp_schedule='linear', strategy='maskgit',
        bootstrap=0):
    # TODO: first frame decoding -> 16frame decoding -> shift decoding
    T, H, W = model.mask_sampler.shape[-3:]
    ratio = 0.25
    step_size = int(step_size * ratio)
    context_size = int(context_size * ratio)
    shape = (batch_size, step_size, H, W)
    c_indices = repeat(torch.tensor([0]), '1 -> b 1', b=batch_size).to(model.device)
    log = dict()
    log['samples'] = []
    code_map = []

    x = torch.zeros(shape, dtype=torch.long, device=model.device)
    context_indices = None
    target_indices = None
    if bootstrap > 0:
        x, context_indices, target_indices, _, _, bs_partial_probs = model.sample(x, None, 1., None, None, bootstrap, context_indices, target_indices, context_temperature=vid_c_temp, skips=False, ctemp_schedule=ctemp_schedule, strategy='bootstrap', debug=True)
    else:
        bs_partial_probs=None
    x, context_indices, _, _, _, final_partial_probs = model.sample(x, None, temperature, top_k, top_p, vid_n_steps, context_indices, target_indices, context_temperature=vid_c_temp, skips=False, ctemp_schedule=ctemp_schedule, strategy=strategy, debug=True)
    curr_t = step_size
    
    # decode to images and stack.
    vq_x = x.reshape(shape)
    code_map.append(vq_x)
    log["class_label"] = c_indices

    # Remaining decoding
    while True:
        if curr_t >= (total_length * ratio):
            break
        # save_memory by forgetting the past
        new_x = torch.zeros(shape, dtype=torch.long, device=model.device)
        new_x[:, :context_size, :, :] = vq_x[:, -context_size:, :, :]
        x = new_x
        context_indices = torch.stack([torch.arange(H*W*context_size) for _ in range(batch_size)]).to(model.device)
        target_indices = torch.stack([torch.arange((step_size-context_size) * H*W) for _ in range(batch_size)]).to(model.device)
        target_indices = target_indices + H*W * context_size
        x = model.sample(x, None, temperature, top_k, top_p, vid_n_steps, context_indices, target_indices, context_temperature=vid_c_temp, skips=False, ctemp_schedule=ctemp_schedule, strategy=strategy)[0]
        
        # decode to images and stack.
        vq_x = x.reshape(shape)
        vq_new = vq_x[:, context_size:, :, :]
        code_map.append(vq_new)
        curr_t += step_size - context_size
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
    log["samples"] = torch.clamp(img_x, -0.5, 0.5) + 0.5
    log['samples'] = log['samples'][:, :, :total_length, :, :]
    if bs_partial_probs is not None:
        final_prob_map = torch.where(final_partial_probs < 0., bs_partial_probs, final_partial_probs)
    else:
        final_prob_map = final_partial_probs
    selected_prob_map = torch.gather(final_prob_map, -1, code_map.view(batch_size, -1, 1)).squeeze(-1)
    score = selected_prob_map.log().sum(-1)
    log['score'] = score

    return log

@torch.no_grad()
def extrapolate(model, vq_input, total_length, step_size, context_size,
        temperature=1.0, top_k=None, top_p=None,
        frame_n_steps=8, vid_n_steps=8, frame_c_temp=4.5, vid_c_temp=4.5,
        no_phase=False, ctemp_schedule='linear', strategy='maskgit',
        bootstrap=0):
    B, T, H, W = vq_input.shape
    batch_size = B
    ratio = 0.25
    step_size = int(step_size * ratio)
    context_size = int(context_size * ratio)
    assert T == step_size

    total_size = int(total_length * ratio)
    jump_size = step_size - context_size
    n_jumps = int(np.ceil((total_size - step_size) / jump_size))

    shape = (B, step_size, H, W)
    c_indices = repeat(torch.tensor([0]), '1 -> b 1', b=batch_size).to(model.device)
    log = dict()
    log['samples'] = []

    curr_t = step_size
    
    # decode to images and stack.
    '''
    code_length = step_size + jump_size * n_jumps
    code_map = torch.zeros(B, code_length, H, W).long().to(model.device)
    code_map[:, :step_size, :, :] = vq_input.clone()
    '''
    code_map = [vq_input.clone()]
    log["class_label"] = c_indices

    # Remaining decoding
    indices = torch.stack([torch.arange(H*W*step_size) for _ in range(batch_size)]).to(model.device)
    indices = rearrange(indices, 'b (t h w) -> b t h w', h=H, w=W)
    context_indices = indices[:, :context_size].view(B, -1)
    target_indices = indices[:, context_size:].view(B, -1)

    x = vq_input
    for j in range(n_jumps):
        # save_memory by forgetting the past
        vq_input = torch.zeros_like(x)
        vq_input[:, :context_size] = code_map[-1][:, -context_size:]

        x = model.sample(vq_input.view(B, -1), None, temperature, top_k, top_p, vid_n_steps, context_indices, target_indices, context_temperature=vid_c_temp, skips=False, edit=True)[0]
        
        # decode to images and stack.
        x = rearrange(x, 'b (t h w) -> b t h w', h=H, w=W)
        code_map.append(x.clone()[:, context_size:])
    code_map = torch.cat(code_map, 1)
    try:
        img_x = model.first_stage_model.decode(code_map)
    except RuntimeError:
        img_x = []
        for i in range(code_map.shape[0]):
            img_x.append(model.first_stage_model.decode(code_map[i:i+1]))
        img_x = torch.cat(img_x, 0)
    log['code_maps'] = code_map
    log["samples"] = torch.clamp(img_x, -0.5, 0.5) + 0.5
    log['samples'] = log['samples'][:, :, :total_length, :, :]

    return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--base', nargs='*', metavar="base_config.yaml")
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--gpt_ckpt', type=str, default='')
    parser.add_argument('--base_np', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--save', type=str, default='./results/mebt')
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--frame_c_temp', type=float, default=4.5)
    parser.add_argument('--vid_c_temp', type=float, default=1.0)
    parser.add_argument('--frame_n_steps', type=int, default=16)
    parser.add_argument('--vid_n_steps', type=int, default=128)
    parser.add_argument('--total_length', type=int, default=32)
    parser.add_argument('--context_size', type=int, default=12)
    parser.add_argument('--step_size', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--n_sample', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='mshapes', choices=['ucf101', 'stl', 'taichi', 'mshapes'])
    parser.add_argument('--format', type=str, default='gif', choices=['webp', 'mp4', 'gif', 'avi'])
    parser.add_argument('--save_videos', action='store_true')
    parser.add_argument('--save_n', type=int, default=5)
    parser.add_argument('--save_codemap', action='store_true')
    parser.add_argument('--no_np', action='store_true')
    parser.add_argument('--no_phase', action='store_true')
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--schedule', type=str, default='cosine')
    parser.add_argument('--decoding_strategy', type=str, default='maskgit', choices=['maskgit', 'random', 'ar'])
    parser.add_argument('--ctemp_schedule', type=str, default='linear', choices=['linear', 'constant', 'cosine'])
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
    os.makedirs(args.save, exist_ok=True)

    config.model.params.class_cond_dim = None
    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=None).cuda().eval()
    gpt.mask_sampler.schedule = args.schedule

    save_dir = f'{args.save}/videos_{args.total_length}/{args.dataset}/VID_n_steps{args.vid_n_steps}'
    save_np = f'{args.save}/numpy_files_{args.total_length}/{args.dataset}/VID_n_steps{args.vid_n_steps}'
    if args.top_k is not None:
        save_dir += f'_k{args.top_k}'
        save_np += f'_k{args.top_k}'
    if args.top_p is not None:
        save_dir += f'_p{args.top_p}'
        save_np += f'_p{args.top_p}'

    save_dir += f'_temp{args.temp}_ctemp{args.vid_c_temp}{args.ctemp_schedule}_{args.decoding_strategy}_{args.schedule}'
    save_np += f'_temp{args.temp}_ctemp{args.vid_c_temp}{args.ctemp_schedule}_{args.decoding_strategy}_{args.schedule}'

    if not args.no_phase:
        assert 0
        print("Warning: generation the first frame first.")
    if args.no_phase:
        save_dir += f'_no_phase'
        save_np += f'_no_phase'

    save_dir += f'_run{args.run}'
    save_np += f'_run{args.run}'

    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_code = []
    all_scores = []
    n_row = min(int(np.sqrt(args.batch_size)), 4)
    n_batch = args.n_sample//args.batch_size+1
    with torch.no_grad():
        if args.base_np == '':
            for sample_id in tqdm.tqdm(range(n_batch)):
                logs = bidirect_sample(gpt, args.batch_size, total_length=args.total_length,
                        step_size=args.step_size, context_size=args.context_size, temperature=args.temp,
                        top_k=args.top_k, top_p=args.top_p,
                        frame_n_steps=args.frame_n_steps, vid_n_steps=args.vid_n_steps, frame_c_temp=args.frame_c_temp, vid_c_temp=args.vid_c_temp,
                        no_phase=args.no_phase, ctemp_schedule=args.ctemp_schedule, strategy=args.decoding_strategy, bootstrap=args.bootstrap)
                if args.save_videos:
                    if sample_id < args.save_n:
                        save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d.%s'%(sample_id, args.format)), n_row)
                all_data.append(logs['samples'].cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128
                all_code.append(logs['code_maps'].cpu().data.numpy())
        else:
            vq_np = np.load(args.base_np)
            for sample_id in tqdm.tqdm(range(n_batch)):
                vq_x = torch.tensor(vq_np[sample_id*args.batch_size:(sample_id+1)*args.batch_size]).long().cuda()
                logs = extrapolate(gpt, vq_x, total_length=args.total_length,
                        step_size=args.step_size, context_size=args.context_size, temperature=args.temp,
                        top_k=args.top_k, top_p=args.top_p,
                        frame_n_steps=args.frame_n_steps, vid_n_steps=args.vid_n_steps, frame_c_temp=args.frame_c_temp, vid_c_temp=args.vid_c_temp,
                        no_phase=args.no_phase, ctemp_schedule=args.ctemp_schedule, strategy=args.decoding_strategy, bootstrap=args.bootstrap)
                if args.save_videos:
                    if sample_id < args.save_n:
                        save_video_grid(logs['samples'], os.path.join(save_dir, 'generation_%d.%s'%(sample_id, args.format)), n_row, fps=30)
                all_data.append(logs['samples'].cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128
                all_code.append(logs['code_maps'].cpu().data.numpy())

    if args.save_codemap:
        print('saving code_map numpy file to %s...'%save_np+'_codemap')
        os.makedirs(os.path.dirname(save_np), exist_ok=True)
        all_code_np = np.concatenate(all_code, 0)
        np.save(save_np+'_codemap', all_code_np[:args.n_sample])

    if not args.no_np:
        print('saving numpy file to %s...'%save_np)
        os.makedirs(os.path.dirname(save_np), exist_ok=True)
        all_data_np = np.array(all_data)
        all_data_np = np.transpose(all_data_np.reshape(-1, 3, args.total_length, resolution, resolution), (0, 2, 3, 4, 1)) # B T H W C
        n_total = all_data_np.shape[0]
        all_data_np = (all_data_np*255).astype(np.uint8)[np.random.permutation(n_total)[:args.n_sample]]
        np.save(save_np, all_data_np)
        
        '''
        all_score_np = np.concatenate(all_scores, 0)
        np.save(save_np+'score', all_score_np[:args.n_sample])
        '''
