# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import tqdm
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from einops import repeat

from mebt import VideoData, Net2NetTransformer, load_transformer, load_vqgan
from mebt.utils import save_video_grid
from mebt.data import preprocess
from mebt.utils import shift_dim

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser.add_argument('--np_file', type=str, default='')
parser.add_argument('--score_file', type=str, default='')
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--n_neighbor', type=int, default=5)
parser.add_argument('--dataset', type=str, default='mshapes', choices=['mshapes', 'ucf101', 'sky', 'taichi'])
parser.add_argument('--compute_fvd', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--sample_fake_n_frames', type=int, default=1)
args = parser.parse_args()
print(args)

save_np = args.np_file
if args.score_file != '':
    score_np = np.load(args.score_file)
else:
    score_np = []
args.batch_size = 32
print('loading numpy file from %s...'%save_np)
all_data_np = np.load(save_np)
# TODO: sort-by scores if score is given.
if len(score_np) != 0:
    # sort in ascending order.
    indices = np.argsort(score_np[:len(all_data_np)])
    indices = indices[-args.n_sample:]
    all_data_np = all_data_np[indices, :]

from mebt.fvd.fvd import FVD_SAMPLE_SIZE, MAX_BATCH, get_fvd_logits, frechet_distance, \
    load_fvd_model, preprocess, TARGET_RESOLUTION, polynomial_mmd
device = torch.device('cuda')
i3d = load_fvd_model(device)
data = VideoData(args, True)
loader = data.train_dataloader() if args.train else data.val_dataloader()
real_embeddings = []
print('computing fvd embeddings for real videos')
while True:
    for batch in tqdm.tqdm(loader):
        # images from the loader have a value b/w -1~1
        if batch['video'].shape[0] % MAX_BATCH == 0:
            real_embeddings.append(get_fvd_logits(shift_dim((batch['video']+0.5)*255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        if len(real_embeddings)*args.batch_size >=args.n_sample: break
    if len(real_embeddings)*args.batch_size >=args.n_sample: break
print('concat fvd embeddings for real videos')
real_embeddings = torch.cat(real_embeddings, 0)[:args.n_sample]
print('computing fvd embeddings for fake videos')
fake_embeddings = []
n_batch = all_data_np.shape[0]//args.batch_size
while True:
    for i in range(n_batch):
        if all_data_np.shape[1] != (args.sequence_length * args.sample_fake_n_frames):
            length = args.sequence_length * args.sample_fake_n_frames
            T = all_data_np.shape[1]
            start_t = random.randint(0, T-length)
            end_t = start_t + length
            fake_embeddings.append(get_fvd_logits(all_data_np[i*args.batch_size:(i+1)*args.batch_size, start_t:end_t:args.sample_fake_n_frames], i3d=i3d, device=device))
        else:
            fake_embeddings.append(get_fvd_logits(all_data_np[i*args.batch_size:(i+1)*args.batch_size], i3d=i3d, device=device))
        if len(fake_embeddings)*args.batch_size >=args.n_sample: break
    if len(fake_embeddings)*args.batch_size >=args.n_sample: break
print('concat fvd embeddings for fake videos')
fake_embeddings = torch.cat(fake_embeddings, 0)[:args.n_sample]
fvd = frechet_distance(fake_embeddings, real_embeddings).cpu().numpy()
kvd = polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())
print('FVD = %.2f'%(fvd))
print('KVD = %.2f'%(kvd))
df = {"FVD":[fvd], "KVD":[kvd]}
df = pd.DataFrame(df)
df.to_csv(save_np.replace('.npy', f'_consq_set_{args.n_neighbor}.csv'))
