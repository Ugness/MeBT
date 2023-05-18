# Towards End-to-End Generative Modeling of Long Videos with Memory-Efficient Bidirectional Transformers (CVPR 2023)

This repository is an official implementation of the paper:  
**Towards End-to-End Generative Modeling of Long Videos with Memory-Efficient Bidirectional Transformers (CVPR 2023)**  
Jaehoon Yoo, Semin Kim, Doyup Lee, Chiheon Kim, Seunghoon Hong  
[Project Page](https://sites.google.com/view/mebt-cvpr2023/home) | [Paper](https://arxiv.org/abs/2303.11251)

![](readme_figs/stl.gif)
![](readme_figs/taichi.gif)
![](readme_figs/ucf.gif)

## Setup
We installed the packages specified in `requirements.txt` based on [this docker image](https://hub.docker.com/layers/pytorch/pytorch/1.10.0-cuda11.3-cudnn8-devel/images/sha256-913e6689c5958b187e65561e528ec6c3ce8a02deedcdd38cb50c9cab301907bb?context=explore) 
```
docker pull pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
docker run -it --shm-size=24G pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime /bin/bash
git clone https://github.com/Ugness/MeBT
mv MeBT
pip install requirements.txt
```

## Datasets
### Download
* [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)
* [Sky-Timelapse](https://github.com/weixiong-ur/mdgan)
* [Taichi](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md)
### Data preprocessing & setup
1. Extract all frames in each video. The filename should be `[VIDEO_ID]_[FRAME_NUM].[png, jpg, ...]`
2. create `train.txt` and `test.txt` containing the directory of the entire frames. For example,
```
find $(pwd)/dataset/train -name "*.png" >> 'train.txt'
find $(pwd)/dataset/test -name "*.png" >> 'test.txt'
```
3. The txt files should be located as `[DATA_PATH]/train.txt` and `[DATA_PATH]/test.txt`.

## Checkpoints
1. VQGAN: Checkpoints for the 3d VQGAN can be found [here](https://github.com/SongweiGe/TATS#datasets-and-trained-models).
2. MeBT: Checkpoints for MeBT can be found [here](https://drive.google.com/drive/folders/1BXo1ABfM2FY_3nRVfsre9MYb2Uaa8cyx?usp=sharing)

## Configuration Files
You may control the experiments with a configuration files.
The default configuration files can be found in the `configs` folder.

Here is an example of the config file.
```yaml
model:
    target: mebt.transformer.Net2NetTransformer
    params:
        unconditional: True
        vocab_size: 16384 # You should follow the vocab_size of 3d VQGAN.
        first_stage_vocab_size: 16384
        block_size: 1024 # total number of input tokens (output of 3d VQGAN.)
        n_layer: 24 # number of layers for MeBT
        n_head: 16 # number of attention heads
        n_embd: 1024 # hidden dimension
        n_unmasked: 0
        embd_pdrop: 0.1 # Dropout ratio
        resid_pdrop: 0.1 # Dropout ratio
        attn_pdrop: 0.1 # Dropout ratio
        sample_every_n_latent_frames: 0
        first_stage_key: video # ignore
        cond_stage_key: label # ignore
        vtokens: False # ignore
        vtokens_pos: False # ignore
        vis_epoch: 100
        sos_emb: 256 # Number of latent tokens.
        avg_loss: True
        mode: # You may stack different type of layers. The total number of layers should be matched with n_layer
            - latent_enc
            - latent_self
            - latent_enc
            - latent_self
            - latent_enc
            - latent_self
            - latent_enc
            - latent_self
            - latent_enc
            - latent_self
            - latent_enc
            - latent_self
            - latent_enc
            - latent_dec
            - lt2l
            - latent_dec
            - lt2l
            - latent_dec
            - lt2l
            - latent_dec
            - lt2l
            - latent_dec
            - lt2l
            - latent_dec
    mask:
        target: mebt.mask_sampler.MaskGen
        params:
            iid: False
            schedule: linear
            max_token: 1024 # total number of input tokens (output of 3d VQGAN.)
            method: 'mlm'
            shape: [4, 16, 16] # shape of the output of 3d VQGAN. (T, H, W)
            t_range: [0.0, 1.0]
            budget: 1024 # total number of input tokens (output of 3d VQGAN.)

    vqvae:
        params:
            ckpt_path: 'ckpts/vqgan_sky_128_488_epoch=12-step=29999-train.ckpt' # Path to the 3d VQGAN checkpoint.
            ignore_keys: ['loss']

data:
    data_path: 'datasets/vqgan_data/stl_128' # [DATA_PATH]
    sequence_length: 16 # Length of the training video (in frames)
    resolution: 128 # Resolution of the training video (in pixels)
    batch_size: 6 # Batch_size per GPU
    num_workers: 8
    image_channels: 3
    smap_cond: 0
    smap_only: False
    text_cond: False
    vtokens: False
    vtokens_pos: False
    spatial_length: 0
    sample_every_n_frames: 1
    image_folder: True
    stft_data: False

exp:
    exact_lr: 1.08e-05 # learning rate
```

## Training
The scripts for training can be found in `scripts` folder. You may excute the script as following:
```bash
bash scripts/train_config_log_gpus.sh [CONFIG_FILE] [LOG_DIR] [GPU_IDs]
```
* [GPU_IDs]:
  * 0, : use GPU_ID 0 only.
  * 0,1,2,3,4,5,6,7 : use 8 GPUs from 0 to 7.
## Inference
The scripts for inference can be found in `scripts` folder. You may excute the script as following:
```bash
bash scripts/valid_dnr_config_ckpt_exp_[stl, taichi, ucf]_[16f, 128f].sh [CONFIG_FILE] [CKPT_PATH] [SAVE_DIR]
```
* You should change the [DATA_PATH] in the script file to measure FVD and KVD.

## Acknowledgements
* Our code is based on [VQGAN](https://github.com/CompVis/taming-transformers) and [TATS](https://github.com/SongweiGe/TATS).
* The development of this open-sourced code was supported in part by the National Research Foundation of Korea (NRF) (No. 2021R1A4A3032834).

## Citation
```
@article{yoo2023mebt,
         title={Towards End-to-End Generative Modeling of Long Videos with Memory-Efficient Bidirectional Transformers},
         author={Jaehoon Yoo, Semin Kim, Doyup Lee, Chiheon Kim, Seunghoon Hong},
         journal={arXiv preprint arXiv:2303.11251},
         year={2023}
}
```
