import os

from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict as edict
import yaml

from models.transformer3d_simplified.configs import Transformer3dSimplifiedConfig
from models.transformer3d_bart.configs import Transformer3dBARTConfig
from models.transformer3d_maskgit.configs import Transformer3dMaskGITConfig
from models.transformer3d_bitvig.configs import Transformer3dBiTViGConfig


def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config


def is_stage1_arch(arch_type):
    return not ('iGPT' in arch_type or 'transformer' in arch_type)


def augment_arch_defaults(arch_config):

    if arch_config.type == 'vqgan_original':
        arch_defaults = OmegaConf.create(
            {
                'ema': None,
                'hparams': {
                    'loss_type': 'l1',
                    'restart_unused_codes': False,
                    'use_padding_idx': False,
                    'masked_dropout': 0.0,
                },
                'checkpointing': False,
            }
        )
    elif is_stage1_arch(arch_config.type):
        arch_defaults = OmegaConf.create(
            {
                'ema': None,
                'hparams': {
                    'loss_type': 'mse',
                    'n_mix_dmol': None,
                    'use_conv_transpose': True,
                    'restart_unused_codes': False,
                    'use_padding_idx': False,
                    'masked_dropout': 0.0,
                    'commitment_loss': 'cumsum',
                },
            }
        )

        if arch_config.hparams.get('loss_type', 'mse') == 'dmol':
            del arch_defaults.hparams.loss_type
            arch_defaults.hparams.n_mix_dmol = 10

    elif arch_config.type == 'transformer3d-simplified':
        arch_defaults = Transformer3dSimplifiedConfig.create(arch_config)
    elif arch_config.type == 'transformer3d-bart':
        arch_defaults = Transformer3dBARTConfig.create(arch_config)
    elif arch_config.type == 'transformer3d-maskgit':
        arch_defaults = Transformer3dMaskGITConfig.create(arch_config)
    elif arch_config.type == 'transformer3d-bitvig':
        arch_defaults = Transformer3dBiTViGConfig.create(arch_config)
    else:
        arch_defaults = OmegaConf.create(
            {
                'ema': None,
                'ar_hierarchy': None,
                'cond_ver': None,
                'cond_size': None,
                'use_padding_emb': False,
                'input_emb_vqvae': False,
                'head_emb_vqvae': False,
                'scaled_head_emb_vqvae': False,
                'cumsum_depth_ctx': False,
            }
        )

    return OmegaConf.merge(arch_defaults, arch_config)


def augment_optimizer_defaults(optim_config):

    defaults = OmegaConf.create(
        {
            'type': 'adamW',
            'max_gn': None,
            'warmup': {
                'mode': 'linear',
                'start_from_zero': (True if optim_config.warmup.epoch > 0 else False),
            },
        }
    )
    return OmegaConf.merge(defaults, optim_config)


def augment_defaults(config):

    defaults = OmegaConf.create(
        {
            'arch': augment_arch_defaults(config.arch),
            'dataset': {
                'transform': {'type': None},
            },
            'optimizer': augment_optimizer_defaults(config.optimizer),
            'experiment': {
                'test_freq': 10,
                'amp': False,
            },
        }
    )

    if 'gan' in config:
        gan_defaults = OmegaConf.merge(defaults.optimizer, config.gan.disc.get('optimizer', {}))
        defaults.gan = OmegaConf.create(
            {
                'disc': {'optimizer': gan_defaults},
            }
        )

    if not is_stage1_arch(config.arch.type):

        model_aux_path = config.vqvae.ckpt
        model_aux_config_path = os.path.join(os.path.dirname(model_aux_path), 'config.yaml')
        stage1_arch_config = load_config(model_aux_config_path).arch

        config.vqvae = stage1_arch_config
        config.vqvae.ckpt = model_aux_path
        # OmegaConf.set_readonly(config.vqvae, True)

        defaults.vqvae = augment_arch_defaults(config.vqvae)
        defaults.arch.vocab_size = config.dataset.vocab_size
        defaults.experiment.sample = {'top_k': None, 'top_p': None, 'strategy': 'maskgit'}

        if config.get('loss', {}).get('type', '') == 'soft_target_cross_entropy':
            defaults.loss = {'temp': 1.0, 'stochastic_codes': False}
        else:
            defaults.loss = {'type': 'cross_entropy', 'temp': 1.0, 'stochastic_codes': False}

    config = OmegaConf.merge(defaults, config)

    return config


def augment_dist_defaults(config, distenv):
    config = config.copy()

    local_batch_size = config.experiment.batch_size
    world_batch_size = distenv.world_size * local_batch_size
    total_batch_size = config.experiment.get('total_batch_size', world_batch_size)

    if total_batch_size % world_batch_size != 0:
        raise ValueError('total batch size must be divisible by world batch size')
    else:
        grad_accm_steps = total_batch_size // world_batch_size

    config.optimizer.grad_accm_steps = grad_accm_steps
    config.experiment.total_batch_size = total_batch_size

    return config


def config_setup(args, distenv, config_path, extra_args=()):

    if args.eval:
        config = load_config(config_path)
        config = augment_defaults(config)

        if hasattr(args, 'test_batch_size'):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, 'seed'):
            config.seed = args.seed

    elif args.resume:
        config = load_config(config_path)
        if distenv.world_size != config.runtime.distenv.world_size:
            raise ValueError("world_size not identical to the resuming config")
        config.runtime = {'args': vars(args), 'distenv': distenv}

    else:  # training
        config_path = args.model_config
        config = load_config(config_path)

        extra_config = OmegaConf.from_dotlist(extra_args)
        config = OmegaConf.merge(config, extra_config)

        config = augment_defaults(config)
        config = augment_dist_defaults(config, distenv)

        config.seed = args.seed
        config.runtime = {'args': vars(args), 'extra_config': extra_config, 'distenv': distenv}

    return config

