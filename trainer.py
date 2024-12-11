#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime
import numpy as np
from box import Box
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union

from datapipe.datasets import create_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image
from utils.util_ops import append_dims

import pyiqa
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

from diffusers import EulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps

_base_seed = 10**6
_INTERPOLATION_MODE = 'bicubic'
_Latent_bound = {'min':-10.0, 'max':10.0}
_positive= 'Cinematic, high-contrast, photo-realistic, 8k, ultra HD, ' +\
           'meticulous detailing, hyper sharpness, perfect without deformations'
_negative= 'Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,' +\
           'painting, drawing, sketch, oil painting'

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.get('global_seeding', False)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loading checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            if self.configs.train.loss_coef.get('ldis', 0) > 0:
                util_net.reload_model(self.discriminator, ckpt['state_dict_dis'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loading EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                util_net.reload_model(self.ema_model, ema_ckpt)
                torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")
            if self.configs.get('discriminator', None) is not None:
                if "amp_scaler_dis" in ckpt:
                    self.amp_scaler_dis.load_state_dict(ckpt["amp_scaler_dis"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler (discriminator) from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = torch.amp.GradScaler('cuda') if self.configs.train.use_amp else None

        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

        if self.configs.train.loss_coef.get('ldis', 0) > 0:
            self.optimizer_dis = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.configs.train.lr_dis,
                    weight_decay=self.configs.train.weight_decay_dis,
                        )
            self.amp_scaler_dis = torch.amp.GradScaler('cuda') if self.configs.train.use_amp else None

    def prepare_compiling(self):
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#stable-diffusion-3
        if not hasattr(self, "prepare_compiling_well") or (not self.prepare_compiling_well):
            torch.set_float32_matmul_precision("high")
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            self.prepare_compiling_well = True

    def build_model(self):
        if self.configs.train.get("compile", True):
            self.prepare_compiling()

        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if not self.configs.train.start_mode:   # Loading the starting model for evaluation
            self.start_model = deepcopy(model)
            assert self.configs.model.ckpt_start_path is not None
            ckpt_start_path = self.configs.model.ckpt_start_path
            if self.rank == 0:
                self.logger.info(f"Loading the starting model from {ckpt_start_path}")
            ckpt = torch.load(ckpt_start_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.start_model, ckpt)
            self.freeze_model(self.start_model)
            self.start_model.eval()
            # delete the started timestep
            start_timestep = max(self.configs.train.timesteps)
            self.configs.train.timesteps.remove(start_timestep)
            # end_timestep = min(self.configs.train.timesteps)
            # self.configs.train.timesteps.remove(end_timestep)

        # setting the training model
        if self.configs.model.get('ckpt_path', None):   # initialize if necessary
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.model.get("compile", False):
            if self.rank == 0:
                self.logger.info("Compile the model...")
            model.to(memory_format=torch.channels_last)
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
        if self.num_gpus > 1:
            model = DDP(model, device_ids=[self.rank,])  # wrap the network
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model)
            self.freeze_model(self.ema_model)
        self.model = model

        # discriminator if necessary
        if self.configs.train.loss_coef.get('ldis', 0) > 0:
            assert hasattr(self.configs, 'discriminator')
            params = self.configs.discriminator.get('params', dict)
            discriminator = util_common.get_obj_from_str(self.configs.discriminator.target)(**params)
            discriminator.cuda()
            if self.configs.discriminator.get("compile", False):
                if self.rank == 0:
                    self.logger.info("Compile the discriminator...")
                discriminator.to(memory_format=torch.channels_last)
                discriminator = torch.compile(discriminator, mode="max-autotune", fullgraph=False)
            if self.num_gpus > 1:
                discriminator = DDP(discriminator, device_ids=[self.rank,])  # wrap the network
            if self.configs.train.loss_coef.get('ldis', 0) > 0:
                if self.configs.discriminator.enable_grad_checkpoint:
                    if self.rank == 0:
                        self.logger.info("Activating gradient checkpointing for discriminator...")
                    self.set_grad_checkpointing(discriminator)
            self.discriminator = discriminator

        # build the stable diffusion
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        # loading the fp16 robust vae for sdxl: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            params_vae['torch_dtype'] = torch.float16
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            if self.rank == 0:
                self.logger.info(f'Loading improved vae from {pipe_id}...')
            vae_pipe = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(**params_vae)
            if self.rank == 0:
                self.logger.info('Loaded Done')
            params['vae'] = vae_pipe
        if ("StableDiffusion3" in self.configs.sd_pipe.target.split('.')[-1]
            and self.configs.sd_pipe.get("model_quantization", False)):
            if self.rank == 0:
                self.logger.info(f'Loading the quantized transformer for SD3...')
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            params_model = dict(self.configs.model_nf4.params)
            torch_dtype = params_model.pop('torch_dtype')
            params_model['torch_dtype'] = get_torch_dtype(torch_dtype)
            params_model['quantization_config'] = nf4_config
            model_nf4 = util_common.get_obj_from_str(self.configs.model_nf4.target).from_pretrained(
                **params_model
            )
            params['transformer'] = model_nf4
        sd_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            if self.rank == 0:
                self.logger.info(f'Loading scheduler of {pipe_id}...')
            sd_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                sd_pipe.scheduler.config
            )
            if self.rank == 0:
                self.logger.info('Loaded Done')
        if ("StableDiffusion3" in self.configs.sd_pipe.target.split('.')[-1]
            and self.configs.sd_pipe.get("model_quantization", False)):
            sd_pipe.enable_model_cpu_offload(gpu_id=self.rank,device='cuda')
        else:
            sd_pipe.to(f"cuda:{self.rank}")
        # freezing model parameters
        if hasattr(sd_pipe, 'unet'):
            self.freeze_model(sd_pipe.unet)
        if hasattr(sd_pipe, 'transformer'):
            self.freeze_model(sd_pipe.transformer)
        self.freeze_model(sd_pipe.vae)
        # compiling
        if self.configs.sd_pipe.get('compile', True):
            if self.rank == 0:
                self.logger.info('Compile the SD model...')
            sd_pipe.set_progress_bar_config(disable=True)
            if hasattr(sd_pipe, 'unet'):
                sd_pipe.unet.to(memory_format=torch.channels_last)
                sd_pipe.unet = torch.compile(sd_pipe.unet, mode="max-autotune", fullgraph=False)
            if hasattr(sd_pipe, 'transformer'):
                sd_pipe.transformer.to(memory_format=torch.channels_last)
                sd_pipe.transformer = torch.compile(sd_pipe.transformer, mode="max-autotune", fullgraph=False)
            sd_pipe.vae.to(memory_format=torch.channels_last)
            sd_pipe.vae = torch.compile(sd_pipe.vae, mode="max-autotune", fullgraph=True)
        # setting gradient checkpoint for vae
        if self.configs.sd_pipe.get("enable_grad_checkpoint_vae", True):
            if self.rank == 0:
                self.logger.info("Activating gradient checkpointing for VAE...")
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder)
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder)
        # setting gradient checkpoint for diffusion model
        if self.configs.sd_pipe.enable_grad_checkpoint:
            if self.rank == 0:
                self.logger.info("Activating gradient checkpointing for SD...")
            if hasattr(sd_pipe, 'unet'):
                self.set_grad_checkpointing(sd_pipe.unet)
            if hasattr(sd_pipe, 'transformer'):
                self.set_grad_checkpointing(sd_pipe.transformer)
        self.sd_pipe = sd_pipe

        # latent LPIPS loss
        if self.configs.train.loss_coef.get('llpips', 0) > 0:
            params = self.configs.llpips.get('params', dict)
            llpips_loss = util_common.get_obj_from_str(self.configs.llpips.target)(**params)
            llpips_loss.cuda()
            self.freeze_model(llpips_loss)

            # loading the pre-trained model
            ckpt_path = self.configs.llpips.ckpt_path
            self.load_model(llpips_loss, ckpt_path, tag='latent lpips')

            if self.configs.llpips.get("compile", True):
                if self.rank == 0:
                    self.logger.info('Compile the llpips loss...')
                llpips_loss.to(memory_format=torch.channels_last)
                llpips_loss = torch.compile(llpips_loss, mode="max-autotune", fullgraph=True)

            self.llpips_loss = llpips_loss

        # model information
        self.print_model_info()

        torch.cuda.empty_cache()

    def set_grad_checkpointing(self, model):
        if hasattr(model, 'down_blocks'):
            for module in model.down_blocks:
                module.gradient_checkpointing = True
                module.training = True

        if hasattr(model, 'up_blocks'):
            for module in model.up_blocks:
                module.gradient_checkpointing = True
                module.training = True

        if hasattr(model, 'mid_blocks'):
            model.mid_block.gradient_checkpointing = True
            model.mid_block.training = True

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.validate.batch,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            if self.configs.train.get('use_fsdp', False):
                num_params *= self.num_gpus
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

            if hasattr(self, 'discriminator'):
                num_params = util_net.calculate_parameters(self.discriminator) / 1000**2
                self.logger.info(f"Number of parameters in discriminator: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch)
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']), phase='train')

            # training phase
            self.training_step(data)

            # update ema model
            if hasattr(self.configs.train, 'ema_rate') and self.rank == 0:
                self.update_ema_model()

            # validation phase
            if ((ii+1) % self.configs.train.save_freq == 0 and
                'val' in self.dataloaders and
                self.rank == 0
                ):
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0 and self.rank == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.get("warmup_iterations", 0)
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def save_ckpt(self):
        ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
        ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                'state_dict': self.model.state_dict(),
                }
        if self.amp_scaler is not None:
            ckpt['amp_scaler'] = self.amp_scaler.state_dict()
        if self.configs.train.loss_coef.get('ldis', 0) > 0:
            ckpt['state_dict_dis'] = self.discriminator.state_dict()
            if self.amp_scaler_dis is not None:
                ckpt['amp_scaler_dis'] = self.amp_scaler_dis.state_dict()
        torch.save(ckpt, ckpt_path)
        if hasattr(self.configs.train, 'ema_rate'):
            ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
            torch.save(self.ema_model.state_dict(), ema_ckpt_path)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_text(self, text_list, phase):
        """
        Args:
            text_list: (b,) list
            phase: 'train' or 'val'
        """
        assert self.local_logging
        if self.local_logging:
            text_path = str(self.image_dir / phase / f"text-{self.log_step_img[phase]}.txt")
            with open(text_path, 'w') as ff:
                for text in text_list:
                    ff.write(text + '\n')

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def load_model(self, model, ckpt_path=None, tag='model'):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def unfreeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = True

    @torch.no_grad()
    def update_ema_model(self):
        decay = min(self.configs.train.ema_rate, (1 + self.current_iters) / (10 + self.current_iters))
        target_params = dict(self.model.named_parameters())
        # if hasattr(self.configs.train, 'ema_rate'):
            # with FSDP.summon_full_params(self.model, writeback=True):
                # target_params = dict(self.model.named_parameters())
        # else:
            # target_params = dict(self.model.named_parameters())

        one_minus_decay = 1.0 - decay

        for key, source_value in self.ema_model.named_parameters():
            target_value = target_params[key]
            if target_value.requires_grad:
                source_value.sub_(one_minus_decay * (source_value - target_value.data))

class TrainerBaseSR(TrainerBase):
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt_latent.size()
            self.queue_gt_latent = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_txt = ["", ] * self.queue_size
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            self.queue_gt_latent = self.queue_gt_latent[idx]
            self.queue_txt = [self.queue_txt[ii] for ii in idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            gt_latent_dequeue = self.queue_gt_latent[0:b, :, :, :].clone()
            txt_dequeue = deepcopy(self.queue_txt[0:b])
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            self.queue_gt_latent[0:b, :, :, :] = self.gt_latent.clone()
            self.queue_txt[0:b] = deepcopy(self.txt)

            self.lq = lq_dequeue
            self.gt = gt_dequeue
            self.gt_latent = gt_latent_dequeue
            self.txt = txt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_gt_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt_latent.clone()
            self.queue_txt[self.queue_ptr:self.queue_ptr + b] = deepcopy(self.txt)
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, phase='train'):
        if phase == 'train' and self.configs.data.get(phase).get('type') == 'realesrgan':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if (not hasattr(self, 'sharpener')) and self.configs.degradation.get('use_sharp', False):
                self.sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode=_INTERPOLATION_MODE)

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            self.lq, self.gt, self.txt = im_lq, im_gt, data['txt']
            if "gt_moment" not in data:
                self.gt_latent = self.encode_first_stage(
                    im_gt.cuda(),
                    center_input_sample=True,
                    deterministic=self.configs.train.loss_coef.get('rkl', 0) > 0,
                )
            else:
                self.gt_latent = self.encode_from_moment(
                    data['gt_moment'].cuda(),
                    deterministic=self.configs.train.loss_coef.get('rkl', 0) > 0,
                )

            if (not self.configs.train.use_text) or self.configs.data.train.params.random_crop:
                self.txt = [_positive,] * im_lq.shape[0]

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            batch = {'lq':self.lq, 'gt':self.gt, 'gt_latent':self.gt_latent, 'txt':self.txt}
        elif phase == 'val':
            resolution = self.configs.data.train.params.gt_size // self.configs.degradation.sf
            batch = {}
            batch['lq'] = data['lq'].cuda()
            if 'gt' in data:
                batch['gt'] = data['gt'].cuda()
            batch['txt'] = [_positive, ] * data['lq'].shape[0]
        else:
            batch = {key:value.cuda().to(dtype=torch.float32) for key, value in data.items()}

        return batch

    @torch.no_grad()
    def encode_from_moment(self, z, deterministic=True):
        dist = DiagonalGaussianDistribution(z)
        init_latents = dist.mode() if deterministic else dist.sample()

        latents_mean = latents_std = None
        if hasattr(self.sd_pipe.vae.config, "latents_mean") and self.sd_pipe.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.sd_pipe.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.sd_pipe.vae.config, "latents_std") and self.sd_pipe.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.sd_pipe.vae.config.latents_std).view(1, 4, 1, 1)

        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=z.device, dtype=z.dtype)
            latents_std = latents_std.to(device=z.device, dtype=z.dtype)
            init_latents = (init_latents - latents_mean) * scaling_factor / latents_std
        else:
            init_latents = init_latents * scaling_factor

        return init_latents

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def encode_first_stage(self, x, deterministic=False, center_input_sample=True):
        if center_input_sample:
            x = x * 2.0 - 1.0
        latents_mean = latents_std = None
        if hasattr(self.sd_pipe.vae.config, "latents_mean") and self.sd_pipe.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.sd_pipe.vae.config.latents_mean).view(1, -1, 1, 1)
        if hasattr(self.sd_pipe.vae.config, "latents_std") and self.sd_pipe.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.sd_pipe.vae.config.latents_std).view(1, -1, 1, 1)

        if deterministic:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.mode()
        else:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.sample()

        trunk_size = self.configs.sd_pipe.vae_split
        if trunk_size < x.shape[0]:
            init_latents = torch.cat([partial_encode(xx) for xx in x.split(trunk_size, 0)], dim=0)
        else:
            init_latents = partial_encode(x)

        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=x.device, dtype=x.dtype)
            latents_std = latents_std.to(device=x.device, dtype=x.dtype)
            init_latents = (init_latents - latents_mean) * scaling_factor / latents_std
        else:
            init_latents = init_latents * scaling_factor

        return init_latents

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def decode_first_stage(self, z, clamp=True):
        z = z / self.sd_pipe.vae.config.scaling_factor

        trunk_size = 1
        if trunk_size < z.shape[0]:
            out = torch.cat(
                [self.sd_pipe.vae.decode(xx).sample for xx in z.split(trunk_size, 0)], dim=0,
            )
        else:
            out = self.sd_pipe.vae.decode(z).sample
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def get_loss_from_discrimnator(self, logits_fake):
        if not (isinstance(logits_fake, list) or isinstance(logits_fake, tuple)):
            g_loss = -torch.mean(logits_fake, dim=list(range(1, logits_fake.ndim)))
        else:
            g_loss = -torch.mean(logits_fake[0], dim=list(range(1, logits_fake[0].ndim)))
            for current_logits in logits_fake[1:]:
                g_loss += -torch.mean(current_logits, dim=list(range(1, current_logits.ndim)))
            g_loss /= len(logits_fake)

        return g_loss

    def training_step(self, data):
        current_bs = data['gt'].shape[0]
        micro_bs = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_bs / micro_bs)

        # grad zero
        self.model.zero_grad()

        # update generator
        if self.configs.train.loss_coef.get('ldis', 0) > 0:
            self.freeze_model(self.discriminator) # freeze discriminator
            z0_pred_list = []
            tt_list = []
            prompt_embeds_list = []
        for jj in range(0, current_bs, micro_bs):
            micro_data = {key:value[jj:jj+micro_bs] for key, value in data.items()}
            last_batch = (jj+micro_bs >= current_bs)
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, zt_noisy, tt = self.backward_step(micro_data, num_grad_accumulate)
            else:
                with self.model.no_sync():
                    losses, z0_pred, zt_noisy, tt = self.backward_step(micro_data, num_grad_accumulate)
            if self.configs.train.loss_coef.get('ldis', 0) > 0:
                z0_pred_list.append(z0_pred.detach())
                tt_list.append(tt)
                prompt_embeds_list.append(self.prompt_embeds.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # update discriminator
        if (self.configs.train.loss_coef.get('ldis', 0) > 0 and
            (self.current_iters < self.configs.train.dis_init_iterations
            or self.current_iters % self.configs.train.dis_update_freq == 0)
            ):
            # grad zero
            self.unfreeze_model(self.discriminator) # update discriminator
            self.discriminator.zero_grad()
            for ii, jj in enumerate(range(0, current_bs, micro_bs)):
                micro_data = {key:value[jj:jj+micro_bs] for key, value in data.items()}
                last_batch = (jj+micro_bs >= current_bs)
                target = micro_data['gt_latent']
                inputs = z0_pred_list[ii]
                if last_batch or self.num_gpus <= 1:
                    logits = self.dis_backward_step(target, inputs, tt_list[ii], prompt_embeds_list[ii])
                else:
                    with self.discriminator.no_sync():
                        logits = self.dis_backward_step(
                            target, inputs, tt_list[ii], prompt_embeds_list[ii]
                        )

            # make logging
            if self.current_iters % self.configs.train.dis_update_freq == 0 and self.rank == 0:
                ndim = logits[0].ndim
                losses['real'] = logits[0].detach().mean(dim=list(range(1, ndim)))
                losses['fake'] = logits[1].detach().mean(dim=list(range(1, ndim)))

            if self.configs.train.use_amp:
                self.amp_scaler_dis.step(self.optimizer_dis)
                self.amp_scaler_dis.update()
            else:
                self.optimizer_dis.step()

        # make logging
        if self.rank == 0:
            self.log_step_train(
                losses, tt, micro_data, z0_pred, zt_noisy, z0_gt=micro_data['gt_latent'],
            )

    @torch.no_grad()
    def log_step_train(self, losses, tt, micro_data, z0_pred, zt_noisy, z0_gt=None, phase='train'):
        '''
        param losses: a dict recording the loss informations
        '''
        '''
        param loss: a dict recording the loss informations
        param micro_data: batch data
        param tt: 1-D tensor, time steps
        '''
        if hasattr(self.configs.train, 'timesteps'):
            if len(self.configs.train.timesteps) < 3:
                record_steps = sorted(self.configs.train.timesteps)
            else:
                record_steps = [min(self.configs.train.timesteps),
                                max(self.configs.train.timesteps)]
        else:
            max_inference_steps = self.configs.train.max_inference_steps
            record_steps = [1, max_inference_steps//2, max_inference_steps]
        if ((self.current_iters //  self.configs.train.dis_update_freq) %
            (self.configs.train.log_freq[0] // self.configs.train.dis_update_freq) == 1):
            self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                              for key in losses.keys() if key not in ['real', 'fake']}
            if self.configs.train.loss_coef.get('ldis', 0) > 0:
                self.logit_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in ['real', 'fake']}
            self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
        for jj in range(len(record_steps)):
            for key, value in losses.items():
                index = record_steps[jj] - 1
                mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                assert value.shape == mask.shape
                current_loss = torch.sum(value.detach() * mask)
                if key in ['real', 'fake']:
                    self.logit_mean[key][jj] += current_loss.item()
                else:
                    self.loss_mean[key][jj] += current_loss.item()
            self.loss_count[jj] += mask.sum().item()

        if ((self.current_iters //  self.configs.train.dis_update_freq) %
            (self.configs.train.log_freq[0] // self.configs.train.dis_update_freq) == 0):
            if torch.any(self.loss_count == 0):
                self.loss_count += 1e-4
            for key in losses.keys():
                if key in ['real', 'fake']:
                    self.logit_mean[key] /= self.loss_count
                else:
                    self.loss_mean[key] /= self.loss_count
            log_str = f"Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, "
            valid_keys = sorted([key for key in losses.keys() if key not in ['loss', 'real', 'fake']])
            for ii, key in enumerate(valid_keys):
                if ii == 0:
                    log_str += f"{key}"
                else:
                    log_str += f"/{key}"
            if self.configs.train.loss_coef.get('ldis', 0) > 0:
                log_str += "/real/fake:"
            else:
                log_str += ":"
            for jj, current_record in enumerate(record_steps):
                for ii, key in enumerate(valid_keys):
                    if ii == 0:
                        if key in ['dis', 'ldis']:
                            log_str += 't({:d}):{:+6.4f}'.format(
                                    current_record,
                                    self.loss_mean[key][jj].item(),
                                    )
                        elif key in ['lpips', 'ldif']:
                            log_str += 't({:d}):{:4.2f}'.format(
                                    current_record,
                                    self.loss_mean[key][jj].item(),
                                    )
                        elif key == 'llpips':
                            log_str += 't({:d}):{:5.3f}'.format(
                                    current_record,
                                    self.loss_mean[key][jj].item(),
                                    )
                        else:
                            log_str += 't({:d}):{:.1e}'.format(
                                    current_record,
                                    self.loss_mean[key][jj].item(),
                                    )
                    else:
                        if key in ['dis', 'ldis']:
                            log_str += f"/{self.loss_mean[key][jj].item():+6.4f}"
                        elif key in ['lpips', 'ldif']:
                            log_str += f"/{self.loss_mean[key][jj].item():4.2f}"
                        elif key == 'llpips':
                            log_str += f"/{self.loss_mean[key][jj].item():5.3f}"
                        else:
                            log_str += f"/{self.loss_mean[key][jj].item():.1e}"
                if self.configs.train.loss_coef.get('ldis', 0) > 0:
                    log_str += f"/{self.logit_mean['real'][jj].item():+4.2f}"
                    log_str += f"/{self.logit_mean['fake'][jj].item():+4.2f}, "
                else:
                    log_str += f", "
            log_str += 'lr:{:.1e}'.format(self.optimizer.param_groups[0]['lr'])
            self.logger.info(log_str)
            self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
        if ((self.current_iters //  self.configs.train.dis_update_freq) %
            (self.configs.train.log_freq[1] // self.configs.train.dis_update_freq) == 0):
            if zt_noisy is not None:
                xt_pred = self.decode_first_stage(zt_noisy.detach())
                self.logging_image(xt_pred, tag='xt-noisy', phase=phase, add_global_step=False)
            if z0_pred is not None:
                x0_pred = self.decode_first_stage(z0_pred.detach())
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=False)
            if z0_gt is not None:
                x0_recon = self.decode_first_stage(z0_gt.detach())
                self.logging_image(x0_recon, tag='x0-recons', phase=phase, add_global_step=False)
            if 'txt' in micro_data:
                self.logging_text(micro_data['txt'], phase=phase)
            self.logging_image(micro_data['lq'], tag='LQ', phase=phase, add_global_step=False)
            self.logging_image(micro_data['gt'], tag='GT', phase=phase, add_global_step=True)

        if ((self.current_iters //  self.configs.train.dis_update_freq) %
            (self.configs.train.save_freq // self.configs.train.dis_update_freq) == 1):
            self.tic = time.time()
        if ((self.current_iters //  self.configs.train.dis_update_freq) %
            (self.configs.train.save_freq // self.configs.train.dis_update_freq) == 0):
            self.toc = time.time()
            elaplsed = (self.toc - self.tic)
            self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
            self.logger.info("="*100)

    @torch.no_grad()
    def validation(self, phase='val'):
        torch.cuda.empty_cache()
        if not (self.configs.validate.use_ema and hasattr(self.configs.train, 'ema_rate')):
            self.model.eval()

        if self.configs.train.start_mode:
            start_noise_predictor = self.ema_model if self.configs.validate.use_ema else self.model
            intermediate_noise_predictor = None
        else:
            start_noise_predictor = self.start_model
            intermediate_noise_predictor = self.ema_model if self.configs.validate.use_ema else self.model
        num_iters_epoch = math.ceil(len(self.datasets[phase]) / self.configs.validate.batch)
        mean_psnr = mean_lpips = 0
        for jj, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            with torch.amp.autocast('cuda'):
                xt_progressive, x0_progressive = self.sample(
                    image_lq=data['lq'],
                    prompt=[_positive,]*data['lq'].shape[0],
                    target_size=tuple(data['gt'].shape[-2:]),
                    start_noise_predictor=start_noise_predictor,
                    intermediate_noise_predictor=intermediate_noise_predictor,
                )
            x0 = xt_progressive[-1]
            num_inference_steps = len(xt_progressive)

            if 'gt' in data:
                if not hasattr(self, 'psnr_metric'):
                    self.psnr_metric = pyiqa.create_metric(
                            'psnr',
                            test_y_channel=self.configs.train.get('val_y_channel', True),
                            color_space='ycbcr',
                            device=torch.device("cuda"),
                            )
                if not hasattr(self, 'lpips_metric'):
                    self.lpips_metric = pyiqa.create_metric(
                            'lpips-vgg',
                            device=torch.device("cuda"),
                            as_loss=False,
                            )
                x0_normalize = util_image.normalize_th(x0, mean=0.5, std=0.5, reverse=True)
                mean_psnr += self.psnr_metric(x0_normalize, data['gt']).sum().item()
                with torch.amp.autocast('cuda'), torch.no_grad():
                    mean_lpips += self.lpips_metric(x0_normalize, data['gt']).sum().item()

            if (jj + 1) % self.configs.validate.log_freq == 0:
                self.logger.info(f'Validation: {jj+1:02d}/{num_iters_epoch:02d}...')

                self.logging_image(data['gt'], tag='GT', phase=phase, add_global_step=False)
                xt_progressive = rearrange(torch.cat(xt_progressive, dim=1), 'b (k c) h w -> (b k) c h w', c=3)
                self.logging_image(
                    xt_progressive,
                    tag='sample-progress',
                    phase=phase,
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
                x0_progressive = rearrange(torch.cat(x0_progressive, dim=1), 'b (k c) h w -> (b k) c h w', c=3)
                self.logging_image(
                    x0_progressive,
                    tag='x0-progress',
                    phase=phase,
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
                self.logging_image(data['lq'], tag='LQ', phase=phase, add_global_step=True)

        if 'gt' in data:
            mean_psnr /= len(self.datasets[phase])
            mean_lpips /= len(self.datasets[phase])
            self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
            self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
            self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

        self.logger.info("="*100)

        if not (self.configs.validate.use_ema and hasattr(self.configs.train, 'ema_rate')):
            self.model.train()
        torch.cuda.empty_cache()

    def backward_step(self, micro_data, num_grad_accumulate):
        loss_coef = self.configs.train.loss_coef

        losses = {}
        z0_gt = micro_data['gt_latent']
        tt = torch.tensor(
            random.choices(self.configs.train.timesteps, k=z0_gt.shape[0]),
            dtype=torch.int64,
            device=f"cuda:{self.rank}",
        ) - 1

        with torch.autocast(device_type="cuda", enabled=self.configs.train.use_amp):
            model_pred = self.model(
                micro_data['lq'], tt, sample_posterior=False, center_input_sample=True,
            )
            z0_pred, zt_noisy_pred, z0_lq = self.sd_forward_step(
                prompt=micro_data['txt'],
                latents_hq=micro_data['gt_latent'],
                image_lq=micro_data['lq'],
                image_hq=micro_data['gt'],
                model_pred=model_pred,
                timesteps=tt,
            )
            # diffusion loss
            if loss_coef.get('ldif', 0) > 0:
                if self.configs.train.loss_type == 'L2':
                    ldif_loss = F.mse_loss(z0_pred, z0_gt, reduction='none')
                elif self.configs.train.loss_type == 'L1':
                    ldif_loss = F.l1_loss(z0_pred, z0_gt, reduction='none')
                else:
                    raise TypeError(f"Unsupported Loss type for Diffusion: {self.configs.train.loss_type}")
                ldif_loss = torch.mean(ldif_loss, dim=list(range(1, z0_gt.ndim)))
                losses['ldif'] = ldif_loss * loss_coef['ldif']
            # Gaussian constraints
            if loss_coef.get('kl', 0) > 0:
                losses['kl'] = model_pred.kl() * loss_coef['kl']
            if loss_coef.get('pkl', 0) > 0:
                losses['pkl'] = model_pred.partial_kl() * loss_coef['pkl']
            if loss_coef.get('rkl', 0) > 0:
                other = Box(
                    {'mean': z0_gt-z0_lq,
                     'var':torch.ones_like(z0_gt),
                     'logvar':torch.zeros_like(z0_gt)}
                )
                losses['rkl'] = model_pred.kl(other) * loss_coef['rkl']
            # discriminator loss
            if loss_coef.get('ldis', 0) > 0:
                if self.current_iters > self.configs.train.dis_init_iterations:
                    logits_fake = self.discriminator(
                        torch.clamp(z0_pred, min=_Latent_bound['min'], max=_Latent_bound['max']),
                        timestep=tt,
                        encoder_hidden_states=self.prompt_embeds,
                    )
                    losses['ldis'] = self.get_loss_from_discrimnator(logits_fake) * loss_coef['ldis']
                else:
                    losses['ldis'] = torch.zeros((z0_gt.shape[0], ), dtype=torch.float32).cuda()
            # perceptual loss
            if loss_coef.get('llpips', 0) > 0:
                losses['llpips'] = self.llpips_loss(z0_pred, z0_gt).view(-1) * loss_coef['llpips']

            for key in ['ldif', 'kl', 'rkl', 'pkl', 'ldis', 'llpips']:
                if loss_coef.get(key, 0) > 0:
                    if not 'loss' in losses:
                        losses['loss'] = losses[key]
                    else:
                        losses['loss'] = losses['loss'] + losses[key]
            loss = losses['loss'].mean() / num_grad_accumulate

        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, zt_noisy_pred, tt

    def dis_backward_step(self, target, inputs, tt, prompt_embeds):
        with torch.autocast(device_type="cuda", enabled=self.configs.train.use_amp):
            logits_real = self.discriminator(target, tt, prompt_embeds)
            inputs = inputs.clamp(min=_Latent_bound['min'], max=_Latent_bound['max'])
            logits_fake = self.discriminator(inputs, tt, prompt_embeds)

            loss = hinge_d_loss(logits_real, logits_fake).mean()

        if self.amp_scaler_dis is None:
            loss.backward()
        else:
            self.amp_scaler_dis.scale(loss).backward()

        return logits_real[-1], logits_fake[-1]

    def scale_sd_input(
        self,
        x:torch.Tensor,
        sigmas: torch.Tensor = None,
        timestep: torch.Tensor = None,
    ) :
        if sigmas is None:
            if not self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1):
                self.sd_pipe.scheduler = EulerDiscreteScheduler.from_pipe(
                    self.configs.sd_pipe.params.pretrained_model_name_or_path,
                    cache_dir=self.configs.sd_pipe.params.cache_dir,
                    subfolder='scheduler',
                )
                assert self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1)
            sigmas = self.sd_pipe.scheduler.sigmas.flip(0).to(x.device)[timestep] # (b,)
            sigmas = append_dims(sigmas, x.ndim)

        if sigmas.ndim < x.ndim:
            sigmas = append_dims(sigmas, x.ndim)
        out = x / ((sigmas**2 + 1) ** 0.5)
        return out

    def prepare_lq_latents(
        self,
        image_lq: torch.Tensor,
        timestep: torch.Tensor,
        height: int = 512,
        width: int = 512,
        start_noise_predictor: torch.nn.Module = None,
    ):
        """
        Input:
            image_lq: low-quality image, torch.Tensor, range in [0, 1]
            hight, width: resolution for high-quality image

        """
        image_lq_up = F.interpolate(image_lq, size=(height, width), mode='bicubic')
        init_latents = self.encode_first_stage(
            image_lq_up, deterministic=False, center_input_sample=True,
        )

        if start_noise_predictor is None:
            model_pred = None
        else:
            model_pred = start_noise_predictor(
                image_lq, timestep, sample_posterior=False, center_input_sample=True,
            )

        # get latents
        sigmas = self.sigmas_cache[timestep]
        sigmas = append_dims(sigmas, init_latents.ndim)
        latents = self.add_noise(init_latents, sigmas, model_pred)

        return latents

    def add_noise(self, latents, sigmas, model_pred=None):
        if sigmas.ndim < latents.ndim:
            sigmas = append_dims(sigmas, latents.ndim)

        if model_pred is None:
            noise = torch.randn_like(latents)
            zt_noisy = latents + sigmas * noise
        else:
            if self.configs.train.loss_coef.get('rkl', 0) > 0:
                mean, std = model_pred.mean, model_pred.std
                zt_noisy = latents + mean + sigmas * std * torch.randn_like(latents)
            else:
                zt_noisy = latents + sigmas * model_pred.sample()

        return zt_noisy

    def retrieve_timesteps(self):
        device=torch.device(f"cuda:{self.rank}")

        num_inference_steps = self.configs.train.get('num_inference_steps', 5)
        timesteps = np.linspace(
            max(self.configs.train.timesteps), 0, num_inference_steps,
            endpoint=False, dtype=np.int64,
        ) - 1
        timesteps = torch.from_numpy(timesteps).to(device)
        self.sd_pipe.scheduler.timesteps = timesteps

        sigmas = self.sigmas_cache[timesteps.long()]
        sigma_last = torch.tensor([0,], dtype=torch.float32).to(device=sigmas.device)
        sigmas = torch.cat([sigmas, sigma_last]).type(torch.float32)
        self.sd_pipe.scheduler.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

        self.sd_pipe.scheduler._step_index = None
        self.sd_pipe.scheduler._begin_index = None

        return self.sd_pipe.scheduler.timesteps, num_inference_steps

class TrainerSDTurboSR(TrainerBaseSR):
    def sd_forward_step(
        self,
        prompt: Union[str, List[str]] = None,
        latents_hq: Optional[torch.Tensor] = None,
        image_lq: torch.Tensor = None,
        image_hq: torch.Tensor = None,
        model_pred: DiagonalGaussianDistribution = None,
        timesteps: List[int] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image_lq (`torch.Tensor`): The low-quality image(s) for enhancement, range in [0, 1].
            image_hq (`torch.Tensor`): The high-quality image(s) for enhancement, range in [0, 1].
            noise_pred (`torch.Tensor`): Predicted noise by the noise prediction model
            latents_hq (`torch.Tensor`, *optional*):
                Pre-generated high-quality latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation.  If not provided, a latents tensor will be generated by sampling using vae .
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
        """
        device=torch.device(f"cuda:{self.rank}")
        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.sd_pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = prompt_embeds

        # select the noise level, self.scheduler.sigmas, [1001,], descending
        if not hasattr(self, 'sigmas_cache'):
            assert self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1)
            self.sigmas_cache = self.sd_pipe.scheduler.sigmas.flip(0)[1:].to(device) #ascending,1000
        sigmas = self.sigmas_cache[timesteps] # (b,)

        # Prepare input for SD
        height, width = image_hq.shape[-2:]
        if self.configs.train.start_mode:
            image_lq_up = F.interpolate(image_lq, size=(height, width), mode='bicubic')
            zt_clean = self.encode_first_stage(
                image_lq_up, center_input_sample=True,
                deterministic=self.configs.train.loss_coef.get('rkl', 0) > 0,
            )
        else:
            if latents_hq is None:
                zt_clean = self.encode_first_stage(
                    image_hq, center_input_sample=True, deterministic=False,
                )
            else:
                zt_clean = latents_hq

        sigmas = append_dims(sigmas, zt_clean.ndim)
        zt_noisy = self.add_noise(zt_clean, sigmas, model_pred)

        prompt_embeds = prompt_embeds.to(device)

        zt_noisy_scale = self.scale_sd_input(zt_noisy, sigmas)
        eps_pred = self.sd_pipe.unet(
            zt_noisy_scale,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]   # eps-mode for sdxl and sdxl-refiner

        if self.configs.train.noise_detach:
            z0_pred = zt_noisy.detach() - sigmas * eps_pred
        else:
            z0_pred = zt_noisy - sigmas * eps_pred

        return z0_pred, zt_noisy, zt_clean

    @torch.no_grad()
    def sample(
        self,
        image_lq: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        target_size: Tuple[int, int] = (1024, 1024),
        start_noise_predictor: torch.nn.Module = None,
        intermediate_noise_predictor: torch.nn.Module = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image_lq (`torch.Tensor` or `PIL.Image.Image` or `np.ndarray` or `List[torch.Tensor]` or `List[PIL.Image.Image]` or `List[np.ndarray]`):
                The image(s) to modify with the pipeline.
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                The required height and width of the super-resolved image.
            strength (`float`, *optional*, defaults to 0.3):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of
                `denoising_start` being declared as an integer, the value of `strength` will be ignored.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
        """
        device=torch.device(f"cuda:{self.rank}")
        batch_size = image_lq.shape[0]

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.sd_pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        timesteps, num_inference_steps = self.retrieve_timesteps()
        latent_timestep = timesteps[:1].repeat(batch_size)

        # Prepare latent variables
        height, width = target_size
        latents = self.prepare_lq_latents(image_lq, latent_timestep.long(), height, width, start_noise_predictor)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.sd_pipe.prepare_extra_step_kwargs(None, 0.0)

        prompt_embeds = prompt_embeds.to(device)

        x0_progressive = []
        images_progressive = []
        for i, t in enumerate(timesteps):
            latents_scaled = self.sd_pipe.scheduler.scale_model_input(latents, t)

            # predict the noise residual
            eps_pred = self.sd_pipe.unet(
                latents_scaled,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            z0_pred = latents - self.sigmas_cache[t.long()] * eps_pred

            # compute the previous noisy sample x_t -> x_t-1
            if intermediate_noise_predictor is not None and i + 1 < len(timesteps):
                t_next = timesteps[i+1]
                noise = intermediate_noise_predictor(image_lq, t_next, center_input_sample=True)
            else:
                noise = None
            extra_step_kwargs['noise'] = noise
            latents = self.sd_pipe.scheduler.step(eps_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            image = self.decode_first_stage(latents)
            images_progressive.append(image)

            x0_pred = self.decode_first_stage(z0_pred)
            x0_progressive.append(x0_pred)

        return images_progressive, x0_progressive

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def hinge_d_loss(
        logits_real: Union[torch.Tensor, List[torch.Tensor,]],
        logits_fake: Union[torch.Tensor, List[torch.Tensor,]],
    ):
    def _hinge_d_loss(logits_real, logits_fake):
        loss_real = F.relu(1.0 - logits_real)
        loss_fake = F.relu(1.0 + logits_fake)
        d_loss = 0.5 * (loss_real + loss_fake)
        loss = d_loss.mean(dim=list(range(1, logits_real.ndim)))

        return loss

    if not (isinstance(logits_real, list) or isinstance(logits_real, tuple)):
        loss = _hinge_d_loss(logits_real, logits_fake)
    else:
        loss = _hinge_d_loss(logits_real[0], logits_fake[0])
        for xx, yy in zip(logits_real[1:], logits_fake[1:]):
            loss += _hinge_d_loss(xx, yy)

        loss /= len(logits_real)

    return loss

def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')
