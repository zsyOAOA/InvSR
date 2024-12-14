#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common
from utils import util_color_fix

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from diffusers import StableDiffusionInvEnhancePipeline, AutoencoderKL

_positive= 'Cinematic, high-contrast, photo-realistic, 8k, ultra HD, ' +\
           'meticulous detailing, hyper sharpness, perfect without deformations'
_negative= 'Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,' +\
           'painting, drawing, sketch, oil painting'

class BaseSampler:
    def __init__(self, configs):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
                configs.sampler_config.{start_timesteps, padding_mod, seed, sf, num_sample_steps}
            seed: int, random seed
        '''
        self.configs = configs

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self):
        # Build Stable diffusion
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        base_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            self.write_log(f'Loading scheduler of {pipe_id}...')
            base_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                base_pipe.scheduler.config
            )
            self.write_log('Loaded Done')
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            torch_dtype = params_vae.pop('torch_dtype')
            params_vae['torch_dtype'] = get_torch_dtype(torch_dtype)
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            self.write_log(f'Loading improved vae from {pipe_id}...')
            base_pipe.vae = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(
                **params_vae,
            )
            self.write_log('Loaded Done')
        if self.configs.base_model in ['sd-turbo', 'sd2base'] :
            sd_pipe = StableDiffusionInvEnhancePipeline.from_pipe(base_pipe)
        else:
            raise ValueError(f"Unsupported base model: {self.configs.base_model}!")
        sd_pipe.to(f"cuda")
        if self.configs.sliced_vae:
            sd_pipe.vae.enable_slicing()
        if self.configs.tiled_vae:
            sd_pipe.vae.enable_tiling()
            sd_pipe.vae.tile_latent_min_size = self.configs.latent_tiled_size
            sd_pipe.vae.tile_sample_min_size = self.configs.sample_tiled_size
        if self.configs.gradient_checkpointing_vae:
            self.write_log(f"Activating gradient checkpoing for vae...")
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder, True)
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder, True)

        model_configs = self.configs.model_start
        params = model_configs.get('params', dict)
        model_start = util_common.get_obj_from_str(model_configs.target)(**params)
        model_start.cuda()
        ckpt_path = model_configs.get('ckpt_path')
        assert ckpt_path is not None
        self.write_log(f"Loading started model from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=f"cuda")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model_start, state)
        self.write_log(f"Loading Done")
        model_start.eval()
        setattr(sd_pipe, 'start_noise_predictor', model_start)

        self.sd_pipe = sd_pipe

class InvSamplerSR(BaseSampler):
    @torch.no_grad()
    def sample_func(self, im_cond):
        '''
        Input:
            im_cond: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            xt: h x w x c, numpy array, [0,1], RGB
        '''
        if self.configs.cfg_scale > 1.0:
            negative_prompt = [_negative,]*im_cond.shape[0]
        else:
            negative_prompt = None

        ori_h_lq, ori_w_lq = im_cond.shape[-2:]
        ori_w_hq = ori_w_lq * self.configs.basesr.sf
        ori_h_hq = ori_h_lq * self.configs.basesr.sf
        vae_sf = (2 ** (len(self.sd_pipe.vae.config.block_out_channels) - 1))
        if hasattr(self.sd_pipe, 'unet'):
            diffusion_sf = (2 ** (len(self.sd_pipe.unet.config.block_out_channels) - 1))
        else:
            diffusion_sf = self.sd_pipe.transformer.patch_size
        mod_lq = vae_sf // self.configs.basesr.sf * diffusion_sf
        idle_pch_size = self.configs.basesr.chopping.pch_size

        if min(im_cond.shape[-2:]) >= idle_pch_size:
            pad_h_up = pad_w_left = 0
        else:
            while min(im_cond.shape[-2:]) < idle_pch_size:
                pad_h_up = max(min((idle_pch_size - im_cond.shape[-2]) // 2, im_cond.shape[-2]-1), 0)
                pad_h_down = max(min(idle_pch_size - im_cond.shape[-2] - pad_h_up, im_cond.shape[-2]-1), 0)
                pad_w_left = max(min((idle_pch_size - im_cond.shape[-1]) // 2, im_cond.shape[-1]-1), 0)
                pad_w_right = max(min(idle_pch_size - im_cond.shape[-1] - pad_w_left, im_cond.shape[-1]-1), 0)
                im_cond = F.pad(im_cond, pad=(pad_w_left, pad_w_right, pad_h_up, pad_h_down), mode='reflect')

        if im_cond.shape[-2] == idle_pch_size and im_cond.shape[-1] == idle_pch_size:
            target_size = (
                im_cond.shape[-2] * self.configs.basesr.sf,
                im_cond.shape[-1] * self.configs.basesr.sf
            )
            res_sr = self.sd_pipe(
                image=im_cond.type(torch.float16),
                prompt=[_positive, ]*im_cond.shape[0],
                negative_prompt=negative_prompt,
                target_size=target_size,
                timesteps=self.configs.timesteps,
                guidance_scale=self.configs.cfg_scale,
                output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
            ).images
        else:
            if not (im_cond.shape[-2] % mod_lq == 0 and im_cond.shape[-1] % mod_lq == 0):
                target_h_lq = math.ceil(im_cond.shape[-2] / mod_lq) * mod_lq
                target_w_lq = math.ceil(im_cond.shape[-1] / mod_lq) * mod_lq
                pad_h = target_h_lq - im_cond.shape[-2]
                pad_w = target_w_lq - im_cond.shape[-1]
                im_cond= F.pad(im_cond, pad=(0, pad_w, 0, pad_h), mode='reflect')

            im_spliter = util_image.ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride= int(idle_pch_size * 0.50),
                sf=self.configs.basesr.sf,
                weight_type=self.configs.basesr.chopping.weight_type,
                extra_bs=self.configs.basesr.chopping.extra_bs,
            )
            for im_lq_pch, index_infos in im_spliter:
                target_size = (
                    im_lq_pch.shape[-2] * self.configs.basesr.sf,
                    im_lq_pch.shape[-1] * self.configs.basesr.sf,
                )

                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                res_sr_pch = self.sd_pipe(
                    image=im_lq_pch.type(torch.float16),
                    prompt=[_positive, ]*im_lq_pch.shape[0],
                    negative_prompt=negative_prompt,
                    target_size=target_size,
                    timesteps=self.configs.timesteps,
                    guidance_scale=self.configs.cfg_scale,
                    output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
                ).images

                # end.record()
                # torch.cuda.synchronize()
                # print(f"Time: {start.elapsed_time(end):.6f}")

                im_spliter.update(res_sr_pch, index_infos)
            res_sr = im_spliter.gather()

        pad_h_up *= self.configs.basesr.sf
        pad_w_left *= self.configs.basesr.sf
        res_sr = res_sr[:, :, pad_h_up:ori_h_hq+pad_h_up, pad_w_left:ori_w_hq+pad_w_left]

        if self.configs.color_fix:
            im_cond_up = F.interpolate(
                im_cond, size=res_sr.shape[-2:], mode='bicubic', align_corners=False, antialias=True
            )
            if self.configs.color_fix == 'ycbcr':
                res_sr = util_color_fix.ycbcr_color_replace(res_sr, im_cond_up)
            elif self.configs.color_fix == 'wavelet':
                res_sr = util_color_fix.wavelet_reconstruction(res_sr, im_cond_up)
            else:
                raise ValueError(f"Unsupported color fixing type: {self.configs.color_fix}")

        res_sr = res_sr.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()

        return res_sr

    def inference(self, in_path, out_path, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if not out_path.exists():
            out_path.mkdir(parents=True)

        if in_path.is_dir():
            data_config = {'type': 'base',
                           'params': {'dir_path': str(in_path),
                                      'transform_type': 'default',
                                      'transform_kwargs': {
                                          'mean': 0.0,
                                          'std': 1.0,
                                          },
                                      'need_path': True,
                                      'recursive': False,
                                      'length': None,
                                      }
                           }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=False, drop_last=False,
            )
            for data in dataloader:
                res = self.sample_func(data['lq'].cuda())

                for jj in range(res.shape[0]):
                    im_name = Path(data['path'][jj]).stem
                    save_path = str(out_path / f"{im_name}.png")
                    util_image.imwrite(res[jj], save_path, dtype_in='float32')
        else:
            im_cond = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_cond = util_image.img2tensor(im_cond).cuda()                   # 1 x c x h x w

            image = self.sample_func(im_cond).squeeze(0)

            save_path = str(out_path / f"{in_path.stem}.png")
            util_image.imwrite(image, save_path, dtype_in='float32')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')

if __name__ == '__main__':
    pass

