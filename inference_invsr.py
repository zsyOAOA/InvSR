#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from sampler_invsr import InvSamplerSR

from utils import util_common
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path")
    parser.add_argument("-o", "--out_path", type=str, default="", help="Output path")
    parser.add_argument("--bs", type=int, default=1, help="Batchsize for loading image")
    parser.add_argument("--chopping_bs", type=int, default=8, help="Batchsize for chopped patch")
    parser.add_argument("-t", "--timesteps", type=int, nargs="+", help="The inversed timesteps")
    parser.add_argument("-n", "--num_steps", type=int, default=1, help="Number of inference steps")
    parser.add_argument(
        "--cfg_path", type=str, default="./configs/sample-sd-turbo.yaml", help="Configuration path.",
    )
    parser.add_argument(
        "--sd_path", type=str, default="", help="Path for Stable Diffusion Model",
    )
    parser.add_argument(
        "--started_ckpt_path", type=str, default="", help="Checkpoint path for noise predictor"
    )
    parser.add_argument(
        "--tiled_vae", type=str2bool, default='true', help="Enabled tiled VAE.",
    )
    parser.add_argument(
        "--color_fix", type=str, default='', choices=['wavelet', 'ycbcr'], help="Fix the color shift",
    )
    parser.add_argument(
        "--chopping_size", type=int, default=128, help="Chopping size when dealing large images"
    )
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load(args.cfg_path)

    if args.timesteps is not None:
        assert len(args.timesteps) == args.num_steps
        configs.timesteps = sorted(args.timesteps, reverse=True)
    else:
        if args.num_steps == 1:
            configs.timesteps = [200,]
        elif args.num_steps == 2:
            configs.timesteps = [200, 100]
        elif args.num_steps == 3:
            configs.timesteps = [200, 100, 50]
        elif args.num_steps == 4:
            configs.timesteps = [200, 150, 100, 50]
        elif args.num_steps == 5:
            configs.timesteps = [250, 200, 150, 100, 50]
        else:
            assert args.num_steps <= 250
            configs.timesteps = np.linspace(
                start=args.started_step, stop=0, num=args.num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    # path to save Stable Diffusion
    sd_path = args.sd_path if args.sd_path else "./weights"
    util_common.mkdir(sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = sd_path

    # path to save noise predictor
    if args.started_ckpt_path:
        started_ckpt_path = args.started_ckpt_path
    else:
        started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
        started_ckpt_dir = "./weights"
        util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
        started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
        if not started_ckpt_path.exists():
            load_file_from_url(
                url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
                model_dir=started_ckpt_dir,
                progress=True,
                file_name=started_ckpt_name,
            )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = args.bs
    configs.tiled_vae = args.tiled_vae
    configs.color_fix = args.color_fix
    configs.basesr.chopping.pch_size = args.chopping_size
    if args.bs > 1:
        configs.basesr.chopping.extra_bs = 1
    else:
        configs.basesr.chopping.extra_bs = args.chopping_bs

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    sampler = InvSamplerSR(configs)

    sampler.inference(args.in_path, out_path=args.out_path, bs=args.bs)

if __name__ == '__main__':
    main()
