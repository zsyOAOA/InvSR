#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import warnings
warnings.filterwarnings("ignore")

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/sd-turbo-sr-ldis.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--ldif",
            type=float,
            default=1.0,
            help="Loss coefficient for diffsuion in latent space",
            )
    parser.add_argument(
            "--llpips",
            type=float,
            default=2.0,
            help="Loss coefficient for latent lpips",
            )
    parser.add_argument(
            "--ldis",
            type=float,
            default=0.1,
            help="Loss coefficient for latent discriminator",
            )
    parser.add_argument(
            "--use_text",
            type=str2bool,
            default='False',
            help="Text Prompt",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    if args.ldif > 0:
        configs.train.loss_coef.ldif = args.ldif
    if args.ldis > 0:
        configs.train.loss_coef.ldis = args.ldis
    if args.llpips > 0:
        configs.train.loss_coef.llpips = args.llpips
    configs.train.use_text = args.use_text

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
