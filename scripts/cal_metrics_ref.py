#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-08-13 21:37:58

'''
Calculate PSNR, SSIM, LPIPS, and NIQE.
'''

import sys
from pathlib import Path

import os
import math
import lpips
import pyiqa
import torch
import argparse
from einops import rearrange
from loguru import logger as base_logger

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import util_image
from utils.util_opts import str2bool
from datapipe.datasets import BaseData

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=16, help="Batch size")
parser.add_argument("--gt_dir", type=str, default="", help="Path to save the HQ images")
parser.add_argument("--sr_dir", type=str, default="", help="Path to save the SR images")
parser.add_argument("--log_name", type=str, default='metrics.log', help="Logging path")
parser.add_argument("--test_y_channel", type=str2bool, default='true', help="Y channel for PSNR and SSIM")
parser.add_argument("--fid", type=str2bool, default='false', help="Calculating FID")
parser.add_argument("--niqe", type=str2bool, default='false', help="Calculating NIQE")
parser.add_argument("--dists", type=str2bool, default='false', help="Calculating DISTS")
parser.add_argument("--maniqa", type=str2bool, default='false', help="Calculating MANIQA")
parser.add_argument("--pi", type=str2bool, default='false', help="Calculating PI")
parser.add_argument("--tocpu", type=str2bool, default='false', help="Moving model to CPU")
args = parser.parse_args()

# setting logger
log_path = str(Path(args.sr_dir).parent / f'{args.log_name}')
logger = base_logger
logger.remove()
logger.add(log_path, format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}", mode='w', level='INFO')
logger.add(sys.stderr, format="{message}", level='INFO')
logger.info(f"Ground truth: {args.gt_dir}")
logger.info(f"SR result: {args.sr_dir}")

if args.test_y_channel:
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr')
else:
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=False, color_space='rgb')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=False, color_space='rgb')
if args.fid:
    fid_metric = pyiqa.create_metric('fid')
if args.niqe:
    niqe_metric = pyiqa.create_metric('niqe')
if args.dists:
    dists_metric = pyiqa.create_metric('dists')
if args.maniqa:
    maniqa_metric = pyiqa.create_metric('maniqa')
if args.pi:
    pi_metric = pyiqa.create_metric('pi')
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
if args.tocpu:
    clipiqa_metric = pyiqa.create_metric('clipiqa').to('cpu')
    musiq_metric = pyiqa.create_metric('musiq').to('cpu')
else:
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    musiq_metric = pyiqa.create_metric('musiq')

dataset = BaseData(
        dir_path=args.sr_dir,
        transform_type='default',
        transform_kwargs={'mean': 0.0, 'std': 1.0},
        extra_dir_path=args.gt_dir,
        extra_transform_type='default',
        extra_transform_kwargs={'mean': 0.0, 'std': 1.0},
        need_path=True,
        im_exts=['png', 'jpg'],
        recursive=False,
        )
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=0
        )
logger.info(f'Number of images: {len(dataset)}')

metrics = {
        'PSNR': 0,
        'SSIM': 0,
        'LPIPS_VGG': 0,
        'LPIPS_ALEX': 0,
        'CLIPIQA': 0,
        'MUSIQ': 0,
        }
if args.niqe:
    metrics['NIQE'] = 0
if args.dists:
    metrics['DISTS'] = 0
if args.maniqa:
    metrics['MANIQA'] = 0
if args.pi:
    metrics['PI'] = 0
for ii, data in enumerate(dataloader):
    im_sr = data['image'].cuda()  # N x h x w x 3, [0,1]
    im_gt = data['gt'].cuda()     # N x h x w x 3, [0,1]
    current_bs = im_sr.shape[0]

    if not (im_sr.shape == im_gt.shape):
        height = min(im_sr.shape[-2], im_gt.shape[-2])
        width = min(im_sr.shape[-1], im_gt.shape[-1])
        im_sr = im_sr[:, :, :height, :width]
        im_gt = im_gt[:, :, :height, :width]

    current_psnr = psnr_metric(im_sr, im_gt).mean().item()
    current_ssim = ssim_metric(im_sr, im_gt).mean().item()
    current_lpips_vgg = loss_fn_vgg(
            (im_gt - 0.5) / 0.5,
            (im_sr - 0.5) / 0.5,
            ).mean().item()
    current_lpips_alex = loss_fn_alex(
            (im_gt - 0.5) / 0.5,
            (im_sr - 0.5) / 0.5,
            ).mean().item()
    if args.tocpu:
        current_clipiqa = clipiqa_metric(im_sr.cpu()).mean().item()
        current_musiq = musiq_metric(im_sr.cpu()).mean().item()
    else:
        current_clipiqa = clipiqa_metric(im_sr).mean().item()
        current_musiq = musiq_metric(im_sr).mean().item()
    if args.niqe:
        current_niqe = niqe_metric(im_sr).mean().item()
    if args.dists:
        current_dists = dists_metric(im_sr, im_gt).mean().item()
    if args.maniqa:
        current_maniqa = maniqa_metric(im_sr).mean().item()
    if args.pi:
        current_pi = pi_metric(im_sr).mean().item()

    if (ii+1) % 30 == 0:
        log_str = ('Processing: {:03d}/{:03d}, PSNR={:5.2f}, LPIPS={:6.4f}/{:6.4f}, CLIPIQA={:6.4f}, MUSIQ={:6.4f}'.format(
                        ii+1,
                        math.ceil(len(dataset) /args.bs),
                        current_psnr,
                        current_lpips_vgg,
                        current_lpips_alex,
                        current_clipiqa,
                        current_musiq,
                        ))
        logger.info(log_str)

    metrics['PSNR'] += current_psnr * current_bs
    metrics['SSIM'] += current_ssim * current_bs
    metrics['LPIPS_VGG'] += current_lpips_vgg * current_bs
    metrics['LPIPS_ALEX'] += current_lpips_alex * current_bs
    metrics['CLIPIQA'] += current_clipiqa * current_bs
    metrics['MUSIQ'] += current_musiq * current_bs
    if args.niqe:
        metrics['NIQE'] += current_niqe * current_bs
    if args.dists:
        metrics['DISTS'] += current_dists * current_bs
    if args.maniqa:
        metrics['MANIQA'] += current_maniqa * current_bs
    if args.pi:
        metrics['PI'] += current_pi * current_bs

for key in metrics.keys():
    metrics[key] /= len(dataset)

if args.fid:
    metrics['FID'] = fid_metric(args.sr_dir, args.gt_dir)

logger.info(f"MEAN PSNR: {metrics['PSNR']:5.2f}")
logger.info(f"MEAN SSIM: {metrics['SSIM']:6.4f}")
logger.info(f"MEAN LPIPS(VGG): {metrics['LPIPS_VGG']:6.4f}")
logger.info(f"MEAN LPIPS(ALEX): {metrics['LPIPS_ALEX']:6.4f}")
logger.info(f"MEAN CLIPIQA: {metrics['CLIPIQA']:6.4f}")
logger.info(f"MEAN MUSIQ: {metrics['MUSIQ']:6.4f}")
if args.fid:
    logger.info(f"MEAN FID: {metrics['FID']:6.2f}")
if args.niqe:
    logger.info(f"MEAN NIQE: {metrics['NIQE']:7.4f}")
if args.dists:
    logger.info(f"MEAN DISTS: {metrics['DISTS']:6.4f}")
if args.maniqa:
    logger.info(f"MEAN MANIQA: {metrics['MANIQA']:6.4f}")
if args.pi:
    logger.info(f"MEAN PI: {metrics['PI']:7.4f}")

