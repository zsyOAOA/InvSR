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
parser.add_argument("--bs", type=int, default=1, help="Batch size")
parser.add_argument("-i", "--indir", type=str, default="", help="Path to save the testing images")
parser.add_argument("-r", "--refdir", type=str, default="", help="Reference images for fid")
parser.add_argument("-t", "--tocpu", type=str2bool, default='false')
parser.add_argument("--pi", type=str2bool, default='false', help="PI metric")
parser.add_argument("--niqe", type=str2bool, default='false', help="NIQE metric")
parser.add_argument("--maniqa", type=str2bool, default='false', help="MANIQA metric")
parser.add_argument("--tres", type=str2bool, default='false', help="TReS metric")
parser.add_argument("--dbcnn", type=str2bool, default='false', help="DBCNN metric")
args = parser.parse_args()

# setting logger
log_path = str(Path(args.indir).parent / 'metrics.log')
logger = base_logger
logger.remove()
logger.add(log_path, format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}", mode='w', level='INFO')
logger.add(sys.stderr, format="{message}", level='INFO')
logger.info(f"Image Floder: {args.indir}")

if args.pi:
    pi_metric = pyiqa.create_metric('pi')
if args.niqe:
    niqe_metric = pyiqa.create_metric('niqe')
if args.maniqa:
    maniqa_metric = pyiqa.create_metric('maniqa')
if args.tres:
    tres_metric = pyiqa.create_metric('tres')
if args.dbcnn:
    dbcnn_metric = pyiqa.create_metric('dbcnn')
if args.refdir:
    fid_metric = pyiqa.create_metric('fid')
if args.tocpu:
    clipiqa_metric = pyiqa.create_metric('clipiqa').to('cpu')
    musiq_metric = pyiqa.create_metric('musiq').to('cpu')
else:
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    musiq_metric = pyiqa.create_metric('musiq')

dataset = BaseData(
        dir_path=args.indir,
        transform_type='default',
        transform_kwargs={'mean': 0.0, 'std': 1.0},
        need_path=True,
        im_exts=['png', 'jpeg', 'jpg', ],
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
        'PI': 0,
        'CLIPIQA': 0,
        'MUSIQ': 0,
        'MANIQA': 0,
        'TRES': 0,
        'DBCNN': 0,
        }
if args.niqe:
    metrics['NIQE'] = 0
for ii, data in enumerate(dataloader):
    im = data['image'].cuda()  # N x h x w x 3, [0,1]
    current_bs = im.shape[0]

    if args.pi:
        current_pi = pi_metric(im).sum().item()
    if args.niqe:
        current_niqe = niqe_metric(im).sum().item()
    if args.maniqa:
        current_maniqa = maniqa_metric(im).sum().item()
    if args.tres:
        current_tres = tres_metric(im).sum().item()
    if args.dbcnn:
        current_dbcnn = dbcnn_metric(im).sum().item()
    if args.tocpu:
        current_clipiqa = clipiqa_metric(im.cpu()).sum().item()
        current_musiq = musiq_metric(im.cpu()).sum().item()
    else:
        current_clipiqa = clipiqa_metric(im).sum().item()
        current_musiq = musiq_metric(im).sum().item()

    if (ii+1) % 10 == 0:
        log_str = ('Processing: {:03d}/{:03d}'.format(ii+1, math.ceil(len(dataset) / args.bs)))
        logger.info(log_str)

    metrics['CLIPIQA'] += current_clipiqa
    metrics['MUSIQ'] += current_musiq
    if args.pi:
        metrics['PI'] += current_pi
    if args.niqe:
        metrics['NIQE'] += current_niqe
    if args.maniqa:
        metrics['MANIQA'] += current_maniqa
    if args.tres:
        metrics['TRES'] += current_tres
    if args.dbcnn:
        metrics['DBCNN'] += current_dbcnn

for key in metrics.keys():
    metrics[key] /= len(dataset)

if args.refdir:
    metrics['FID'] = fid_metric(args.indir, args.refdir, mode='legacy_pytorch')

logger.info(f"MEAN CLIPIQA: {metrics['CLIPIQA']:6.4f}")
logger.info(f"MEAN MUSIQ: {metrics['MUSIQ']:6.4f}")
if args.pi:
    logger.info(f"MEAN PI: {metrics['PI']:6.4f}")
if args.niqe:
    logger.info(f"MEAN NIQE: {metrics['NIQE']:6.4f}")
if args.maniqa:
    logger.info(f"MEAN MANIQA: {metrics['MANIQA']:6.4f}")
if args.tres:
    logger.info(f"MEAN TRES: {metrics['TRES']:6.4f}")
if args.dbcnn:
    logger.info(f"MEAN DBCNN: {metrics['DBCNN']:6.4f}")
if args.refdir:
    logger.info(f"MEAN FID: {metrics['FID']:6.4f}")

