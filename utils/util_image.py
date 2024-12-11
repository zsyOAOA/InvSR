#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 16:54:19

import sys
import cv2
import math
import torch
import random
import numpy as np
from scipy import fft
from pathlib import Path
from einops import rearrange
from skimage import img_as_ubyte, img_as_float32

# --------------------------Metrics----------------------------
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(im1, im2, border=0, ycbcr=False):
    '''
    SSIM the same outputs as MATLAB's
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    if im1.ndim == 2:
        return ssim(im1, im2)
    elif im1.ndim == 3:
        if im1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(im1[:,:,i], im2[:,:,i]))
            return np.array(ssims).mean()
        elif im1.shape[2] == 1:
            return ssim(np.squeeze(im1), np.squeeze(im2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    '''
    PSNR metric.
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def normalize_np(im, mean=0.5, std=0.5, reverse=False):
    '''
    Input:
        im: h x w x c, numpy array
        Normalize: (im - mean) / std
        Reverse: im * std + mean

    '''
    if not isinstance(mean, (list, tuple)):
        mean = [mean, ] * im.shape[2]
    mean = np.array(mean).reshape([1, 1, im.shape[2]])

    if not isinstance(std, (list, tuple)):
        std = [std, ] * im.shape[2]
    std = np.array(std).reshape([1, 1, im.shape[2]])

    if not reverse:
        out = (im.astype(np.float32) - mean) / std
    else:
        out = im.astype(np.float32) * std + mean
    return out

def normalize_th(im, mean=0.5, std=0.5, reverse=False):
    '''
    Input:
        im: b x c x h x w, torch tensor
        Normalize: (im - mean) / std
        Reverse: im * std + mean

    '''
    if not isinstance(mean, (list, tuple)):
        mean = [mean, ] * im.shape[1]
    mean = torch.tensor(mean, device=im.device).view([1, im.shape[1], 1, 1])

    if not isinstance(std, (list, tuple)):
        std = [std, ] * im.shape[1]
    std = torch.tensor(std, device=im.device).view([1, im.shape[1], 1, 1])

    if not reverse:
        out = (im - mean) / std
    else:
        out = im * std + mean
    return out

# ------------------------Image format--------------------------
def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: uint8 [0,255] or float [0,1]
        only_y: only return Y channel
    '''
    # transform to float64 data type, range [0, 255]
    if im.dtype == np.uint8:
        im_temp = im.astype(np.float64)
    else:
        im_temp = (im * 255).astype(np.float64)

    # convert
    if only_y:
        rlt = np.dot(im_temp, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im_temp, np.array([[65.481,  -37.797, 112.0  ],
                                           [128.553, -74.203, -93.786],
                                           [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if im.dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(im.dtype)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    # transform to range [0,255.0]
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        scale = torch.tensor(
            [[65.481,  -37.797, 112.0  ],
             [128.553, -74.203, -93.786],
             [24.966,  112.0,   -18.214]],
            device=im.device, dtype=im.dtype
        ) / 255.0
        bias = torch.tensor([16, 128, 128], device=im.device, dtype=im.dtype).view([-1, 1, 1, 3])
        rlt = torch.matmul(im_temp, scale) + bias

    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def ycbcr2rgbTorch(im):
    '''
    same as matlab ycbcr2rgb
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    # transform to range [0,255.0]
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C
    # convert
    scale = torch.tensor(
        [[0.00456621, 0.00456621, 0.00456621],
         [0, -0.00153632, 0.00791071],
         [0.00625893, -0.00318811, 0]],
        device=im.device, dtype=im.dtype
        ) * 255.0
    bias = torch.tensor(
        [-222.921, 135.576, -276.836], device=im.device, dtype=im.dtype
    ).view([-1, 1, 1, 3])
    rlt = torch.matmul(im_temp, scale) + bias
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    flag_tensor = torch.is_tensor(tensor)
    if flag_tensor:
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1 and flag_tensor:
        result = result[0]
    return result

def img2tensor(imgs, bgr2rgb=False, out_type=torch.float32):
    """Convert image numpy arrays into torch tensor.
    Args:
        imgs (Array or list[array]): Accept shapes:
            3) list of numpy arrays
            1) 3D numpy array of shape (H x W x 3/1);
            2) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.

    Returns:
        (array or list): 4D ndarray of shape (1 x C x H x W)
    """

    def _img2tensor(img):
        if img.ndim == 2:
            tensor = torch.from_numpy(img[None, None,]).type(out_type)
        elif img.ndim == 3:
            if bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rearrange(img, 'h w c -> c h w')).type(out_type).unsqueeze(0)
        else:
            raise TypeError(f'2D or 3D numpy array expected, got{img.ndim}D array')
        return tensor

    if not (isinstance(imgs, np.ndarray) or (isinstance(imgs, list) and all(isinstance(t, np.ndarray) for t in imgs))):
        raise TypeError(f'Numpy array or list of numpy array expected, got {type(imgs)}')

    flag_numpy = isinstance(imgs, np.ndarray)
    if flag_numpy:
        imgs = [imgs,]
    result = []
    for _img in imgs:
        result.append(_img2tensor(_img))

    if len(result) == 1 and flag_numpy:
        result = result[0]
    return result

# ------------------------Image resize-----------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

# ------------------------Image I/O-----------------------------
def imread(path, chn='rgb', dtype='float32', force_gray2rgb=True, force_rgba2rgb=False):
    '''
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    out:
        im: h x w x c, numpy tensor
    '''
    try:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    except:
        print(str(path))

    if im is None:
        print(str(path))

    if chn.lower() == 'gray':
        assert im.ndim == 2, f"{str(path)} can't be successfuly read!"
    else:
        if im.ndim == 2:
            if force_gray2rgb:
                im = np.stack([im, im, im], axis=2)
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        elif im.ndim == 4:
            if force_rgba2rgb:
                im = im[:, :, :3]
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        else:
            if chn.lower() == 'rgb':
                im = bgr2rgb(im)
            elif chn.lower() == 'bgr':
                pass

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im

def imwrite(im_in, path, chn='rgb', dtype_in='float32', qf=None):
    '''
    Save image.
    Input:
        im: h x w x c, numpy tensor
        path: the saving path
        chn: the channel order of the im,
    '''
    im = im_in.copy()
    if isinstance(path, str):
        path = Path(path)
    if dtype_in != 'uint8':
        im = img_as_ubyte(im)

    if chn.lower() == 'rgb' and im.ndim == 3:
        im = rgb2bgr(im)

    if qf is not None and path.suffix.lower() in ['.jpg', '.jpeg']:
        flag = cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    else:
        flag = cv2.imwrite(str(path), im)

    return flag

def jpeg_compress(im, qf, chn_in='rgb'):
    '''
    Input:
        im: h x w x 3 array
        qf: compress factor, (0, 100]
        chn_in: 'rgb' or 'bgr'
    Return:
        Compressed Image with channel order: chn_in
    '''
    # transform to BGR channle and uint8 data type
    im_bgr = rgb2bgr(im) if chn_in.lower() == 'rgb' else im
    if im.dtype != np.dtype('uint8'): im_bgr = img_as_ubyte(im_bgr)

    # JPEG compress
    flag, encimg = cv2.imencode('.jpg', im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    assert flag
    im_jpg_bgr = cv2.imdecode(encimg, 1)    # uint8, BGR

    # transform back to original channel and the original data type
    im_out = bgr2rgb(im_jpg_bgr) if chn_in.lower() == 'rgb' else im_jpg_bgr
    if im.dtype != np.dtype('uint8'): im_out = img_as_float32(im_out).astype(im.dtype)
    return im_out

# ------------------------Augmentation-----------------------------
def data_aug_np(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out.copy()

def inverse_data_aug_np(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

# ----------------------Visualization----------------------------
def imshow(x, title=None, cbar=False):
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def imblend_with_mask(im, mask, alpha=0.25):
    """
    Input:
        im, mask: h x w x c numpy array, uint8, [0, 255]
        alpha: scaler in [0.0, 1.0]
    """
    edge_map = cv2.Canny(mask, 100, 200).astype(np.float32)[:, :, None] / 255.

    assert mask.dtype == np.uint8
    mask = mask.astype(np.float32) / 255.
    if mask.ndim == 2:
        mask = mask[:, :, None]

    back_color = np.array([159, 121, 238], dtype=np.float32).reshape((1,1,3))
    blend = im.astype(np.float32) * alpha + (1 - alpha) * back_color
    blend = np.clip(blend, 0, 255)
    out = im.astype(np.float32) * (1 - mask) + blend * mask

    # paste edge
    out = out * (1 - edge_map) + np.array([0,255,0], dtype=np.float32).reshape((1,1,3)) * edge_map

    return out.astype(np.uint8)
# -----------------------Covolution------------------------------
def imgrad(im, pading_mode='mirror'):
    '''
    Calculate image gradient.
    Input:
        im: h x w x c numpy array
    '''
    from scipy.ndimage import correlate  # lazy import
    wx = np.array([[0, 0, 0],
                   [-1, 1, 0],
                   [0, 0, 0]], dtype=np.float32)
    wy = np.array([[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=np.float32)
    if im.ndim == 3:
        gradx = np.stack(
                [correlate(im[:,:,c], wx, mode=pading_mode) for c in range(im.shape[2])],
                axis=2
                )
        grady = np.stack(
                [correlate(im[:,:,c], wy, mode=pading_mode) for c in range(im.shape[2])],
                axis=2
                )
        grad = np.concatenate((gradx, grady), axis=2)
    else:
        gradx = correlate(im, wx, mode=pading_mode)
        grady = correlate(im, wy, mode=pading_mode)
        grad = np.stack((gradx, grady), axis=2)

    return {'gradx': gradx, 'grady': grady, 'grad':grad}

def imgrad_fft(im):
    '''
    Calculate image gradient.
    Input:
        im: h x w x c numpy array
    '''
    wx = np.rot90(np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]], dtype=np.float32), k=2)
    gradx = convfft(im, wx)
    wy = np.rot90(np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32), k=2)
    grady = convfft(im, wy)
    grad = np.concatenate((gradx, grady), axis=2)

    return {'gradx': gradx, 'grady': grady, 'grad':grad}

def convfft(im, weight):
    '''
    Convolution with FFT
    Input:
        im: h1 x w1 x c numpy array
        weight: h2 x w2 numpy array
    Output:
        out: h1 x w1 x c numpy array
    '''
    axes = (0,1)
    otf = psf2otf(weight, im.shape[:2])
    if im.ndim == 3:
        otf = np.tile(otf[:, :, None], (1,1,im.shape[2]))
    out = fft.ifft2(fft.fft2(im, axes=axes) * otf, axes=axes).real
    return out

def psf2otf(psf, shape):
    """
    MATLAB psf2otf function.
    Borrowed from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py.
    Input:
        psf : h x w numpy array
        shape : list or tuple, output shape of the OTF array
    Output:
        otf : OTF array with the desirable shape
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def convtorch(im, weight, mode='reflect'):
    '''
    Image convolution with pytorch
    Input:
        im: b x c_in x h x w torch tensor
        weight: c_out x c_in x k x k torch tensor
    Output:
        out: c x h x w torch tensor
    '''
    radius = weight.shape[-1]
    chn = im.shape[1]
    im_pad = torch.nn.functional.pad(im, pad=(radius // 2, )*4, mode=mode)
    out = torch.nn.functional.conv2d(im_pad, weight, padding=0, groups=chn)
    return out

# ----------------------Patch Cropping----------------------------
def random_crop(im, pch_size):
    '''
    Randomly crop a patch from the give image.
    '''
    h, w = im.shape[:2]
    # padding if necessary
    if h < pch_size or w < pch_size:
        pad_h = min(max(0, pch_size - h), h)
        pad_w = min(max(0, pch_size - w), w)
        im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    h, w = im.shape[:2]
    if h == pch_size:
        ind_h = 0
    elif h > pch_size:
        ind_h = random.randint(0, h-pch_size)
    else:
        raise ValueError('Image height is smaller than the patch size')
    if w == pch_size:
        ind_w = 0
    elif w > pch_size:
        ind_w = random.randint(0, w-pch_size)
    else:
        raise ValueError('Image width is smaller than the patch size')

    im_pch = im[ind_h:ind_h+pch_size, ind_w:ind_w+pch_size,]

    return im_pch

class ToTensor:
    def __init__(self, max_value=1.0):
        self.max_value = max_value

    def __call__(self, im):
        assert isinstance(im, np.ndarray)
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
        if im.dtype == np.uint8:
            assert self.max_value == 255.
            out = torch.from_numpy(im.astype(np.float32).transpose(2,0,1) / self.max_value)
        else:
            assert self.max_value == 1.0
            out = torch.from_numpy(im.transpose(2,0,1))
        return out

class RandomCrop:
    def __init__(self, pch_size, pass_crop=False):
        self.pch_size = pch_size
        self.pass_crop = pass_crop

    def __call__(self, im):
        if self.pass_crop:
            return im
        if isinstance(im, list) or isinstance(im, tuple):
            out = []
            for current_im in im:
                out.append(random_crop(current_im, self.pch_size))
        else:
            out = random_crop(im, self.pch_size)
        return out

class ImageSpliterNp:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: h x w x c, numpy array, [0, 1], low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        if im.ndim == 2:
            im = im[:, :, None]

        height, width, chn = im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = np.zeros([height*sf, width*sf, chn], dtype=im.dtype)
        self.pixel_count = np.zeros([height*sf, width*sf, chn], dtype=im.dtype)

    def extract_starts(self, length):
        starts = list(range(0, length, self.stride))
        if starts[-1] + self.pch_size > length:
            starts[-1] = length - self.pch_size
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx] * self.sf
            w_end = w_start + self.pch_size * self.sf

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx] * self.sf
            h_end = h_start + self.pch_size * self.sf

            pch = self.im_ori[h_start:h_end, w_start:w_end,]
            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration(0)

        return pch, (h_start, h_end, w_start, w_end)

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: pch_size x pch_size x 3, [0,1]
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert np.all(self.pixel_count != 0)
        return self.im_res / self.pixel_count

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1, weight_type='Gaussian'):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
            pch_bs: aggregate pchs to processing, only used when inputing single image
        '''
        assert weight_type in ['Gaussian', 'ones']
        self.weight_type = weight_type
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width= im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.dtype = torch.float64
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=self.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=self.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[self.count_pchs:self.count_pchs+self.extra_bs]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch =  current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii].type(self.dtype)
            current_weight = self.get_weight(current_pch.shape[-2], current_pch.shape[-1])
            self.im_res[:, :, h_start:h_end, w_start:w_end] +=  current_pch * current_weight
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += current_weight

    @staticmethod
    def generate_kernel_1d(ksize):
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # opencv default setting
        if ksize % 2 == 0:
            kernel = cv2.getGaussianKernel(ksize=ksize+1, sigma=sigma, ktype=cv2.CV_64F)
            kernel = kernel[1:, ]
        else:
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_64F)

        return kernel

    def get_weight(self, height, width):
        if self.weight_type == 'ones':
            kernel = torch.ones(1, 1, height, width)
        elif self.weight_type == 'Gaussian':
            kernel_h = self.generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self.generate_kernel_1d(width).reshape(1, -1)
            kernel = np.matmul(kernel_h, kernel_w)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0) # 1 x 1 x pch_size x pch_size
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")

        return kernel.to(dtype=self.dtype, device=self.im_ori.device)

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

# ----------------------Patch Cliping----------------------------
class Clamper:
    def __init__(self, min_max=(-1, 1)):
        self.min_bound, self.max_bound = min_max[0], min_max[1]

    def __call__(self, im):
        if isinstance(im, np.ndarray):
            return np.clip(im, a_min=self.min_bound, a_max=self.max_bound)
        elif isinstance(im, torch.Tensor):
            return torch.clamp(im, min=self.min_bound, max=self.max_bound)
        else:
            raise TypeError(f'ndarray or Tensor expected, got {type(im)}')

# ----------------------Interpolation----------------------------
class Bicubic:
    def __init__(self, scale=None, out_shape=None, activate_matlab=True, resize_back=False):
        self.scale = scale
        self.activate_matlab = activate_matlab
        self.out_shape = out_shape
        self.resize_back = resize_back

    def __call__(self, im):
        if self.activate_matlab:
            out = imresize_np(im, scale=self.scale)
            if self.resize_back:
                out = imresize_np(out, scale=1/self.scale)
        else:
            out = cv2.resize(
                    im,
                    dsize=self.out_shape,
                    fx=self.scale,
                    fy=self.scale,
                    interpolation=cv2.INTER_CUBIC,
                    )
            if self.resize_back:
                out = cv2.resize(
                        out,
                        dsize=self.out_shape,
                        fx=1/self.scale,
                        fy=1/self.scale,
                        interpolation=cv2.INTER_CUBIC,
                        )
        return out

class SmallestMaxSize:
    def __init__(self, max_size, pass_resize=False, interpolation=None):
        self.pass_resize = pass_resize
        self.max_size = max_size
        self.interpolation = interpolation
        self.str2mode = {
                'nearest': cv2.INTER_NEAREST_EXACT,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC
                }
        if self.interpolation is not None:
            assert interpolation in self.str2mode, f"Not supported interpolation mode: {interpolation}"

    def get_interpolation(self, size):
        if self.interpolation is None:
            if size < self.max_size:   # upsampling
                interpolation = cv2.INTER_CUBIC
            else:                      # downsampling
                interpolation = cv2.INTER_AREA
        else:
            interpolation = self.str2mode[self.interpolation]

        return interpolation

    def __call__(self, im):
        h, w = im.shape[:2]
        if self.pass_resize or min(h, w) == self.max_size:
            out = im
        else:
            if h < w:
                dsize = (int(self.max_size * w / h), self.max_size)
                out = cv2.resize(im, dsize=dsize, interpolation=self.get_interpolation(h))
            else:
                dsize = (self.max_size, int(self.max_size * h / w))
                out = cv2.resize(im, dsize=dsize, interpolation=self.get_interpolation(w))
            if out.dtype == np.uint8:
                out = np.clip(out, 0, 255)
            else:
                out = np.clip(out, 0, 1.0)

        return out

# ----------------------augmentation----------------------------
class SpatialAug:
    def __init__(self, pass_aug, only_hflip=False, only_vflip=False, only_hvflip=False):
        self.only_hflip = only_hflip
        self.only_vflip = only_vflip
        self.only_hvflip = only_hvflip
        self.pass_aug = pass_aug

    def __call__(self, im, flag=None):
        if self.pass_aug:
            return im

        if flag is None:
            if self.only_hflip:
                flag = random.choice([0, 5])
            elif self.only_vflip:
                flag = random.choice([0, 1])
            elif self.only_hvflip:
                flag = random.choice([0, 1, 5])
            else:
                flag = random.randint(0, 7)

        if isinstance(im, list) or isinstance(im, tuple):
            out = []
            for current_im in im:
                out.append(data_aug_np(current_im, flag))
        else:
            out = data_aug_np(im, flag)
        return out

if __name__ == '__main__':
    im = np.random.randn(64, 64, 3).astype(np.float32)

    grad1 = imgrad(im)['grad']
    grad2 = imgrad_fft(im)['grad']

    error = np.abs(grad1 -grad2).max()
    mean_error = np.abs(grad1 -grad2).mean()
    print('The largest error is {:.2e}'.format(error))
    print('The mean error is {:.2e}'.format(mean_error))
