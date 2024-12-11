import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path

import albumentations

import torch.nn.functional as F
from torch.utils import data as data

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

from utils import util_image

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, mode='training'):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # file client (lmdb io backend)
        self.image_paths = []
        self.text_paths = []
        self.moment_paths = []
        if opt.get('data_source', None) is not None:
            for ii in range(len(opt['data_source'])):
                configs = opt['data_source'].get(f'source{ii+1}')
                root_path = Path(configs.root_path)
                im_folder = root_path / configs.image_path
                im_ext = configs.im_ext
                image_stems = sorted([x.stem for x in im_folder.glob(f"*.{im_ext}")])
                if configs.get('length', None) is not None:
                    assert configs.length < len(image_stems)
                    image_stems = image_stems[:configs.length]

                if configs.get("text_path", None) is not None:
                    text_folder = root_path / configs.text_path
                    text_stems = [x.stem for x in text_folder.glob("*.txt")]
                    image_stems = sorted(list(set(image_stems).intersection(set(text_stems))))
                    self.text_paths.extend([str(text_folder / f"{x}.txt") for x in image_stems])
                else:
                    self.text_paths.extend([None, ] * len(image_stems))

                self.image_paths.extend([str(im_folder / f"{x}.{im_ext}") for x in image_stems])

                if configs.get("moment_path", None) is not None:
                    moment_folder = root_path / configs.moment_path
                    self.moment_paths.extend([str(moment_folder / f"{x}.npy") for x in image_stems])
                else:
                    self.moment_paths.extend([None, ] * len(image_stems))

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21
        self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(opt['blur_kernel_size2'], opt['blur_kernel_size2']).float()
        self.pulse_tensor[opt['blur_kernel_size2']//2, opt['blur_kernel_size2']//2] = 1

        self.mode = mode

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.image_paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                index = random.randint(0, self.__len__())
                gt_path = self.image_paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            finally:
                retry -= 1
        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(
                        max_size=self.opt['gt_size'],
                        interpolation=cv2.INTER_AREA,
                        ),
                    albumentations.CenterCrop(self.opt['gt_size'], self.opt['gt_size']),
                    ])
            img_gt = self.test_aug(image=img_gt)['image']
        elif self.mode == 'training':
            # -------------------- Do augmentation for training: flip, rotation -------------------- #
            if self.opt['use_hflip'] or self.opt['use_rot']:
                img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

            h, w = img_gt.shape[0:2]
            gt_size = self.opt['gt_size']

            # resize or pad
            if not self.opt['random_crop']:
                if not min(h, w) == gt_size:
                    if not hasattr(self, 'smallest_resizer'):
                        self.smallest_resizer = util_image.SmallestMaxSize(
                            max_size=gt_size, pass_resize=False,
                        )
                    img_gt = self.smallest_resizer(img_gt)

                # center crop
                if not hasattr(self, 'center_cropper'):
                    self.center_cropper = albumentations.CenterCrop(gt_size, gt_size)
                img_gt = self.center_cropper(image=img_gt)['image']
            else:
                img_gt = random_crop(img_gt, self.opt['gt_size'])
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        if self.text_paths[index] is None or self.opt['random_crop']:
            prompt = ""
        else:
            with open(self.text_paths[index], 'r') as ff:
                prompt = ff.read()
            if self.opt.max_token_length is not None:
                prompt = prompt[:self.opt.max_token_length]

        return_d = {
                'gt': img_gt,
                'gt_path': gt_path,
                'txt': prompt,
                'kernel1': kernel,
                'kernel2': kernel2,
                'sinc_kernel': sinc_kernel,
                }
        if self.moment_paths[index] is not None and (not self.opt['random_crop']):
            return_d['gt_moment'] = np.load(self.moment_paths[index])

        return return_d

    def __len__(self):
        return len(self.image_paths)

    def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts

        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                conf_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < conf_degradation['second_order_prob']:
            if random.random() < conf_degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    conf_degradation['resize_prob2'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, conf_degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(conf_degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                    mode=mode,
                    )
            # add noise
            gray_noise_prob = conf_degradation['gray_noise_prob2']
            if random.random() < conf_degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=conf_degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=conf_degradation['poisson_scale_range2'],
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
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
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

        # clamp and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq':im_lq.contiguous(), 'gt':im_gt}
