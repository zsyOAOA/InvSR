# Arbitrary-steps Image Super-resolution via Diffusion Inversion

[Zongsheng Yue](https://zsyoaoa.github.io/), [Kang Liao](https://kangliao929.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

<!--[Paper](https://arxiv.org/abs/2307.12348) | [Project Page](https://zsyoaoa.github.io/projects/resshift/) | [Demo](https://www.youtube.com/watch?v=8DB-6Xvvl5o)-->

<!--<a href="https://colab.research.google.com/drive/1CL8aJO7a_RA4MetanrCLqQO5H7KWO8KI?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> [![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/cjwbw/resshift) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=zsyOAOA/ResShift) -->


:star: If InvSR is helpful to your researches or projects, please help star this repo. Thanks! :hugs: 

---
>This study presents a new image super-resolution (SR) technique based on diffusion inversion, aiming at harnessing the rich image priors encapsulated in large pre-trained diffusion models to improve SR performance. We design a \textit{Partial noise Prediction} strategy to construct an intermediate state of the diffusion model, which serves as the starting sampling point. Central to our approach is a deep noise predictor to estimate the optimal noise maps for the forward diffusion process. Once trained, this noise predictor can be used to initialize the sampling process partially along the diffusion trajectory, generating the desirable high-resolution result. Compared to existing approaches, our method offers a flexible and efficient sampling mechanism that supports an arbitrary number of sampling steps, ranging from one to five. Even with a single sampling step, our method demonstrates superior or comparable performance to recent state-of-the-art approaches.
><img src="./assets/framework.png" align="middle" width="800">
---
## Update
- **2024.12.11**: Create this repo.

## Requirements
* Python 3.10, Pytorch 2.4.0, [xformers](https://github.com/facebookresearch/xformers) 0.0.27.post2
* More detail (See [environment.yaml](environment.yaml))
* A suitable [conda](https://conda.io/) environment named `invsr` can be created and activated with:

```
conda create -n invsr python=3.10
conda activate invsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"
pip install -r requirements.txt
```

## Applications
### :point_right: Real-world Image Super-resolution
[<img src="assets/real-7.png" height="235"/>](https://imgsli.com/MzI2MTU5) [<img src="assets/real-1.png" height="235"/>](https://imgsli.com/MzI2MTUx) [<img src="assets/real-2.png" height="235"/>](https://imgsli.com/MzI2MTUy)
[<img src="assets/real-4.png" height="361"/>](https://imgsli.com/MzI2MTU0) [<img src="assets/real-6.png" height="361"/>](https://imgsli.com/MzI2MTU3) [<img src="assets/real-5.png" height="361"/>](https://imgsli.com/MzI2MTU1)

### :point_right: General Image Inhancement
[<img src="assets/enhance-1.png" height="246.5"/>](https://imgsli.com/MzI2MTYw) [<img src="assets/enhance-2.png" height="246.5"/>](https://imgsli.com/MzI2MTYy) 
[<img src="assets/enhance-3.png" height="207"/>](https://imgsli.com/MzI2MjAx) [<img src="assets/enhance-4.png" height="207"/>](https://imgsli.com/MzI2MjAz) [<img src="assets/enhance-5.png" height="207"/>](https://imgsli.com/MzI2MjA0)

### :point_right: AIGC Image Inhancement
[<img src="assets/sdxl-1.png" height="272"/>](https://imgsli.com/MzI2MjQy) [<img src="assets/sdxl-2.png" height="272"/>](https://imgsli.com/MzI2MjQ1) [<img src="assets/sdxl-3.png" height="272"/>](https://imgsli.com/MzI2MjQ3)
[<img src="assets/flux-1.png" height="272"/>](https://imgsli.com/MzI2MjQ5) [<img src="assets/flux-2.png" height="272"/>](https://imgsli.com/MzI2MjUw) [<img src="assets/flux-3.png" height="272"/>](https://imgsli.com/MzI2MjUx)

<!--## Online Demo-->
<!--You can try our method through an online demo:-->
<!--```-->
<!--python app.py-->
<!--```-->

## Inference
### :rocket: Fast testing 
```
python inference_invsr.py -i [image folder/image path] -o [result folder] --num_steps 1
```
1. This script will automatically download the pre-trained [noise predictor](https://huggingface.co/OAOA/InvSR/tree/main) and [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo/tree/main). If you have pre-downloaded them manually, please include them via ``--started_ckpt_path`` and ``--sd_path``.
2. To deal with large images, e.g., 1k---->4k, we recommend adding the option ``--chopping_size 256``.
3. You can freely adjust the sampling steps via ``--num_steps``.

### :airplane: Reproducing our paper results
+ Synthetic dataset of ImageNet-Test: [Google Drive](https://drive.google.com/file/d/1PRGrujx3OFilgJ7I6nW7ETIR00wlAl2m/view?usp=sharing).

+ Real data for image super-resolution: [RealSRV3](https://github.com/csjcai/RealSR) | [RealSet80](testdata/RealSet80)

+ To reproduce the quantitative results on Imagenet-Test and RealSRV3, please add the color fixing options by ``--color_fix wavelet``.

## Training
### :turtle: Preparing stage
1. Download the finetuned LPIPS model from this [link](https://huggingface.co/OAOA/InvSR/resolve/main/vgg16_sdturbo_lpips.pth?download=true) and put it in the folder of "weights".
2. Prepare the [config](configs/sd-turbo-sr-ldis.yaml) file:
    + SD-Turbo path: configs.sd_pipe.params.cache_dir.
    + Training data path: data.train.params.data_source.
    + Validation data path: data.val.params.dir_path (low-quality image) and data.val.params.extra_dir_path (high-quality image).
    + Batchsize: configs.train.batch and configs.train.microbatch (total batchsize = microbatch * #GPUS * num_grad_accumulation)

### :dolphin: Begin training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --save_dir [Logging Folder] 
```

### :whale: Resume from interruption
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --save_dir [Logging Folder] --resume save_dir/ckpts/model_xx.pth
```

## License

This project is licensed under [NTU S-Lab License 1.0](LICENSE). Redistribution and use should follow this license.

## Acknowledgement

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [diffusers](https://github.com/huggingface/diffusers). Thanks for their awesome works.

### Contact
If you have any questions, please feel free to contact me via `zsyzam@gmail.com`.
