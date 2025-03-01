# Arbitrary-steps Image Super-resolution via Diffusion Inversion (CVPR 2025)

[Zongsheng Yue](https://zsyoaoa.github.io/), [Kang Liao](https://kangliao929.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.09013-b31b1b.svg)](https://arxiv.org/abs/2412.09013) [![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/zsyoaoa/invsr) [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/OAOA/InvSR) <a href="https://colab.research.google.com/drive/1hjgCFnAU4oUUhh9VRfTwsFN1AiIjdcSR?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> ![visitors](https://visitor-badge.laobi.icu/badge?page_id=zsyOAOA/InvSR) 

<!--[![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/cjwbw/resshift)-->


:star: If you've found InvSR useful for your research or projects, please show your support by starring this repo. Thanks! :hugs: 

---
>This study presents a new image super-resolution (SR) technique based on diffusion inversion, aiming at harnessing the rich image priors encapsulated in large pre-trained diffusion models to improve SR performance. We design a \textit{Partial noise Prediction} strategy to construct an intermediate state of the diffusion model, which serves as the starting sampling point. Central to our approach is a deep noise predictor to estimate the optimal noise maps for the forward diffusion process. Once trained, this noise predictor can be used to initialize the sampling process partially along the diffusion trajectory, generating the desirable high-resolution result. Compared to existing approaches, our method offers a flexible and efficient sampling mechanism that supports an arbitrary number of sampling steps, ranging from one to five. Even with a single sampling step, our method demonstrates superior or comparable performance to recent state-of-the-art approaches.
><img src="./assets/framework.png" align="middle" width="800">
---
## Update
- **2025.03.01**: InvSR has been accepted by CVPR 2025.
- **2025.01.08**: Update gradio demo for batch processing.
- **2024.12.14**: Add [![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/zsyoaoa/invsr).
- **2024.12.13**: Add [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/OAOA/InvSR) and <a href="https://colab.research.google.com/drive/1hjgCFnAU4oUUhh9VRfTwsFN1AiIjdcSR?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.
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

<!--### :point_right: General Image Enhancement-->
<!--[<img src="assets/enhance-1.png" height="246.5"/>](https://imgsli.com/MzI2MTYw) [<img src="assets/enhance-2.png" height="246.5"/>](https://imgsli.com/MzI2MTYy) -->
<!--[<img src="assets/enhance-3.png" height="207"/>](https://imgsli.com/MzI2MjAx) [<img src="assets/enhance-4.png" height="207"/>](https://imgsli.com/MzI2NTk1) [<img src="assets/enhance-5.png" height="207"/>](https://imgsli.com/MzI2MjA0)-->

### :point_right: AIGC Image Enhancement
[<img src="assets/sdxl-1.png" height="272"/>](https://imgsli.com/MzI2MjQy) [<img src="assets/sdxl-2.png" height="272"/>](https://imgsli.com/MzI2MjQ1) [<img src="assets/sdxl-3.png" height="272"/>](https://imgsli.com/MzI2MjQ3)
[<img src="assets/flux-1.png" height="272"/>](https://imgsli.com/MzI2MjQ5) [<img src="assets/flux-2.png" height="272"/>](https://imgsli.com/MzI2MjUw) [<img src="assets/flux-3.png" height="272"/>](https://imgsli.com/MzI2MjUx)


## Inference
### :rocket: Fast testing 
```
python inference_invsr.py -i [image folder/image path] -o [result folder] --num_steps 1
```
1. **To deal with large images, e.g., 1k---->4k, we recommend adding the option** ``--chopping_size 256``.
2. Other options:
    + Specify the pre-downloaded [SD Turbo](https://huggingface.co/stabilityai/sd-turbo) Model: ``--sd_path``.
    + Specify the pre-downloaded noise predictor: ``--started_ckpt_path``.
    + The number of sampling steps: ``--num_steps``.
    + If your GPU memory is limited, please add the option ``--chopping_bs 1``.

### :railway_car: Online Demo
You can try our method through an online demo:
```
python app.py
```

### :whale: Now also available in Docker
```bash
docker compose up -d # Go to http://127.0.0.1:7860/
```

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
