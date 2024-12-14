# Prediction interface for Cog ⚙️
# https://cog.run/python


import shutil, os
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path

import numpy as np
from utils import util_common
from sampler_invsr import InvSamplerSR
from basicsr.utils.download_util import load_file_from_url

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.configs = OmegaConf.load("./configs/sample-sd-turbo.yaml")

    def set_configs(self, num_steps=1, chopping_size=128, seed=12345):
        if num_steps == 1:
            self.configs.timesteps = [200,]
        elif num_steps == 2:
            self.configs.timesteps = [200, 100]
        elif num_steps == 3:
            self.configs.timesteps = [200, 100, 50]
        elif num_steps == 4:
            self.configs.timesteps = [200, 150, 100, 50]
        elif num_steps == 5:
            self.configs.timesteps = [250, 200, 150, 100, 50]
        else:
            assert num_steps <= 250
            self.configs.timesteps = np.linspace(
                start=250, stop=0, num=num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
        print(f'Setting timesteps for inference: {self.configs.timesteps}')

        # path to save Stable Diffusion
        sd_path = "./weights"
        util_common.mkdir(sd_path, delete=False, parents=True)
        self.configs.sd_pipe.params.cache_dir = sd_path

        # path to save noise predictor
        started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
        started_ckpt_dir = "./weights"
        util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
        started_ckpt_path = os.path.join(started_ckpt_dir, started_ckpt_name)
        if not os.path.exists(started_ckpt_path):
            load_file_from_url(
                url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
                model_dir=started_ckpt_dir,
                progress=True,
                file_name=started_ckpt_name,
            )
        self.configs.model_start.ckpt_path = started_ckpt_path

        self.configs.bs = 1
        self.configs.seed = 12345
        self.configs.basesr.chopping.pch_size = chopping_size
        if chopping_size == 128:
            self.configs.basesr.chopping.extra_bs = 4
        elif chopping_size == 256:
            self.configs.basesr.chopping.extra_bs = 2
        else:
            self.configs.basesr.chopping.extra_bs = 1

    def predict(
        self,
        in_path: Path = Input(description="Input low-quality image"),
        num_steps: int = Input(
            choices=[1,2,3,4,5], description="Number of sampling steps.", default=1
        ),
        chopping_size: int = Input(
            choices=[128, 256, 512], description="Chopping resolution", default=128
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=12345
        ),
    ) -> Path:
        # setting configurations
        self.set_configs(num_steps, chopping_size, seed)

        sampler = InvSamplerSR(self.configs)

        out_dir = 'invsr_output'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        sampler.inference(in_path, out_path=out_dir, bs=1)

        out = "/tmp/out.png"
        shutil.copy(os.path.join(out_dir, os.listdir(out_dir)[0]), out)

        return Path(out)
