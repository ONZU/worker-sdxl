'''
Contains the handler function that will be called by the serverless.
'''

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.configuration_utils import FrozenDict


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.upscaler = None
        self.load_models()

    @staticmethod
    def load_base() -> StableDiffusionXLPipeline:
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            '../builder/juggernautXL_version6Rundiffusion.safetensors',
            torch_dtype=torch.float16, variant="fp16", )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    @staticmethod
    def load_refiner() -> StableDiffusionXLImg2ImgPipeline:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16, local_files_only=True
        )
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, add_watermarker=False, local_files_only=True
        )
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    @staticmethod
    def load_upscaler() -> StableDiffusionLatentUpscalePipeline:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                        torch_dtype=torch.float16,
                                                                        use_safetensors=True, local_files_only=True)
        upscaler = upscaler.to("cuda", silence_dtype_warnings=True)
        upscaler.enable_xformers_memory_efficient_attention()
        return upscaler

    def load_models(self) -> None:
        self.base = self.load_base()
        self.refiner = self.load_refiner()
        self.upscaler = self.load_upscaler()


def make_scheduler(name: str, config: FrozenDict) -> object:
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(pos_prompt: str, neg_prompt: str, seed: int, model_handler: ModelHandler,
                   resolution: Tuple[int, int] = (1024, 1024)) -> Image:
    generator = torch.Generator("cuda").manual_seed(seed)

    # Generate latent image using pipe
    image = model_handler.base(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        height=resolution[0],
        width=resolution[1],
        num_inference_steps=25,
        guidance_scale=7.5,
        denoising_end=None,
        output_type="latent",
        num_images_per_prompt=1,
        generator=generator
    ).images

    output = model_handler.refiner(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=50,
        strength=0.3,
        image=image,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    output = model_handler.upscaler(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=output,
        num_inference_steps=20,
        guidance_scale=0.,
        generator=generator
    ).images[0]

    return output


def process_prompts(df_path: Path) -> None:
    torch.cuda.empty_cache()
    model_handler = ModelHandler()
    model_handler.base.scheduler = make_scheduler(
        'DDIM', model_handler.base.scheduler.config)

    prompts_df = pd.read_csv(df_path)
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)

    for i, row in prompts_df.iterrows():
        print(f'Processing row {i}')
        pos_prompt = row['positive_prompt']
        neg_prompt = row['negative_prompt']
        topic = row['topic']
        subtopic = row['subtopic']
        variant = row['variant']
        style = row['style']
        img_base_name = f'{topic}_{subtopic}_{variant}_{style}'
        for seed in np.random.randint(0, 1000, size=3, dtype=np.int32):
            img = generate_image(pos_prompt, neg_prompt, int(seed), model_handler, (1024, 576))
            img_name = f'{img_base_name}_{seed}.png'
            img.save(out_dir / img_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--df_path", type=Path, required=True)
    args = parser.parse_args()

    process_prompts(args.df_path)
