'''
Contains the handler function that will be called by the serverless.
'''

import concurrent.futures

import matplotlib.pyplot as plt
from diffusers.configuration_utils import FrozenDict
import numpy as np
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

torch.cuda.empty_cache()


# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.upscaler = None
        self.load_models()

    @staticmethod
    def load_base() -> StableDiffusionXLPipeline:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, add_watermarker=False
        )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    @staticmethod
    def load_refiner() -> StableDiffusionXLImg2ImgPipeline:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, add_watermarker=False
        )
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    @staticmethod
    def load_upscaler() -> StableDiffusionLatentUpscalePipeline:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                        torch_dtype=torch.float16,
                                                                        use_safetensors=True)
        upscaler = upscaler.to("cuda", silence_dtype_warnings=True)
        upscaler.enable_xformers_memory_efficient_attention()
        return upscaler

    def load_models(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)
            future_upscaler = executor.submit(self.load_upscaler)

            self.base = future_base.result()
            self.refiner = future_refiner.result()
            self.upscaler = future_upscaler.result()


MODELS = ModelHandler()


# ---------------------------------- Helper ---------------------------------- #


def make_scheduler(name: str, config: FrozenDict) -> object:
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(seed: int = 42, scheduler: str = "DDIM", **kwargs) -> Image:
    generator = torch.Generator("cuda").manual_seed(seed)

    MODELS.base.scheduler = make_scheduler(
        scheduler, MODELS.base.scheduler.config)

    # Generate latent image using pipe
    image = MODELS.base(
        prompt='astronaut riding a dinosaur',
        negative_prompt=None,
        height=1024,
        width=1024,
        num_inference_steps=25,
        guidance_scale=7.5,
        denoising_end=None,
        output_type="latent",
        num_images_per_prompt=1,
        generator=generator
    ).images

    output = MODELS.refiner(
        prompt='astronaut riding a dinosaur',
        num_inference_steps=50,
        strength=0.3,
        image=image,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    plt.imshow(np.array(output))
    plt.show()

    output = MODELS.upscaler(
        prompt='astronaut riding a dinosaur',
        image=output,
        num_inference_steps=20,
        guidance_scale=0.,
        generator=generator
    ).images[0]

    plt.imshow(np.array(output))
    plt.show()

    return output


if __name__ == "__main__":
    generate_image()
