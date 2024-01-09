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
from diffusers import StableDiffusionXLPipeline
from diffusers.configuration_utils import FrozenDict
from transformers.models.clip import CLIPTextModel, CLIPTokenizer


class ModelHandler:
    def __init__(self):
        self.base = None
        self.upscaler = None
        self.load_models()

    @staticmethod
    def load_base() -> StableDiffusionXLPipeline:
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            '../builder/juggernautXL_version6Rundiffusion.safetensors',
            # torch_dtype=torch.float16, variant="fp16",
        )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    @staticmethod
    def load_upscaler() -> StableDiffusionLatentUpscalePipeline:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                        # torch_dtype=torch.float16,
                                                                        use_safetensors=True, local_files_only=True)
        upscaler = upscaler.to("cuda", silence_dtype_warnings=True)
        upscaler.enable_xformers_memory_efficient_attention()
        return upscaler

    def load_models(self) -> None:
        self.base = self.load_base()
        self.upscaler = self.load_upscaler()


def make_scheduler(name: str, config: FrozenDict) -> object:
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


def tokenize_and_embed(pos_prompt: str, neg_prompt: str,
                       tokenizer: CLIPTokenizer,
                       text_encoder: CLIPTextModel) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_encoder_length = tokenizer.model_max_length

    # positive and negative embeddings must have the same shape
    n_tokens_pos = len(tokenizer(pos_prompt).input_ids)
    n_tokens_neg = len(tokenizer(neg_prompt).input_ids)
    max_n_tokens = max(n_tokens_pos, n_tokens_neg)

    input_ids = tokenizer(pos_prompt, truncation=False, padding="max_length", max_length=max_n_tokens,
                          return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")

    negative_ids = tokenizer(neg_prompt, truncation=False, padding="max_length", max_length=max_n_tokens,
                             return_tensors="pt").input_ids
    negative_ids = negative_ids.to("cuda")

    concat_pos_embeds = []
    concat_pooled_pos_embeds = []
    concat_neg_embeds = []
    concat_pooled_neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_encoder_length):
        pos_embeds = text_encoder(input_ids[:, i: i + max_encoder_length], output_hidden_states=True)
        pos_embeds, pooled_pos_embeds = pos_embeds.hidden_states[-2], pos_embeds.hidden_states[-1]
        concat_pos_embeds.append(pos_embeds)
        concat_pooled_pos_embeds.append(pooled_pos_embeds)

        neg_embeds = text_encoder(negative_ids[:, i: i + max_encoder_length], output_hidden_states=True)
        neg_embeds, pooled_neg_embeds = neg_embeds.hidden_states[-2], neg_embeds.hidden_states[-1]
        concat_neg_embeds.append(neg_embeds)
        concat_pooled_neg_embeds.append(pooled_neg_embeds)

    prompt_embeds = torch.cat(concat_pos_embeds, dim=1)
    pooled_prompt_embeds = torch.cat(concat_pooled_pos_embeds, dim=1)
    pooled_prompt_embeds = text_encoder.base_model.text_model.final_layer_norm(pooled_prompt_embeds)
    pooled_prompt_embeds = pooled_prompt_embeds[
        torch.arange(pooled_prompt_embeds.shape[0], device=pooled_prompt_embeds.device),
        input_ids.to(dtype=torch.int, device=pooled_prompt_embeds.device).argmax(dim=-1),
    ]

    negative_prompt_embeds = torch.cat(concat_neg_embeds, dim=1)
    pooled_negative_prompt_embeds = torch.cat(concat_pooled_neg_embeds, dim=1)
    pooled_negative_prompt_embeds = text_encoder.base_model.text_model.final_layer_norm(pooled_negative_prompt_embeds)
    pooled_negative_prompt_embeds = pooled_negative_prompt_embeds[
        torch.arange(pooled_negative_prompt_embeds.shape[0], device=pooled_negative_prompt_embeds.device),
        negative_ids.to(dtype=torch.int, device=pooled_negative_prompt_embeds.device).argmax(dim=-1),
    ]

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds


def build_embeddings(pos_prompt: str, neg_prompt: str, model_handler: ModelHandler) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_embeds, neg_embeds, _, _ = tokenize_and_embed(pos_prompt, neg_prompt, model_handler.base.tokenizer,
                                                      model_handler.base.text_encoder)
    pos_embeds_2, neg_embeds_2, pooled_pos_embeds, pooled_neg_embeds = tokenize_and_embed(pos_prompt, neg_prompt,
                                                                                          model_handler.base.tokenizer_2,
                                                                                          model_handler.base.text_encoder_2)

    pos_embeds = torch.concat([pos_embeds, pos_embeds_2], dim=-1)
    neg_embeds = torch.concat([neg_embeds, neg_embeds_2], dim=-1)

    return pos_embeds, neg_embeds, pooled_pos_embeds, pooled_neg_embeds


@torch.inference_mode()
def generate_image(pos_prompt: str, neg_prompt: str, seed: int, model_handler: ModelHandler,
                   resolution: Tuple[int, int] = (1024, 1024)) -> Image:
    generator = torch.Generator("cuda").manual_seed(seed)

    pos_embeds, neg_embeds, pooled_pos_embeds, pooled_neg_embeds = build_embeddings(pos_prompt, neg_prompt,
                                                                                    model_handler)

    # Generate latent image using pipe
    output = model_handler.base(
        prompt_embeds=pos_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pooled_pos_embeds,
        negative_pooled_prompt_embeds=pooled_neg_embeds,
        height=resolution[0],
        width=resolution[1],
        num_inference_steps=50,
        guidance_scale=9,
        denoising_end=None,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]

    # output = model_handler.upscaler(
    #     prompt=pos_prompt,
    #     negative_prompt=neg_prompt,
    #     image=output,
    #     num_inference_steps=20,
    #     guidance_scale=0.,
    #     generator=generator
    # ).images[0]

    return output


def process_prompts(df_path: Path) -> None:
    torch.cuda.empty_cache()
    model_handler = ModelHandler()
    model_handler.base.scheduler = make_scheduler(
        'K_EULER', model_handler.base.scheduler.config)

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
