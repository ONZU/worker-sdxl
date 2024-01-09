INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'K_EULER'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'upscaler_inference_steps': {
        'type': int,
        'required': False,
        'default': 20
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 9.
    },
    'upscaler_guidance_scale': {
        'type': float,
        'required': False,
        'default': 0.
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'high_noise_frac': {
        'type': float,
        'required': False,
        'default': None
    },
}
