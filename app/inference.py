""" Generate and return image """

import random
from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key, shard
from vqgan_jax.modeling_flax_vqgan import VQModel
import numpy as np
from PIL import Image
from tqdm.notebook import trange
import wandb
from dalle_mini import DalleBart, DalleBartProcessor
from transformers import CLIPProcessor, FlaxCLIPModel
from .helpers import store_images


@dataclass
class Inference:
    """ Inference pipeline """

    text_prompt: str = field(
        default = None,
        metadata = {"help": "Text inputed by user."},
    )
    imgs: bytearray = field(
        default = None,
        metadata = {"help": "Generated images."}
    )

    def __init__(self, text_prompt):
        self.text_prompt = text_prompt
        assert (
            self.text_prompt is not None
        ), "Nothing is inputed."


    def generate_image(text_prompt):
        """ Take text prompt and return generated image """
        # Model to generate image tokens
        MODEL = "fedorajuandy/dalle-mini/model-st6x232l:v26"
        MODEL_COMMIT_ID = None

        # VQGAN to decode image tokens
        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

        login_wandb()

        # load models
        model, params = DalleBart.from_pretrained(
            MODEL, revision=MODEL_COMMIT_ID, dtype=jnp.float32, _do_init=False
        )

        vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        # replicate parameters to each device
        params = replicate(params)
        vqgan_params = replicate(vqgan_params)

        # functions are parallelised to each device
        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, TEMPERATURE, condition_scale
        ):
            """ Model inference """
            return model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                TEMPERATURE=TEMPERATURE,
                condition_scale=condition_scale,
            )

        @partial(jax.pmap, axis_name="batch")
        def p_decode(indices, params):
            """ Decode image tokens """
            return vqgan.decode_code(indices, params=params)

        # generate key that is passed to each device to generate different images
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

        processor = DalleBartProcessor.from_pretrained(MODEL, revision=MODEL_COMMIT_ID)

        tokenized_prompts = processor(text_prompt)
        # replicate prompts to each device
        tokenized_prompt = replicate(tokenized_prompts)

        # number of predictions
        N_PREDICTIONS = 2

        # generetion parameters
        GEN_TOP_K = None
        GEN_TOP_P = None
        TEMPERATURE = None
        COND_SCALE = 10.0

        # generate images
        images = []
        for i in trange(max(N_PREDICTIONS // jax.device_count(), 1)):
            key, subkey = jax.random.split(key)
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                params,
                GEN_TOP_K,
                GEN_TOP_P,
                TEMPERATURE,
                COND_SCALE,
            )
            # remove BOS token
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
                # display(img)
                # print()
                store_images()

        return images


    def score_images(imgs, text_prompt):
        """ Take generated images, score them, and return highest scored one """
        # CLIP
        CLIP_REPO = "openai/clip-vit-base-patch32"
        CLIP_COMMIT_ID = None

        # Load model
        clip, clip_params = FlaxCLIPModel.from_pretrained(
            CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
        clip_params = replicate(clip_params)

        # Score images
        @partial(jax.pmap, axis_name="batch")
        def p_clip(inputs, params):
            logits = clip(params=params, **inputs).logits_per_image
            return logits

        # Get scores
        clip_inputs = clip_processor(
            text=text_prompt * jax.device_count(),
            images=imgs,
            return_tensors="np",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).data
        logits = p_clip(shard(clip_inputs), clip_params)

        # Organize scores
        p = len(text_prompt)
        logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()

        for i, prompt in enumerate(text_prompt):
            for idx in logits[i].argsort()[::-1]:
                display(images[idx * p + i])
                print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
            # print()

        highest_scored = ""

        return highest_scored
