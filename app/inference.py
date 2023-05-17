""" Generate and return image """

import random
from functools import partial
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key, shard
from vqgan_jax.modeling_flax_vqgan import VQModel
import numpy as np
from PIL import Image
from tqdm.notebook import trange
from dalle_mini import DalleBart, DalleBartProcessor
from transformers import CLIPProcessor, FlaxCLIPModel
# from .helpers import *


def generate_image(text_prompt):
    """ Take text prompt and return generated image """

    # Model to generate image tokens
    model = "fedorajuandy/dalle-mini/model-st6x232l:v26"
    model_commit_id = None

    # VQGAN to decode image tokens
    vqgan_repo = "dalle-mini/vqgan_imagenet_f16_16384"
    vqgan_commit_id = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    # load models
    model, params = DalleBart.from_pretrained(
        model, revision=model_commit_id, dtype=jnp.float32, _do_init=False
    )

    vqgan, vqgan_params = VQModel.from_pretrained(
        vqgan_repo, revision=vqgan_commit_id, _do_init=False
    )

    # replicate parameters to each device
    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # functions are parallelised to each device
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, condition_scale
    ):
        """ Model inference """
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            condition_scale=condition_scale,
        )

    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        """ Decode image tokens """
        return vqgan.decode_code(indices, params=params)

    # generate key that is passed to each device to generate different images
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    processor = DalleBartProcessor.from_pretrained(model, revision=model_commit_id)

    texts = []
    texts.append(text_prompt)
    tokenized_prompts = processor(texts)
    # replicate prompts to each device
    tokenized_prompt = replicate(tokenized_prompts)

    # number of predictions
    n_predictions = 8

    # generetion parameters
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    # generate images
    images = []
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        key, subkey = jax.random.split(key)
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            cond_scale,
        )
        # remove BOS token
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)

    # CLIP
    clip_repo = "openai/clip-vit-base-patch32"
    clip_commit_id = None

    # Load model
    clip, clip_params = FlaxCLIPModel.from_pretrained(
        clip_repo, revision=clip_commit_id, dtype=jnp.float16, _do_init=False
    )
    clip_processor = CLIPProcessor.from_pretrained(clip_repo, revision=clip_commit_id)
    clip_params = replicate(clip_params)

    # Score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = clip(params=params, **inputs).logits_per_image
        return logits

    # Get scores
    clip_inputs = clip_processor(
        text=texts * jax.device_count(),
        images=images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), clip_params)

    # Organize scores
    prompt_num = len(texts)
    logits = np.asarray([logits[:, i::prompt_num, i] for i in range(prompt_num)]).squeeze()

    imgs = []
    for i in enumerate(texts):
        for idx in logits[i].argsort()[::-1]:
            imgs.append(images[idx * prompt_num + i])
            print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
        print()

    result = []
    result.append(imgs[0])

    return result
