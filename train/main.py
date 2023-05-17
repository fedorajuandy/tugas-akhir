#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-2022 The HuggingFace & DALL·E Mini team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'
# pylint: disable=too-many-lines
# pylint: disable=line-too-long
# pylint: f-string-without-interpolation
"""
Training DALL·E Mini.
Script adapted from run_summarization_flax.py

run_summarization_flax.py (CLI; passed through script's arguments) for running summarization models on using Flax library.
- input: JSON (list of input text)
- output: JSON (summaries)
Flax is a neural network library built on JAX
JAX is a Python library for high-performance numerical computing.

Used for transformer (BART)

CHANGE LATER = change when already passed the ridiculously long trials
DELETE LATER = if situation does not go to the abyss, delete; or just not used but can be fatal if deleted
CHECK LATER = need to be checked
EDITED = change content
IDK = perhaps still needed for exp, or delete it, or modif it
RESULT = calculation
RA = monolog and step 0 debugging (diff from logs)
REPORT = on WANDB report for reasons
MISSING = not seen/typo
DILUC = bookmark (don't judge me; I keep forgotting the last bookmark
KAEYA = I felt like I have seen it before, but memory gone awry
SHIRA = change later if something wen t awry
"""

import io
# for input/output operation; to work with stream (files, strings, other data; stored in memory (not file)
# DELETE LATER; for bucket (ha, as if I have that much money-)
import logging
# loggin messagese for debugging and monitoring
import os
# for files, directories, creating processes, getting info about the process and environment
import sys
# interact with interpreter (sys.arg; CL argv,) sys.path(), sys.exit()
import tempfile
# create temporary files and directories (deleted after run)
import time
# get time, sleep, measure elapsed time (this trigger something that I should... contemplate later)
from dataclasses import asdict
# create class representing data
# dataclass: class' decorator, generate methods, such as __init__(), __repr__(), and __eq__(), based on the class's fields; boilerplate to hold data
# asdict(): returns dictionary containing values of given instance of dataclass (dictionary => JSON)
# field(): specify metadata about the fields of a dataclass (default values, whether they are mutable, how they should be compared for equality); help match requirements
from functools import partial
# create a new function with some of the arguments of an existing function "pre-filled" (using same fuction with some same arg)
from pathlib import Path
# provides an object-oriented interface to working with file paths and file systems in a platform-independent way (windows, linux, wutevar does not matter)
from typing import Any, Callable, NamedTuple, Optional
# hints for Python code to provide additional information about the types of variables and functions (ex: def some_func(a: int, b: int) -> int)
# Any: any types
# Callable: another function (operation: Callable[[int, int], int])
# NamedTuple:light weight data class in tuples {see report for example}
# Optional: can be None
import datasets
# for NLP
import flax
# NN library on JAX
import jax
import jax.numpy as jnp
# Numberical computing library for GPU and TPU (like numpy; working with arrays said accelerators... does accelerator's vector actually functions just to manipulate the vector's speed? How does one's brain even calculate that on spot-)
import jaxlib # pylint: disable=import-error # type: ignore
# Low-level utilities (basic software tools for computing and numerical analysis; for efficiency and speed; building blocks for complex algorithms) for jax (rng, matrix stuff, algebra)
import numpy as np
# computing for CPU
import optax # pylint: disable=import-error # type: ignore
# gradient optimisation (find parameter val that minimise loss)
import transformers
# pre-trained for NLP, fine-tune, and dataset evaluation
import wandb
# ML tracking (logging, visualisation, sharing)
from datasets import Dataset
# for iterating dataset; accessing examples
from flax import traverse_util
# core: define and manipulate NN model
# struct: to define structured data types
# traverse_util: traverse and modify nested data structure
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# manipulate immutable dictionary (data structure; traditional dictionary/hash map that, well, immutable)
from flax.serialization import from_bytes, to_bytes
# serialise and deserialise Flax objects (bytes oc)
from flax.training.common_utils import onehot
# for NN; oneshot: return integers from encoding
from jax.experimental import PartitionSpec, maps
# specify & manipulate data sharding and parallelism
# PartitionSpec: specify sharding pattern for tensor
# maps: specify parallelism strategy
from jax.experimental.compilation_cache import compilation_cache as cc
# compilation_cache: JAX cache (yey to no cache) to reuse compiled functions [result: not used if < 1s]
from jax.experimental.pjit import pjit, with_sharding_constraint
# decorator (takes another function as input, returns modified one: like inheritance, but already made) for parallelism
from scalable_shampoo.distributed_shampoo import GraftingType, distributed_shampoo #pylint: disable=import-error
# distributed training
# GraftingType: enumeration (named constant values)
from tqdm import tqdm
# provides a progress bar (whoever made this, thank you)
from transformers import HfArgumentParser
# parse CL args for pre-trained models

import dalle_mini
# lightweight DALL-E (I want more GPU-)
from dalle_mini.data import Dataset # pylint: disable=import-error # pylint: disable=reimported
# load & preprocess image data
from dalle_mini.model import (
    DalleBart,
    # generate image from image and text (the decoder?)
    DalleBartConfig,
    # stores config option (layer, etc)
    DalleBartTokenizer,
    # tokenise text (encoder)
    set_partitions,
    # divide input image to be processed (the part of 'codebook', I supposed)
)
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from .train_state import TrainState

# DELETE LATER
# try:
#     from google.cloud import storage
#     # for GCS (might be used if I'm desperate enough to pull a trial... but where should I get a credit card-)
# except:
#     storage = None

logger = logging.getLogger(__name__)
# logger.debug('This is a debug message'); logger.info(), logger.warning(), or logger.error()

cc.initialize_cache("jax_cache")


# Goes four times in Kaggle, runs trainable_params, nothing happens, and goes one more time. Then error
def split_params(data):
    """Split params between scanned and non-scanned"""
    print(f"Ra's here. starting spliting params...")

    flat = traverse_util.flatten_dict(unfreeze(data))
    # flat dictionary: nested dictionary in path... repreesntation -> seek report
#     print(f"RA: flat = {flat}")
    # Kaggle: flat = {('lm_head', 'kernel'): ShapeDtypeStruct(shape=(1024, 16385), dtype=float32), ('model', 'decoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(256, 1024), dtype=float32), ('model', 'decoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(16385, 1024), dtype=float32), ('model', 'decoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_2', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(64, 1024), dtype=float32), ('model', 'encoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(50264, 1024), dtype=float32), ('model', 'encoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}
    # Kaggle: flat = {('lm_head', 'kernel'): ShapeDtypeStruct(shape=(1024, 16385), dtype=float32), ('model', 'decoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(256, 1024), dtype=float32), ('model', 'decoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(16385, 1024), dtype=float32), ('model', 'decoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_2', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(64, 1024), dtype=float32), ('model', 'encoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(50264, 1024), dtype=float32), ('model', 'encoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}

    split = {"standard": {}, "scanned_encoder": {}, "scanned_decoder": {}}
#     print(f"RA: split = {split}")
    # Kaggle: split = {'standard': {}, 'scanned_encoder': {}, 'scanned_decoder': {}}
    # Kaggle: split = {'standard': {}, 'scanned_encoder': {}, 'scanned_decoder': {}}
    for k, v in flat.items(): # pylint: disable=invalid-name
        if "FlaxBartEncoderLayers" in k:
            split["scanned_encoder"][k] = v
        elif "FlaxBartDecoderLayers" in k:
            split["scanned_decoder"][k] = v
        else:
            split["standard"][k] = v
#     print(f"RA: split after loop = {split}")
    # Kaggle: split after loop = {'standard': {('lm_head', 'kernel'): ShapeDtypeStruct(shape=(1024, 16385), dtype=float32), ('model', 'decoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(256, 1024), dtype=float32), ('model', 'decoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(16385, 1024), dtype=float32), ('model', 'decoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(64, 1024), dtype=float32), ('model', 'encoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(50264, 1024), dtype=float32), ('model', 'encoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32)}, 'scanned_encoder': {('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}, 'scanned_decoder': {('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_2', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}}
    # Kaggle: split after loop = {'standard': {('lm_head', 'kernel'): ShapeDtypeStruct(shape=(1024, 16385), dtype=float32), ('model', 'decoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(256, 1024), dtype=float32), ('model', 'decoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(16385, 1024), dtype=float32), ('model', 'decoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'decoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'embed_positions', 'embedding'): ShapeDtypeStruct(shape=(64, 1024), dtype=float32), ('model', 'encoder', 'embed_tokens', 'embedding'): ShapeDtypeStruct(shape=(50264, 1024), dtype=float32), ('model', 'encoder', 'final_ln', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'bias'): ShapeDtypeStruct(shape=(1024,), dtype=float32), ('model', 'encoder', 'layernorm_embedding', 'scale'): ShapeDtypeStruct(shape=(1024,), dtype=float32)}, 'scanned_encoder': {('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'encoder', 'layers', 'FlaxBartEncoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}, 'scanned_decoder': {('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_0', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'k_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'out_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'q_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'FlaxBartAttention_1', 'v_proj', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_0', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_1', 'kernel'): ShapeDtypeStruct(shape=(12, 1024, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'Dense_2', 'kernel'): ShapeDtypeStruct(shape=(12, 2730, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'GLU_0', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 2730), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_0', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_1', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_2', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'bias'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32), ('model', 'decoder', 'layers', 'FlaxBartDecoderLayers', 'LayerNorm_3', 'scale'): ShapeDtypeStruct(shape=(12, 1024), dtype=float32)}}

    # remove empty keys; EDITED: DELETE LATER (if using controlled dataset), but for checking... let's see later
    split = {k: v for k, v in split.items() if v}
    for k, v in split.items():
        split[k] = freeze(traverse_util.unflatten_dict(v))
#     print(f"RA: split after removing non-captions = {split}")
    # REPORT
    return split


def unsplit_params(data):
    """" Unsplitting parameterse """
    print(f"Ra's here. Starting unspliting params...")
    
    flat = {}
    for k in ["standard", "scanned_encoder", "scanned_decoder"]:
        if k in data:
            flat.update(traverse_util.flatten_dict(unfreeze(data[k])))
#     print(f"RA: flat = {flat}")
            
    return freeze(traverse_util.unflatten_dict(flat))


def trainable_params(data, embeddings_only):
    """Keep only trainable parameters"""
    print(f"Ra's here. Filtering trainable params...")

    if not embeddings_only:
        return data
#     print(f"RA: data = {data}")

    # DELETE LATER; by I mean delete, edit the whole functionality
    data = unfreeze(data)
#     print(f"RA: frozen data = {data}")

    trainable = {
        "lm_head": data["lm_head"],
        "model": {
            "decoder": {
                # CHECK LATER
                layer: data["model"]["decoder"][layer]
                for layer in [
                    "embed_positions",
                    "embed_tokens",
                    "final_ln",
                    "layernorm_embedding",
                ]
                # dictionary comprehension; transform one dict to another -> seek report
            }
        },
    }
#     print(f"RA: trainable = {trainable}")

    return freeze(trainable)

# CHECK LATER
def init_embeddings(model, params):
    """Reinitialize trainable embeddings"""
    print(f"Ra's here. initialising embedding...")

    # Must match params in trainable_params() above
    trainable_keypaths = [
        "lm_head.kernel",
        "model.decoder.embed_positions.embedding",
        "model.decoder.embed_tokens.embedding",
        "model.decoder.final_ln.bias",
        "model.decoder.layernorm_embedding.bias",
        "model.decoder.layernorm_embedding.scale",
    ]
#     print(f"RA: trainable_keypaths = {trainable_keypaths}")

    # Note: using private _missing_keys
    init_keys = {tuple(k.split(".")) for k in trainable_keypaths}
#     print(f"RA: init_keys = {init_keys}")
    model._missing_keys = init_keys

    return model.init_weights(model.key, model.input_shape, params=params)


def main():
    print(f"Ra's here. Starting...")
    print(f"Ra's here. Parsing arguments...")

    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    print(f"RA: parser = {parser}")
    # Kaggle: parser = HfArgumentParser(prog='train_edited.py', usage=None, description=None, formatter_class=<class 'argparse.ArgumentDefaultsHelpFormatter'>, conflict_handler='error', add_help=True)
    print(f"RA: ModelArguments = {ModelArguments}")
    print(f"RA: DataTrainingArguments = {DataTrainingArguments}")
    print(f"RA: TrainingArguments = {TrainingArguments}")

    # DELETE LATER (the json part)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # print(f"RA: parser.parse_args_into_dataclasses() = {parser.parse_args_into_dataclasses()}")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # print(f"RA: model_args = {model_args}")
        # Colab: model_args = ModelArguments(model_name_or_path=None, config_name='/kaggle/working/train/config/edited', tokenizer_name='boris/dalle-mini-tokenizer', dtype='float32', restore_state=False, dropout=None, activation_dropout=None, attention_dropout=None)
        # Kaggle: ModelArguments(model_name_or_path=None, config_name='/kaggle/working/tugas-akhir/train/config/edited', tokenizer_name='boris/dalle-mini-tokenizer', dtype='float32', restore_state=False, dropout=None, activation_dropout=None, attention_dropout=None)
        # print(f"RA: data_args = {data_args}")
        # RA: data_args = DataTrainingArguments(text_column='caption', encoding_column='encoding', dataset_repo_or_path='/kaggle/input/celeba-hq-encoded-512/encoded_data_512', train_file=None, validation_file=None, streaming=True, use_auth_token=False, shard_by_host=False, blank_caption_prob=0.0, clip_score_column='clip_score', min_clip_score=None, max_clip_score=None, filter_column=None, filter_value=None, multi_eval_ds=False, max_train_samples=None, max_eval_samples=None, preprocessing_num_workers=None, overwrite_cache=False, seed_dataset=None)
        # Kaggle: data_args = DataTrainingArguments(text_column='caption', encoding_column='encoding', dataset_repo_or_path='/kaggle/input/celeba-hq-encoded-512/encoded_data_512', train_file=None, validation_file=None, streaming=True, use_auth_token=False, shard_by_host=False, blank_caption_prob=0.0, clip_score_column='clip_score', min_clip_score=None, max_clip_score=None, filter_column=None, filter_value=None, multi_eval_ds=False, max_train_samples=None, max_eval_samples=None, preprocessing_num_workers=None, overwrite_cache=False, seed_dataset=None)
        # print(f"RA: training_args = {training_args}")
        # Kaggle: training_args = TrainingArguments(output_dir='/kaggle/working/tugas-akhir/train/who_are_you', overwrite_output_dir=False, do_train=True, do_eval=False, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1, gradient_checkpointing=False, learning_rate=5e-05, optim='distributed_shampoo', weight_decay=0.0, beta1=0.9, beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, block_size=1024, preconditioning_compute_steps=10, skip_preconditioning_dim_size_gt=4096, graft_type='rmsprop_normalized', nesterov=False, optim_quantized=False, shard_shampoo_across='dp', num_train_epochs=1, warmup_steps=0, lr_decay=None, lr_transition_steps=None, lr_decay_rate=None, lr_staircase=False, lr_offset=0, logging_steps=40, eval_steps=400, save_steps=4000, log_model=False, log_norm_steps=40, log_histogram_steps=False, seed_model=42, embeddings_only=False, init_embeddings=False, wandb_entity=None, wandb_project='dalle-mini', wandb_job_type='Seq2Seq', assert_TPU_available=False, use_vmap_trick=True, mp_devices=1, dp_devices=2

    # check arguments
    # CHECK LATER
    if training_args.mp_devices > jax.local_device_count():
        assert (
            data_args.seed_dataset is not None
        ), "Seed dataset must be provided when model is split over multiple hosts"

    print(f"Ra's here. Start logging...")
    # Make one log on every process with the configuration for debugging.
    # DELETE LATER; useful for now tho...
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        # name = the producing one
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    print(f"Ra's here. load dataset...")
    # Load dataset
    dataset = Dataset(
        **asdict(data_args),
        # unpack dict (the dataset)
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )

    # DELETE LATER
    logger.info(f"Local TPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs: {jax.device_count()}")

    # Set up wandb run
    # DELETE LATER (just set it up in notebook)
    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=parser.parse_args(),
        )

    print(f"Ra's here. Getting config...")
    # Set up our new model config
    # CHECK LATER
    config_args = {
        k: getattr(model_args, k)
        for k in ["dropout", "activation_dropout", "attention_dropout"]
        if getattr(model_args, k) is not None
    }
#     print(f"RA: config_args = {config_args}")
    # Kaggle: config_args = {}
    
    config_args["gradient_checkpointing"] = training_args.gradient_checkpointing
    
    # IDK
    if model_args.config_name:
        config = DalleBartConfig.from_pretrained(model_args.config_name)
    else:
        config = None
#     print(f"RA: config = {config}")
    # REPORT

    # Load or create new model
    # DELETE LATER; WAIT, DO NOT DELETE THE ELSE
    if model_args.model_name_or_path:
        model, params = DalleBart.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        
        if training_args.embeddings_only and training_args.init_embeddings:
            params = init_embeddings(model, params)
    else:
        model = DalleBart(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        params = None
#     print(f"RA: model = {model}")
    # Kaggle: model = <dalle_mini.model.modeling.DalleBart object at 0x7f5a19c48510>
#     print(f"RA: params = {params}")
    # Kaggle: params = None
        
    # CHECK LATER
    for k, v in config_args.items():
        setattr(model.config, k, v)
    params_shape = model.params_shape_tree
#     print(f"RA: params_shape = {params_shape}")
    # REPORT

    # get model metadata
    # CHECK LATER
    model_metadata = model_args.get_metadata()
#     print(f"RA: model_metadata = {model_metadata}")
    # Kaggle: model_metadata = {}
    
    print(f"Ra's here. Processing dataset...")
    # get PartitionSpec for model params (required to be a dict)
    # break to small chunks for parallel
    param_spec = set_partitions(params_shape, model.config.use_scan)
#     print(f"RA: param_spec = {param_spec}")
    # REPORT
    # set_partitions(tuple of model's paramemter, bool )
    params_shape = freeze(params_shape)
#     print(f"RA: params_shape = {params_shape}")
    # REPORT
    if params is not None:
        params = freeze(params)
#         print(f"RA: params = {params}")
        # Skip for Kaggle

    # Load tokenizer
    tokenizer = DalleBartTokenizer.from_pretrained(
        model_args.tokenizer_name, use_fast=True
    )
#     print(f"RA: tokenizer = {tokenizer}")
    # Kaggle: tokenizer = DalleBartTokenizer(name_or_path='boris/dalle-mini-tokenizer', vocab_size=50265, model_max_length=1024, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)})

    # Preprocessing the datasets.
    # We need to normalize and tokenize inputs and targets.
    # tokenize those words according to tokenizer and config (layer, etc)
    dataset.preprocess(tokenizer=tokenizer, config=model.config)

    print(f"Ra's here. Config details and all... stuff...")
    # Initialize our training
    dropout_rng = jax.random.PRNGKey(training_args.seed_model)
    # PRNGKey (data type) = 128 bit (2 x 64 for JAX)  seed val for PRNG (based on Threefry counter-based RNG; peseudorandom from counter val)
#     print(f"RA: dropout_rng = {dropout_rng}")
    # Kaggle: dropout_rng = [ 0 42]

    # Store some constant
    num_epochs = training_args.num_train_epochs
#     print(f"RA: num_epochs = {num_epochs}")
    # Kaggle: num_epochs = 1
    # batch size
    batch_size_per_node_per_grad_step = (
        training_args.per_device_train_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
#     print(f"RA: per_device_train_batch_size = {training_args.per_device_train_batch_size}")
    # Kaggle: per_device_train_batch_size = 1
#     print(f"RA: local_device_count = {jax.local_device_count()}")
    # Kaggle: local_device_count = 2
#     print(f"RA: mp_devices = {training_args.mp_devices}")
    # RESULT: 1 * 1 // 1 = 1
    # Kaggle: mp_devices = 1
    batch_size_per_node = (
        batch_size_per_node_per_grad_step * training_args.gradient_accumulation_steps
    )
#     print(f"RA: batch_size_per_node_per_grad_step = {batch_size_per_node_per_grad_step}")
    # Kaggle: batch_size_per_node_per_grad_step = 2
#     print(f"RA: gradient_accumulation_steps = {training_args.gradient_accumulation_steps}")
    # RESULT: 1 * 1 = 1
    # Kaggle: gradient_accumulation_steps = 1
    batch_size_per_step = batch_size_per_node * jax.process_count()
#     print(f"RA: batch_size_per_step = {batch_size_per_step}")
    # RESULT = 1 * 1 = 1
    # MISSING
    
    # DELETE LATER
    eval_batch_size_per_node = (
        training_args.per_device_eval_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
    )
#     print(f"RA: eval_batch_size_per_node = {eval_batch_size_per_node}")
    # Kaggle: eval_batch_size_per_node = 2
    eval_batch_size_per_step = eval_batch_size_per_node * jax.process_count()
#     print(f"RA: eval_batch_size_per_step = {eval_batch_size_per_step}")
    # Kaggle: eval_batch_size_per_step = 2
    
    # DELETE LATER the eval part
    len_train_dataset, len_eval_dataset = dataset.length
#     print(f"RA: len_train_dataset = {len_train_dataset}")
    # Kaggle: len_train_dataset = None
    steps_per_epoch = (
        len_train_dataset // batch_size_per_node
        if len_train_dataset is not None
        else None
    )
#     print(f"RA: batch_size_per_node {batch_size_per_node}")
    # The result is 2. Wut.
    # Kaggle: batch_size_per_node 2
#     print(f"RA: steps_per_epoch = {steps_per_epoch}")
    # RESULT = PENDING
    # Kaggle: steps_per_epoch = None
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )
#     print(f"RA: num_train_steps = {num_train_steps}")
    # Kaggle: num_train_steps = None
    num_params = model.num_params(params_shape)
#     print(f"RA: num_params = {num_params}")
    # Kaggle: num_params = 437833712

    print(f"Ra's here. Start training...")
    # DELETE LATER
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Batch size per dp device = {training_args.per_device_train_batch_size}"
    )
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(
        f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Batch size per update = {batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")
    
    # set up wandb run; DELETE LATER
    print(f"Ra's here. Initialising WANDB metrics...")
    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        wandb.define_metric("*", step_metric="train/step")

        # add interesting config parameters
        wandb.config.update(
            {
                "len_train_dataset": len_train_dataset,
                "len_eval_dataset": len_eval_dataset,
                "batch_size_per_step": batch_size_per_step,
                "num_params": num_params,
                "model_config": model.config.to_dict(),
                "num_devices": jax.device_count(),
                "versions": {
                    "jax": jax.__version__,
                    "jaxlib": jaxlib.__version__,
                    "flax": flax.__version__,
                    "transformers": transformers.__version__,
                    "datasets": datasets.__version__,
                    "wandb": wandb.__version__,
                    "dalle_mini": dalle_mini.__version__,
                },
            }
        )

    # Create learning rate schedule
    # EDITED: creates a learning rate for an optimizer (to be adjusted; read: experimented. Yey)
    # hyperparameter that controls the step size at each iteration of an optimization algorithm
    def create_learning_rate_fn() -> Callable[[int], jnp.array]:
        # jnp.array = NumPy array (multi-dimensional, homogeneous array of fixed-size items; same data type)
        """Create the learning rate function."""
        print(f"Ra's here. Creating learning rate...")
        
        # gradually increases learning rate from init_value to training_args.learning_rate over training_args.warmup_steps; for helping model converge faster during the initial phase of training
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=training_args.learning_rate,
            transition_steps=training_args.warmup_steps + 1,  # ensure not 0
        )
#         print(f"RA: end_value = {training_args.learning_rate}")
        # Kaggle: end_value = 5e-05
#         print(f"RA: transition_steps = {training_args.warmup_steps + 1}")
        # Kaggle: transition_steps = 1
#         print(f"RA: warmup_fn = {warmup_fn}")
        # Kaggle: warmup_fn = <function polynomial_schedule.<locals>.schedule at 0x754647245320>
        
        last_boundary = training_args.warmup_steps
#         print(f"RA: last_boundary = {last_boundary}")
        # Kaggle: last_boundary = 0
        
        # offset step when resuming
        # DELETE LATER... maaaaaybe. Let's see the GPU limit first. Or luck
#         print(f"RA: lr_offset = {training_args.lr_offset}")
        # Kaggle: lr_offset = 0
        if training_args.lr_offset:
            # appends a constant schedule of 0 before the warm-up schedule, and updates the last boundary (for resuming training from a checkpoint)
            warmup_fn = optax.join_schedules(
                schedules=[optax.constant_schedule(0.0), warmup_fn],
                boundaries=[training_args.lr_offset],
            )
#             print(f"RA: schedules = {schedules}")
#             print(f"RA: boundaries = {boundaries}")
#             print(f"RA: warmup_fn = {warmup_fn}")
            last_boundary += training_args.lr_offset
#             print(f"RA: last_boundary = {last_boundary}")
            
#         print(f"RA: lr_decay = {training_args.lr_decay}")
        # Kaggle: lr_decay = None
        if training_args.lr_decay is None:
#             print(f"RA: warmup_fn = {warmup_fn}")
            # Kaggle: warmup_fn = <function polynomial_schedule.<locals>.schedule at 0x754647245320>
            return warmup_fn
        elif training_args.lr_decay == "linear":
            assert (
                num_train_steps is not None
            ), "linear decay requires knowing the dataset length"
            decay_fn = optax.linear_schedule(
                init_value=training_args.learning_rate,
                end_value=0,
                transition_steps=num_train_steps - training_args.warmup_steps,
            )
#             print(f"RA: learning_rate = {training_args.learning_rate}")
#             print(f"RA: transition_steps = {transition_steps}")
#             print(f"RA: decay_fn = {decay_fn}")
        elif training_args.lr_decay == "exponential":
            # useful when the model is close to converging and requires smaller updates
            decay_fn = optax.exponential_decay(
                init_value=training_args.learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
#             print(f"RA: learning_rate = {training_args.learning_rate}")
#             print(f"RA: transition_steps = {transition_steps}")
#             print(f"RA: decay_rate = {decay_rate}")
#             print(f"RA: staircase = {staircase}")
#             print(f"RA: decay_fn = {decay_fn}")
            
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[last_boundary],
        )
#         print(f"RA: schedules = {schedules}")
#         print(f"RA: boundaries = {boundaries}")
#         print(f"RA: schedule_fn = {schedule_fn}")
        
        return schedule_fn

    learning_rate_fn = create_learning_rate_fn()
#     print(f"RA: learning_rate_fn = {learning_rate_fn}")
    # Kaggle: learning_rate_fn = <function polynomial_schedule.<locals>.schedule at 0x754647245320>
    
    # create optimizer
    print(f"Ra's here. Cereate optimizer...")
    trainable_params_shape = trainable_params(
        params_shape, training_args.embeddings_only
    )
#     print(f"RA: params_shape = {params_shape}")
#     print(f"RA: embeddings_only = {training_args.embeddings_only}")
#     print(f"RA: trainable_params_shape = {trainable_params_shape}")

    # REMOVE LATER; most likely only us
    if training_args.optim == "distributed_shampoo":
        # parameters from https://github.com/tensorflow/lingvo/blob/03ee9d7cd50764b0424c7c863733c91fc0b053ec/lingvo/jax/optimizers.py#L729
        print(f"Ra's here. Using distributed shampoo...")

        # CHECK LATER
        # depends on the chosen one, that is the value (and that is what I get)
        graft_type = {
            # Stochastic Gradient Descent
            "sgd": GraftingType.SGD,
            # Adaptive Gradient Algorithm
            "adagrad": GraftingType.ADAGRAD,
            # Root Mean Square Propagation
            "rmsprop": GraftingType.RMSPROP,
            # Root Mean Square Propagation with Normalization
            "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
            # Square-root N scaling
            "sqrt_n": GraftingType.SQRT_N,
            # Adaptive Gradient Algorithm with Normalization
            "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
            # Adaptive Gradient Algorithm with Normalization
        }[training_args.graft_type]
#         print(f"RA: training_args.graft_type = {training_args.graft_type}")
#         print(f"RA: graft_type = {graft_type}")
        
        # partitioning the training across devices (one is one ;v;)
        statistics_partition_spec = (
            PartitionSpec(None, training_args.shard_shampoo_across, None)
            # class to partition tensor (mesh, mp, dp)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(None, "dp", "mp")
        )
#         print(f"RA: shard_shampoo_across = {training_args.shard_shampoo_across}")
#         print(f"RA: statistics_partition_spec = {statistics_partition_spec}")
        
        # CHECK LATER
        opt = distributed_shampoo(
            learning_rate_fn,
            # KAEYA
            block_size=training_args.block_size,
            # for block-diagonal preconditioner
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-6,
            weight_decay=training_args.weight_decay,
            start_preconditioning_step=max(
                training_args.preconditioning_compute_steps + 1, 101
            ),
            # when and how often the preconditioner and statistics are computed during training (and the two following lines too)
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            # allow the optimizer to handle inputs with unknown shapes
            graft_type=graft_type,
            nesterov=training_args.nesterov,
            # Nesterov momentum
            exponent_override=0,
            # usually 0.5
            statistics_partition_spec=statistics_partition_spec,
            preconditioner_partition_spec=PartitionSpec(
                training_args.shard_shampoo_across, None, None
            )
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(
                "mp" if training_args.mp_devices > training_args.dp_devices else "dp",
                None,
                None,
            ),
            num_devices_for_pjit=training_args.dp_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            # if the inverse operation fails to converge within this threshold, the algorithm will fall back to a less accurate but more stable approximation
            moving_average_for_momentum=True,
            # help smooth out fluctuations in the momentum value, which can improve convergence
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            # on dimensions that are larger than a certain size. Skipping preconditioning can reduce memory usage
            clip_by_scaled_gradient_norm=None,
            # whether to clip the gradient by its scaled norm. Gradient clipping prevent the gradient from exploding or vanishing
            precision=jax.lax.Precision.HIGHEST,
            # SHIRA
            # improve accuracy with more memory and computation time
            best_effort_memory_usage_reduction=training_args.optim_quantized,
            # reducing memory usage during optimization (bool)
        )
#         print(f"RA: preconditioning_compute_steps = {training_args.preconditioning_compute_steps}")
#         print(f"RA: opt = {opt}")
        
        # get the real optimizer and helper functions
        update_fn = opt.update
#         print(f"RA: update_fn = {update_fn}")

        optimizer = {}
        opt_fn = {}

        for k, p in split_params(trainable_params_shape).items():
            if "scanned" in k:
                p = jax.eval_shape(
                    # CHECK LATER
                    lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p
                    # extracts the first element of each value in the parameter tree
                )
#                 print(f"RA: update_fn = {update_fn}")
                
            optimizer[k] = opt.init(p)
            # initialize the optimizer with the trainable parameter values
#             print(f"RA: optimizer[k] = {optimizer[k]}")
            # CHECK LATER
            opt_fn[k] = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
                optimizer[k].pspec_fn, optimizer[k].shape_and_dtype_fn
            )
#             print(f"RA: opt_fn[k] = {opt_fn[k]}")
            # CHECK LATER
            optimizer[k] = optax.GradientTransformation(optimizer[k].init_fn, update_fn)
#             print(f"RA: optimizer[k] = {optimizer[k]}")
            
    elif training_args.optim == "adam":
        print(f"Ra's here. Using adam...")
        # adam optimizier includes weight decay regularization
        optimizer = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
#         print(f"RA: optimizer = {optimizer}")
        optimizer = {k: optimizer for k in split_params(trainable_params_shape)}
        # creates a dictionary where the keys are the names of the trainable parameters and the values are the optimizer object created
        # splits the trainable parameters into separate groups based on their shapes (for applying different optimization strategies to different groups of parameters, such as applying weight decay only to the weights and not the biases)
#         print(f"RA: optimizer = {optimizer}")

    elif training_args.optim == "adafactor":
        print(f"Ra's here. Using adafactor...")
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=learning_rate_fn,
            clipping_threshold=training_args.max_grad_norm,
            # prevent gradient becoming too large (esp with many parameters)
            weight_decay_rate=training_args.weight_decay,
        )
#         print(f"RA: optimizer = {optimizer}")
        optimizer = {k: optimizer for k in split_params(trainable_params_shape)}
#         print(f"RA: optimizer = {optimizer}")

    # get PartitionSpec for optimizer state
    def get_opt_state_spec_and_shape():
        print(f"Ra's here. Getting PartitionSpec...")
        # get opt_state shape without actual init
        opt_state_shape = {}
        
        # DILUC for explanation
        print(f"Ra's here. Start looping...")
        for k, p in split_params(trainable_params_shape).items():
            if "scanned" not in k:
                opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)
#                 print(f"RA: opt_state_shape[k] = {opt_state_shape[k]}")
            else:
                opt_state_shape[k] = jax.eval_shape(jax.vmap(optimizer[k].init), p)
        print(f"Ra's here. Ends loop...")
        
        # most likely remove if
        if training_args.optim == "adafactor":
            print(f"Ra's here. Using adafactor...")
            # factorized state must be replicated (rank different than params)
            opt_state_spec = {k: None for k in split_params(trainable_params_shape)}
#             print(f"RA: opt_state_spec = {opt_state_spec}")

        elif training_args.optim in ["adam", "distributed_shampoo"]:
            print(f"Ra's here. Using distributed shampoo or adam...")
            def _opt_state_spec_per_leaf(x, spec):
                print(f"Ra's here. _opt_state_spec_per_leaf starts...")
#                 print(f"RA: x = {x}")
#                 print(f"RA: spec = {spec}")
                if isinstance(x, FrozenDict):
                    # variables with same structure as params
                    return spec
                else:
                    # other variables such as count
                    return None

            print(f"Ra's here. _opt_state_spec_per_leaf ends.")
            split_spec = split_params(set_partitions(trainable_params_shape, False))
#             print(f"RA: split_spec = {split_spec}")
            opt_state_spec = {}
            
#             print(f"RA: trainable_params_shape = {trainable_params_shape}")
            print(f"Ra's here. Start looping...")
            for k, p in split_params(trainable_params_shape).items():
                if "scanned" in k:
                    p = jax.eval_shape(
                        lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p
                    )
#                     print(f"RA: p = {p}")
                    
                if training_args.optim == "adam":
                    print(f"Ra's here. Using adam...")
                    opt_state_spec[k] = jax.tree_util.tree_map(
                        partial(_opt_state_spec_per_leaf, spec=split_spec[k]),
                        opt_state_shape[k],
                        # return None spec for empty elements
                        is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
                    )
#                     print(f"RA: opt_state_spec[k] = {opt_state_spec[k]}")
#                     print(f"RA: _opt_state_spec_per_leaf = {_opt_state_spec_per_leaf}")
#                     print(f"RA: opt_state_shape[k] = {opt_state_shape[k]}")
#                     print(f"RA: is_leaf = {is_leaf}")
                elif training_args.optim == "distributed_shampoo":
                    print(f"Ra's here. Using distributed shampoo...")
                    opt_state_spec[k] = opt_fn[k].pspec_fn(
                        p,
                        split_spec[k],
                        statistics_partition_spec,
                    )
#                     print(f"RA: opt_state_shape[k] = {opt_state_shape[k]}")
                    
                # add dimension for scanned params
                if "scanned" in k:
                    opt_state_spec[k] = jax.tree_util.tree_map(
                        lambda x: PartitionSpec(*(None,) + x)
                        if x is not None
                        else None,
                        opt_state_spec[k],
                        is_leaf=lambda x: isinstance(x, PartitionSpec),
                    )
#                     print(f"RA: opt_state_shape[k] = {opt_state_shape[k]}")
#                     print(f"RA: is_leaf = {is_leaf}")
                    
            print(f"Ra's here. Loop ended.")
        else:
            raise NotImplementedError
            
#         print(f"RA: opt_state_spec = {opt_state_spec}")
#         print(f"RA: opt_state_shape = {opt_state_shape}")
        return freeze(opt_state_spec), freeze(opt_state_shape)

    print(f"Ra's here. Mesh stuff here..?")
    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape()
#     print(f"RA: opt_state_spec = {opt_state_spec}")
#     print(f"RA: opt_state_shape = {opt_state_shape}")

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
#     print(f"RA: mesh_shape (dp, mp) = {mesh_shape}")
    # Kaggle: mesh_shape (dp, mp) = (2, 1)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
#     print(f"RA: devices = {devices}")
    # devices = [[StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)]
    # [StreamExecutorGpuDevice(id=1, process_index=0, slice_index=0)]]
    mesh = maps.Mesh(devices, ("dp", "mp"))
    # reshape it into a grid of the specified shape to distribute computation
#     print(f"RA: mesh = {mesh}")
    logger.info(f"  Mesh shape: {mesh_shape}")


    # define state spec
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
        apply_fn=model.__call__,
        tx=optimizer,
    )

    # init params if not available yet
    def maybe_init_params(params):
#         print(f"RA: params = {params}")
        if params is not None:
            # model params are correctly loaded
            return params
        else:
            # params have not been initialized yet
            return model.init_weights(model.key, model.input_shape)

    with mesh:
        logger.info("  Creating state")

        # restore metadata
        attr_state = {}
        keys = ["train_time", "train_samples"]
        if model_args.restore_state:
            keys += ["step", "epoch"]
        attr_state = {k: v for k, v in model_metadata.items() if k in keys}

        if not model_args.restore_state:

            def init_state(params):
                return TrainState.create(
                    apply_fn=model.__call__,
                    tx=optimizer,
                    params=maybe_init_params(params),
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                init_state,
                in_axis_resources=(param_spec,)
                if model_args.model_name_or_path
                else None,
                out_axis_resources=state_spec,
                donate_argnums=(0,),
            )(params)

        else:
            # load opt_state
            opt_state = from_bytes(opt_state_shape, model_args.get_opt_state())

            def restore_state(params, opt_state):
                return TrainState(
                    apply_fn=model.__call__,
                    tx=optimizer,
                    params=params,
                    opt_state=opt_state,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            print(f"RA: WHY")
            state = pjit(
                restore_state,
                in_axis_resources=(
                    param_spec,
                    opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(params, opt_state)

            # remove opt_state from CPU
            del opt_state

    # free CPU memory
    del params, opt_state_spec, opt_state_shape

    # define batch specs
    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    # define loss
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # "vmap trick" avoids a crash when mp_devices > 1 (not sure why it happens)
    # lead to better perf: see https://wandb.ai/dalle-mini/dalle-mini/reports/JAX-pmap-vs-pjit--VmlldzoxNDg1ODA2
    use_vmap_trick = training_args.use_vmap_trick

    # make grad_param_spec for vmap
    if use_vmap_trick:
        grad_param_spec = jax.tree_util.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            param_spec,
        )

    # Define gradient update step fn
    def train_step(state, batch, train_time):
        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        def compute_loss(params, minibatch, dropout_rng):
            # minibatch has dim (batch_size, ...)
            minibatch, labels = minibatch.pop("labels")
#             print(f"RA: minibatch = {minibatch}")
            # Kaggle: minibatch = FrozenDict({
            # attention_mask: Traced<ShapedArray(int32[1,64])>with<BatchTrace(level=2/0)> with
            #   val = Traced<ShapedArray(int32[2,1,64])>with<DynamicJaxprTrace(level=1/0)>
            #   batch_dim = 0,
            # decoder_input_ids: Traced<ShapedArray(float32[1,1024])>with<BatchTrace(level=2/0)> with
            #   val = Traced<ShapedArray(float32[2,1,1024])>with<DynamicJaxprTrace(level=1/0)>
            #   batch_dim = 0,
            # input_ids: Traced<ShapedArray(int32[1,64])>with<BatchTrace(level=2/0)> with
            #   val = Traced<ShapedArray(int32[2,1,64])>with<DynamicJaxprTrace(level=1/0)>
            #   batch_dim = 0
            # })
#             print(f"RA: labels = {labels}")
            # Kaggle: labels = Traced<ShapedArray(int32[1,1024])>with<BatchTrace(level=2/0)> with
            # val = Traced<ShapedArray(int32[2,1,1024])>with<DynamicJaxprTrace(level=1/0)>
            # batch_dim = 0
            logits = state.apply_fn( # KAGGLE
                **minibatch, params=params, dropout_rng=dropout_rng, train=True # KAGGLE
            )[0]
            print(f"Kaeya, the bestest baby brother, is here~")
#             print(f"RA: minibatch = {minibatch}")
#             print(f"RA: params = {params}")
#             print(f"RA: dropout_rng = {dropout_rng}")
#             print(f"RA: logits = {logits}")
            return loss_fn(logits, labels)

        grad_fn = jax.value_and_grad(compute_loss)
        
        def loss_and_grad(grad_idx, dropout_rng):
            # minibatch at grad_idx for gradient accumulation (None otherwise)
            minibatch = (
                get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            )
            # ensure it is sharded properly
            minibatch = with_sharding_constraint(minibatch, batch_spec)
            # only 1 single rng per grad step, let us handle larger batch size (not sure why)
            dropout_rng, _ = jax.random.split(dropout_rng)

            if use_vmap_trick:
                # "vmap trick", calculate loss and grads independently per dp_device
                loss, grads = jax.vmap( # KAGGLE
                    grad_fn, in_axes=(None, 0, None), out_axes=(0, 0)
                )(state.params, minibatch, dropout_rng) # KAGGLE
#                 print(f"RA: loss = {loss}")
#                 print(f"RA: grads = {grads}")
#                 print(f"RA: grad_fn = {grad_fn}")
                # ensure they are sharded correctly
                loss = with_sharding_constraint(loss, batch_spec)
                grads = with_sharding_constraint(grads, grad_param_spec)
                # average across all devices
                # Note: we could average per device only after gradient accumulation, right before params update
                loss, grads = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0), (loss, grads)
                )
            else:
                # "vmap trick" does not work in multi-hosts and requires too much hbm
                loss, grads = grad_fn(state.params, minibatch, dropout_rng)
            # ensure grads are sharded
            grads = with_sharding_constraint(grads, param_spec)
            # return loss and grads
            return loss, grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, dropout_rng = loss_and_grad(None, state.dropout_rng) # KAGGLE
#             print(f"RA: loss = {loss}")
#             print(f"RA: grads = {grads}")
#             print(f"RA: dropout_rng = {dropout_rng}")
#             print(f"RA: state.dropout_rng = {state.dropout_rng}")
        else:
            # create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                0.0,
                with_sharding_constraint(
                    jax.tree_util.tree_map(jnp.zeros_like, state.params), param_spec
                ),
                state.dropout_rng,
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                cumul_loss, cumul_grads, dropout_rng = cumul_loss_grad_dropout
                loss, grads, dropout_rng = loss_and_grad(grad_idx, dropout_rng) # KAGGLE
#                 print(f"RA: loss = {loss}")
#                 print(f"RA: grads = {grads}")
#                 print(f"RA: dropout_rng = {dropout_rng}")
#                 print(f"RA: grad_idx = {grad_idx}")
                cumul_loss, cumul_grads = jax.tree_util.tree_map(
                    jnp.add, (cumul_loss, cumul_grads), (loss, grads)
                )
                cumul_grads = with_sharding_constraint(cumul_grads, param_spec)
                return cumul_loss, cumul_grads, dropout_rng

            # loop over gradients
            loss, grads, dropout_rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step, # KAGGLE
            )
#             print(f"RA: loss = {loss}")
#             print(f"RA: grads = {grads}")
#             print(f"RA: dropout_rng = {dropout_rng}")
#             print(f"RA: cumul_minibatch_step = {cumul_minibatch_step}")
#             print(f"RA: init_minibatch_step = {init_minibatch_step}")
            grads = with_sharding_constraint(grads, param_spec)
            # sum -> mean
            loss, grads = jax.tree_util.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps, (loss, grads)
            )

        grads = with_sharding_constraint(grads, param_spec)

        # update state
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=dropout_rng,
            train_time=train_time,
            train_samples=state.train_samples + batch_size_per_step,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }

        def maybe_fn(fn, val, zeros, freq):
            """Call fn only if it is a logging step"""
            return jax.lax.cond(
                state.step % freq == 0,
                fn,
                lambda _: zeros,
                val,
            )

        # log additional metrics
        params = trainable_params(state.params, training_args.embeddings_only)
        grads = trainable_params(grads, training_args.embeddings_only)
        if training_args.log_norm_steps:
            zeros_norm = jax.tree_util.tree_map(lambda _: jnp.float32(0), params)

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val)

            gradients_norm = maybe_fn(
                norm, grads, zeros_norm, training_args.log_norm_steps
            )
            params_norm = maybe_fn(
                norm, params, zeros_norm, training_args.log_norm_steps
            )

            metrics.update(
                {
                    "gradients_norm": gradients_norm,
                    "params_norm": params_norm,
                }
            )

        if training_args.log_histogram_steps:
            zeros_hist = jax.tree_util.tree_map(
                lambda _: jnp.histogram(jnp.zeros(1), density=True), params
            )

            def histogram(val):
                return jax.tree_util.tree_map(
                    lambda x: jnp.histogram(x, density=True), val
                )

            gradients_hist = maybe_fn(
                histogram, grads, zeros_hist, training_args.log_histogram_steps
            )
            params_hist = maybe_fn(
                histogram, params, zeros_hist, training_args.log_histogram_steps
            )

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return state, metrics

    # Define eval fn
    eval_model = (
        model
        if model_args.dtype == "float32"
        else DalleBart(
            model.config,
            seed=training_args.seed_model,
            dtype=jnp.float32,
            _do_init=False,
        )
    )

    def eval_step(state, batch):
        def compute_eval_loss(batch):
            batch, labels = batch.pop("labels")
            logits = eval_model(**batch, params=state.params, train=False)[0]
            return loss_fn(logits, labels)

        if use_vmap_trick:
            loss = jax.vmap(compute_eval_loss)(batch)
            # ensure they are sharded correctly
            loss = with_sharding_constraint(loss, batch_spec)
            # average across all devices
            loss = jnp.mean(loss)
        else:
            loss = compute_eval_loss(batch)

        return loss

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(
            state_spec,
            grad_batch_spec
            if training_args.gradient_accumulation_steps > 1
            else batch_spec,
            None,
        ),
        out_axis_resources=(state_spec, None),
        donate_argnums=(0,),
    )
    print(f"RA p_train_step = {p_train_step}")
    # Kaggle: p_train_step = <jaxlib.xla_extension.PjitFunction object at 0x7038eeacc1d0>
    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(state_spec, batch_spec),
        out_axis_resources=None,
    )


    # keep local copy of state
    local_state = {
        k: jax.device_get(getattr(state, k)).item()
        for k in ["step", "epoch", "train_time", "train_samples"]
    }
    # init variables
    start_time = time.perf_counter() - local_state["train_time"]
    train_metrics = None
    evaluation_ran = False
    save_model_ran = False
    metrics_logger = MetricsLogger(local_state["step"])
    epochs = tqdm(
        range(local_state["epoch"], num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )
    
    print(f"Sadness upon your soul ;-;")
    def run_evaluation():
        # ======================== Evaluating ==============================
        if training_args.do_eval:
            start_eval_time = time.perf_counter()
            # get validation datasets
            val_datasets = list(
                dataset.other_eval_datasets.keys()
                if hasattr(dataset, "other_eval_datasets")
                else []
            )
            val_datasets += ["eval"]
            for val_dataset in val_datasets:
                eval_loader = dataset.dataloader(
                    val_dataset,
                    eval_batch_size_per_step
                    * max(1, training_args.mp_devices // jax.local_device_count()),
                )
                eval_steps = (
                    len_eval_dataset // eval_batch_size_per_step
                    if len_eval_dataset is not None
                    else None
                )
                eval_loss = []
                for batch in tqdm(
                    eval_loader,
                    desc="Evaluating...",
                    position=2,
                    leave=False,
                    total=eval_steps,
                    disable=jax.process_index() > 0,
                ):
                    # need to keep only eval_batch_size_per_node items relevant to the node
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape(
                            (jax.process_count(), eval_batch_size_per_node)
                            + x.shape[1:]
                        ),
                        batch,
                    )
                    batch = jax.tree_util.tree_map(
                        lambda x: x[jax.process_index()], batch
                    )

                    # add dp dimension when using "vmap trick"
                    if use_vmap_trick:
                        bs_shape = (
                            jax.local_device_count() // training_args.mp_devices,
                            training_args.per_device_eval_batch_size,
                        )
                        batch = jax.tree_util.tree_map(
                            lambda x: x.reshape(bs_shape + x.shape[1:]), batch
                        )

                    # freeze batch to pass safely to jax transforms
                    batch = freeze(batch)
                    # accumulate losses async
                    eval_loss.append(p_eval_step(state, batch))

                # get the mean of the loss
                eval_loss = jnp.stack(eval_loss)
                eval_loss = jnp.mean(eval_loss)
                eval_metrics = {"loss": eval_loss}

                # log metrics
                metrics_logger.log(eval_metrics, prefix=val_dataset)

                # Print metrics and update progress bar
                desc = f"Epoch... ({epoch + 1}/{num_epochs} | {val_dataset} Loss: {eval_metrics['loss']})"
                epochs.write(desc)
                epochs.desc = desc

            # log time
            metrics_logger.log_time("eval", time.perf_counter() - start_eval_time)

            return eval_metrics

    print(f"Yaya")
    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:

            start_save_time = time.perf_counter()
            output_dir = training_args.output_dir
            use_bucket = output_dir.startswith("gs://")
            if use_bucket:
                bucket_path = Path(output_dir[5:]) / wandb.run.id / f"step_{state.step}"
                bucket, dir_path = str(bucket_path).split("/", 1)
                tmp_dir = tempfile.TemporaryDirectory()
                output_dir = tmp_dir.name

            # save model
            params = jax.device_get(state.params)
            model.save_pretrained(
                output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # copy to bucket
            if use_bucket:
                client = storage.Client()
                bucket = client.bucket(bucket)
                for filename in Path(output_dir).glob("*"):
                    blob_name = str(Path(dir_path) / "model" / filename.name)
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(filename))
                tmp_dir.cleanup()

            # save state
            opt_state = jax.device_get(state.opt_state)
            if use_bucket:
                blob_name = str(Path(dir_path) / "state" / "opt_state.msgpack")
                blob = bucket.blob(blob_name)
                blob.upload_from_file(io.BytesIO(to_bytes(opt_state)))
            else:
                with (Path(output_dir) / "opt_state.msgpack").open("wb") as f:
                    f.write(to_bytes(opt_state))

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("20GB"))

                metadata = {
                    k: jax.device_get(getattr(state, k)).item()
                    for k in ["step", "epoch", "train_time", "train_samples"]
                }
                metadata["num_params"] = num_params
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics

                # create model artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/model"
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="DalleBart_model",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact.add_reference(metadata["bucket_path"])
                else:
                    for filename in [
                        "config.json",
                        "flax_model.msgpack",
                        "merges.txt",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    ]:
                        artifact.add_file(
                            f"{Path(training_args.output_dir) / filename}"
                        )
                wandb.run.log_artifact(artifact)

                # create state artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/state"
                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="DalleBart_state",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact_state.add_reference(metadata["bucket_path"])
                else:
                    artifact_state.add_file(
                        f"{Path(training_args.output_dir) / 'opt_state.msgpack'}"
                    )
                wandb.run.log_artifact(artifact_state)
            metrics_logger.log_time("save_model", time.perf_counter() - start_save_time)
    
    logger.info("  Ready to start training")
    with mesh:
        for epoch in epochs:
            print(f"Ra's here. starting epoch...")
            state = state.replace(epoch=epoch)
            local_state["epoch"] = epoch
            # ======================== Training ================================
            metrics_logger.update_state_metrics(local_state)
            metrics_logger.log({})

            if training_args.do_train:
                print(f"Ra's here. Starting the actual thing...")
                # load data - may be replicated on multiple nodes
                node_groups = max(
                    1, training_args.mp_devices // jax.local_device_count()
                )
                loader_bs = batch_size_per_node * node_groups
                train_loader = dataset.dataloader(
                    "train",
                    loader_bs,
                    epoch,
                )
                # train
                print(f"Ra's here. Batching...")
                for batch in tqdm(
                    train_loader,
                    desc="Training...",
                    position=1,
                    leave=False,
                    total=steps_per_epoch,
                    disable=jax.process_index() > 0,
                ):
                    # calculate delta time (we have a lag of one step but it's ok)
                    train_time = time.perf_counter() - start_time

                    # reset control variables
                    evaluation_ran = False
                    save_model_ran = False

                    # set correct shape to batch
                    # - add grad_step dim if gradient_accumulation_steps > 1
                    print(f"Ra's here. Shaping bs...")
                    bs_shape = (
                        (batch_size_per_node_per_grad_step * node_groups,)
                        if not use_vmap_trick
                        else (
                            jax.local_device_count()
                            * node_groups
                            // training_args.mp_devices,  # local dp devices
                            training_args.per_device_train_batch_size,
                        )
                    )
                    print(f"RA: bs_shape = {bs_shape}")
                    if training_args.gradient_accumulation_steps > 1:
                        # reshape data into (gradient_accumulation_steps, batch_per_node, ...)
                        # to avoid any data redistribution when sharding
                        bs_shape = (
                            training_args.gradient_accumulation_steps,
                        ) + bs_shape

                    print(f"Sora")
                    # reshape batch
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape(bs_shape + x.shape[1:]),
                        batch,
                    )
                    print(f"Sara")
                    # freeze batch to pass safely to jax transforms
                    batch = freeze(batch)
                    print(f"Nobody")

                    # train step
                    print(f"HERE GOES NOTHING.")
                    print(f"RA: state = {state}")
                    print(f"RA: train_metrics = {train_metrics}")
                    print(f"RA: state = {state}")
                    print(f"RA: batch = {batch}")
                    print(f"RA: train_time = {train_time}")
                    state, train_metrics = p_train_step(state, batch, train_time) # KAGGLE
                    local_state["step"] += 1
                    local_state["train_time"] = train_time
                    local_state["train_samples"] += batch_size_per_step

                    if (
                        local_state["step"] % training_args.logging_steps == 0
                        and jax.process_index() == 0
                    ):
                        metrics_logger.update_state_metrics(local_state)
                        metrics_logger.log(train_metrics, prefix="train")

                    eval_metrics = None
                    if local_state["step"] % training_args.eval_steps == 0:
                        eval_metrics = run_evaluation()
                        evaluation_ran = True

                    if local_state["step"] % training_args.save_steps == 0:
                        run_save_model(state, eval_metrics)
                        save_model_ran = True

                # log final train metrics
                if train_metrics is not None:
                    metrics_logger.update_state_metrics(local_state)
                    metrics_logger.log(train_metrics, prefix="train")

                    epochs.write(
                        f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metrics['loss']}, Learning Rate: {train_metrics['learning_rate']})"
                    )

            # Final evaluation at the end of each epoch
            if not evaluation_ran:
                eval_metrics = run_evaluation()

            # save checkpoint after each epoch
            if not save_model_ran:
                run_save_model(state, eval_metrics)


# import tensorflow as tf


if __name__ == "__main__":
    # # Detect and initialize the TPU
    # tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    # tf.config.experimental_connect_to_cluster(tpu)
    # tf.tpu.experimental.initialize_tpu_system(tpu)

    # # Create a distributed strategy for the TPU
    # strategy = tf.distribute.TPUStrategy(tpu)

    # # Use the strategy to run your code
    # with strategy.scope():
    #     # Your model and training code here

    # OTHER METHOD

    # # detect and init the TPU
    # tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # # instantiate a distribution strategy
    # tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # # instantiating the model in the strategy scope creates the model on the TPU
    # with tpu_strategy.scope():
    #     model = tf.keras.Sequential( … ) # define your model normally
    #     model.compile( … )

    #     # train model normally
    #     model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)

    main()
