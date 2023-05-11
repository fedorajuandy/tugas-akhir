# !/usr/bin/env python
# coding=utf-8
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
# limitations under the License.
"""
Image generation training script from scratch adapted from DALL-E mini's training script.

RA: REMEMBER CHECK LATER.
"""

import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional
import datasets
import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import transformers
from datasets import Dataset
from flax import core, struct, traverse_util
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.training.common_utils import onehot
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from scalable_shampoo.distributed_shampoo import GraftingType, distributed_shampoo
from tqdm import tqdm
from transformers import HfArgumentParser

import dalle_mini
from dalle_mini.data import Dataset
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
)

@dataclass
class ModelArguments:
    """
    Arguments of model/config/tokenizer is going to be fine-tuned or trained from scratch.
    """

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name_or_path"
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format of the computations [float32, float16, bfloat16]"
        },
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            assert (
                self.tokenizer_name is not None
            ), "Tokenizer name or model name/path needs to be specified"

    def get_metadata(self):
        """
        Indicating training from zero.
        """
        return dict()


@dataclass
class DataTrainingArguments:
    """
    Datas are going to be inputed training.
    """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "The name of the column in the datasets containing the full texts."
        },
    )
    encoding_column: Optional[str] = field(
        default="encoding",
        metadata={
            "help": "The name of the column in the datasets containing the image encodings."
        },
    )
    dataset_repo_or_path: str = field(
        default=None,
        metadata={"help": "The dataset repository containing encoded files."},
    )
    # CHECK LATER
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file (glob & braceexpand acceptable)."
        },
    )
    # data loading should not be a bottleneck so we use "streaming" mode by default
    streaming: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to stream the dataset."},
    )
    shard_by_host: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to shard data files by host in multi-host environments."
        },
    )
    clip_score_column: Optional[str] = field(
        default="clip_score",
        metadata={"help": "Column that containts clip score for filtering."},
    )
    min_clip_score: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum clip score required."},
    )
    max_clip_score: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum clip score required."},
    )
    # CHECK LATER
    filter_column: Optional[str] = field(
        default=None,
        metadata={"help": "Column that containts classes to be filtered."},
    )
    # CHECK LATER
    filter_value: Optional[str] = field(
        default=None,
        metadata={"help": "Class value to be kept during filtering."},
    )
    seed_dataset: int = field(
        default=None,
        metadata={
            "help": "Random seed for the dataset that will be set at the beginning of training."
        },
    )

    def __post_init__(self):
        if self.dataset_repo_or_path is None:
            raise ValueError("Need a dataset repository or path.")


@dataclass
class TrainingArguments:
    """
    Arguments related to training parameters.
    """

    output_dir: str = field(
        metadata={
            "help": "The output directory for model predictions and checkpoints."
        },
    )

    do_train: bool = field(
        default=True,
        metadata={
            "help": "Whether to run training."
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per data parallel device for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing an update pass."
        },
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate."}
    )
    optim: str = field(
        default="distributed_shampoo",
        metadata={
            "help": 'The optimizer to use. Can be "distributed_shampoo", "adam" or "adafactor"'
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay applied to parameters."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    block_size: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": "Type of grafting"
            "['rmsprop_normalized', 'rmsprop', 'adagrad', 'adagrad_normalized', 'sgd', 'sqrt_n']"
        },
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Use Nesterov momentum for Distributed Shampoo."},
    )
    optim_quantized: bool = field(
        default=True,
        metadata={
            "help": "Whether to quantize optimizer (only supported with Distributed Shampoo)."
        },
    )
    shard_shampoo_across: str = field(
        default="dp",
        metadata={
            "help": "Shard the optimizer across data devices (dp), model devices (mp), both (2d)."
        },
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    lr_decay: str = field(
        default=None,
        metadata={
            "help": "In the learning rate scheduler. None (default), linear or exponential."
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": "Number associated with learning rate decay when using exponential decay."
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={
            "help": "Decay rate associated with learning rate when using exponential decay."
        },
    )
    lr_staircase: bool = field(
        default=False,
        metadata={
            "help": "Whether to use staircase or continuous learning rate when using exponential decay."
        },
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
     )
    logging_steps: int = field(
        default=40, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=400, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=4000, metadata={"help": "Save checkpoint every X updates steps."}
    )
    # CHECK LATER
    log_model: bool = field(
        default=False,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )
    # CHECK LATER
    log_norm_steps: int = field(
        default=True,
        metadata={"help": "Log parameters and gradients norm at this frequency."},
    )
    # CHECK LATER
    log_histogram_steps: int = field(
        default=False,
        metadata={
            "help": "Frequency to log parameters and gradients histograms. Slows down training."
        },
    )
    seed_model: int = field(
        default=42,
        metadata={
            "help": "Random seed for the model that will be set at the beginning of training."
        },
    )
    # CHECK LATER
    embeddings_only: bool = field(
        default=False, metadata={"help": "Train only embedding layers."}
    )
    # CHECK LATER
    init_embeddings: bool = field(
        default=False,
        metadata={"help": "When training embedding layers, initialize them."},
    )
    assert_tpu_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )
    use_vmap_trick: bool = field(
        default=True,
        metadata={"help": "Framework to apply same operation to many for parallel processing"},
    )
    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of devices required for model parallelism."
            "The other dimension of available devices is used for data parallelism."
        },
    )
    dp_devices: int = field(init=False)
    def __post_init__(self):
        if self.assert_tpu_available:
            assert (
                jax.local_device_count() == 8
            ), "TPUs in use, please check running processes"

        assert self.optim in [
            "distributed_shampoo",
        ], f"Selected optimizer not supported: {self.optim}"
        
        if self.optim == "adafactor" and self.weight_decay == 0:
            self.weight_decay = None
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        
        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"
        
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
            
        if self.log_norm_steps is True:
            self.log_norm_steps = self.logging_steps
            
        if not self.do_train:
            self.num_train_epochs = 1
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
            
        assert self.shard_shampoo_across in [
            "dp",
            "mp",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."
        
        assert (
            self.mp_devices > 0
        ), f"Number of devices for model parallelism must be > 0"
        
        assert (
            jax.device_count() % self.mp_devices == 0
        ), f"Number of available devices ({jax.device_count()} must be divisible by number of devices used for model parallelism ({self.mp_devices})."
        self.dp_devices = jax.device_count() // self.mp_devices