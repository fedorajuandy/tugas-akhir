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
""" Main training file adapted from DALL-E mini's training script """

import time
import tempfile
from dataclasses import asdict, dataclass, field
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
import wandb
from datasets import Dataset
from flax import core, struct, traverse_util
from flax.core.frozen_dict import freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.training.common_utils import onehot
from jax.sharding import PartitionSpec
from jax.experimental import maps
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

cc.initialize_cache("jax_cache")


@dataclass
class ModelArguments:
    """ Model's confirguration """
    # Notes: modify save-point to anticipate workplace runtime limitation, then use checkpoint

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for weights initialization."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "If using different config name or path from model."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "If using different tokenizer name or path from model."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Computations format (memory--, precision--): float32, bfloat16, float16."
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Restore optimizer and training state with checkpoint."
        },
    )
    dropout: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Rate to prevent overfitting by neuron's output. Overwrites config."
        },
    )
    activation_dropout: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Rate to prevent overfitting by layer's output. Overwrites config."
        },
    )
    attention_dropout: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Rate to prevent overfitting by transformers. Overwrites config."
        },
    )


    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
            assert (
                self.tokenizer_name is not None
            ), "tokenizer_name or model_name_or_path needs to be specified."
        if self.restore_state:
            assert (
                self.model_name_or_path is not None and "/model-" in self.model_name_or_path
            ), "model_name_or_path with W&B artifact is needed."


    def get_metadata(self):
        """ Get artifact's metadata or empty dict """

        if self.model_name_or_path is not None and ":" in self.model_name_or_path:
            if jax.process_index() == 0:
                artifact = wandb.run.use_artifact(self.model_name_or_path)
            else:
                artifact = wandb.Api().artifact(self.model_name_or_path)
            return artifact.metadata
        else:
            return dict()


    def get_opt_state(self):
        """ Get training state """

        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.restore_state is True:
                state_artifact = self.model_name_or_path.replace(
                    "/model-", "/state-", 1
                )

                if jax.process_index() == 0:
                    artifact = wandb.run.use_artifact(state_artifact)
                else:
                    artifact = wandb.Api().artifact(state_artifact)

                artifact_dir = artifact.download(tmp_dir)
                self.restore_state = str(Path(artifact_dir) / "opt_state.msgpack")

            with Path(self.restore_state).open("rb") as file:
                return file.read()


@dataclass
class DataTrainingArguments:
    """ Dataset's configuration """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "Dataset's column name for captions."
        },
    )
    encoding_column: Optional[str] = field(
        default="encoding",
        metadata={
            "help": "Dataset's column name for image encodings."
        },
    )
    dataset_repo_or_path: str = field(
        default=None,
        metadata={
            "help": "Dataset's location."
        },
    )
    streaming: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to stream the dataset to prevent bottleneck."
        },
    )
    seed_dataset: int = field(
        default=None,
        metadata={
            "help": "Random seed so the training will not repeat the same data when interrupted."
        },
    )

    def __post_init__(self):
        if self.dataset_repo_or_path is None:
            raise ValueError("Need a dataset repository or path.")


@dataclass
class TrainingArguments:
    """ Training's configuration """

    output_dir: str = field(
        metadata={
            "help": "Directory to write model predictions and checkpoints."
        },
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite or continue from checkpoint if enabled "
            )
        },
    )
    do_train: bool = field(
        default=True,
        metadata={
            "help": "Whether to run training."
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run eval on the validation set."
        }
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size per data parallel device."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before updating pass."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "To reduce memory usage with more time usage."
        },
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={
            "help": "The initial learning rate."
        },
    )
    optim: str = field(
        default="distributed_shampoo",
        metadata={
            "help": "Optimizer."
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "Weight decays applied to parameters."
        },
    )
    beta1: float = field(
        default=0.9,
        metadata={
            "help": "Hyperparameters; exponential decay rates."
        },
    )
    beta2: float = field(
        default=0.999,
        metadata={
            "help": "Hyperparameters; second moment (squared gradient average)."
        },
    )
    block_size: int = field(
        default=1024,
        metadata={
            "help": "Chunked size for large layers."
        },
    )
    preconditioning_compute_steps: int = field(
        default=10,
        metadata={
            "help": "Number of steps to update preconditioner."
        },
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={
            "help": "Max size for preconditioning."
        },
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": "Maintain moving average of the square of gradients."
            # rmsprop_normalized, rmsprop, adagrad, adagrad_normalized, sgd, sqrt_n
        },
    )
    nesterov: bool = field(
        default=False,
        metadata={
            "help": "Nesterov momentum to improve convergence."
        },
    )
    optim_quantized: bool = field(
        default=True,
        metadata={
            "help": "Shard optimizer across devices."
        },
    )
    shard_shampoo_across: str = field(
        default="dp",
        metadata={
            "help": "Shard optimizer across data devices (dp), model devices (mp), or both (2d)."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={
            "help": "Total number of complete iteration."
        },
    )
    warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Linear warmup over warmup_steps; gradually increases learning rate."
        },
    )
    lr_decay: str = field(
        default=None,
        metadata={
            "help": "Learning rate scheduler's decay; none, linear, or exponential."
        },
    )
    lr_offset: int = field(
        default=0,
        metadata={
            "help": "Number of steps to offset learning rate and keep it at 0."
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": "Learning rate's transition steps when using exponential decay."
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={
            "help": "Learning rate's decay rate (number of steps) when using exponential decay."
        },
    )
    lr_staircase: bool = field(
        default=False,
        metadata={
            "help": "Staircase or continuous learning rate when using exponential decay."
        },
    )
    save_steps: int = field(
        default=3000,
        metadata={
            "help": "Save checkpoint every save_steps updates steps."
        },
    )
    log_model: bool = field(
        default=True,
        metadata={
            "help": "Log model to W&B at save_steps frequency."
        },
    )
    log_norm_steps: int = field(
        default=True,
        metadata={
            "help": "Log parameters and gradients norm at this frequency."
        },
    )
    seed_model: int = field(
        default=42,
        metadata={
            "help": "Random seed to initialise weights."
        },
    )
    wandb_project: str = field(
        default="tugas-akhir",
        metadata={
            "help": "W&B project's name."
        },
    )
    wandb_job_type: str = field(
        default="Seq2Seq",
        metadata={
            "help": "W&B project's job type."
        },
    )
    assert_tpu_available: bool = field(
        default=False,
        metadata={
            "help": "Whether using TPU or not."
        },
    )
    use_vmap_trick: bool = field(
        default=True,
        metadata={
            "help": "Apply same operation to many at the same time for parallel processing"
        },
    )
    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of devices for model parallelism."
        },
    )
    dp_devices: int = field(
        init=False,
        metadata={
            "help": "Number of devices for data parallelism based on mp_devices."
        },
    )

    def __post_init__(self):
        if self.assert_tpu_available:
            assert (
                jax.local_device_count() == 8
            ), "TPUs in use, please check running processes."

        assert self.optim in [
            "distributed_shampoo",
        ], f"Selected optimizer not supported: {self.optim}"

        assert self.graft_type in [
            "rmsprop_normalized",
            # "rmsprop",
            # "adagrad",
            # "adagrad_normalized",
            # "sgd",
            # "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"

        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"

        assert self.shard_shampoo_across in [
            "dp",
            "mp",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."

        assert (
            self.mp_devices > 0
        ), "Number of devices for model parallelism must be > 0"

        assert (
            jax.device_count() % self.mp_devices == 0
        ), "Number of available devices must be divisible by mp_devices."

        self.dp_devices = jax.device_count() // self.mp_devices


def split_params(data):
    """ Split train params between scanned and non-scanned """

    flat = traverse_util.flatten_dict(unfreeze(data))
    split = {"standard": {}, "scanned_encoder": {}, "scanned_decoder": {}}

    for k, v in flat.items():
        if "FlaxBartEncoderLayers" in k:
            split["scanned_encoder"][k] = v
        elif "FlaxBartDecoderLayers" in k:
            split["scanned_decoder"][k] = v
        else:
            split["standard"][k] = v

    split = {k: v for k, v in split.items() if v}

    for k, v in split.items():
        split[k] = freeze(traverse_util.unflatten_dict(v))

    return split


def unsplit_params(data):
    """" Unsplit train params """

    flat = {}

    for k in ["standard", "scanned_encoder", "scanned_decoder"]:
        if k in data:
            flat.update(traverse_util.flatten_dict(unfreeze(data[k])))

    return freeze(traverse_util.unflatten_dict(flat))


def trainable_params(data):
    """ Keep only trainable parameters """

    return data


def main():
    """ Main function """

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.mp_devices > jax.local_device_count():
        assert (
            data_args.seed_dataset is not None
        ), "Seed dataset must be provided when model is split over multiple hosts."

    # Load dataset
    dataset = Dataset(
        **asdict(data_args),
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )

    # Initialize wandb
    if jax.process_index() == 0:
        wandb.init(
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=parser.parse_args(),
        )

    # Set up config
    config_args = {
        k: getattr(model_args, k)
        for k in ["dropout", "activation_dropout", "attention_dropout"]
        if getattr(model_args, k) is not None
    }
    config_args["gradient_checkpointing"] = training_args.gradient_checkpointing

    if model_args.config_name:
        config = DalleBartConfig.from_pretrained(model_args.config_name)
    else:
        config = None

    # Load or create model
    if model_args.model_name_or_path:
        model, params = DalleBart.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
    else:
        model = DalleBart(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        params = None

    for k, v in config_args.items():
        setattr(model.config, k, v)

    params_shape = model.params_shape_tree
    model_metadata = model_args.get_metadata()

    # break parameters for parallel computation
    param_spec = set_partitions(params_shape, model.config.use_scan)
    params_shape = freeze(params_shape)

    if params is not None:
        params = freeze(params)

    # Load tokenizer
    tokenizer = DalleBartTokenizer.from_pretrained(
        model_args.tokenizer_name, use_fast=True
    )

    # Tokenize words according tokenizer and config
    dataset.preprocess(tokenizer=tokenizer, config=model.config)

    # Initialize training
    dropout_rng = jax.random.PRNGKey(training_args.seed_model)
    num_epochs = training_args.num_train_epochs

    batch_size_per_node_per_grad_step = (
        training_args.per_device_train_batch_size * jax.local_device_count() // training_args.mp_devices
    )
    batch_size_per_node = (
        batch_size_per_node_per_grad_step * training_args.gradient_accumulation_steps
    )
    batch_size_per_step = batch_size_per_node * jax.process_count()

    len_train_dataset, len_eval_dataset = dataset.length
    steps_per_epoch = (
        len_train_dataset // batch_size_per_node
        if len_train_dataset is not None
        else None
    )
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )
    num_params = model.num_params(params_shape)

    if jax.process_index() == 0:
        wandb.define_metric("*", step_metric="train/step") # x-axis
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

    def create_learning_rate_fn() -> Callable[[int], jnp.array]:
        """ Create the learning rate (step size each update) function """

        # gradually increases learning rate from init_value to training_args.learning_rate
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=training_args.learning_rate,
            transition_steps=training_args.warmup_steps + 1,  # ensure not 0
        )
        last_boundary = training_args.warmup_steps

        if training_args.lr_offset:
            # Append 0 constant schedule before warm-up schedule then update last boundary (train from checkpoint)
            warmup_fn = optax.join_schedules(
                schedules=[optax.constant_schedule(0.0), warmup_fn],
                boundaries=[training_args.lr_offset],
            )
            last_boundary += training_args.lr_offset

        if training_args.lr_decay is None:
            return warmup_fn
        elif training_args.lr_decay == "linear":
            assert (
                num_train_steps is not None
            ), "Linear decay requires knowing the dataset length."
            decay_fn = optax.linear_schedule(
                init_value=training_args.learning_rate,
                end_value=0,
                transition_steps=num_train_steps - training_args.warmup_steps,
            )
        elif training_args.lr_decay == "exponential":
            decay_fn = optax.exponential_decay(
                init_value=training_args.learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )

        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[last_boundary],
        )

        return schedule_fn

    # create optimizer
    learning_rate_fn = create_learning_rate_fn()
    trainable_params_shape = trainable_params(
        params_shape
    )

    graft_type = {
        "sgd": GraftingType.SGD,
        "adagrad": GraftingType.ADAGRAD,
        "rmsprop": GraftingType.RMSPROP,
        "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
        "sqrt_n": GraftingType.SQRT_N,
        "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
    }[training_args.graft_type]

    statistics_partition_spec = (
        PartitionSpec(None, training_args.shard_shampoo_across, None)
        if training_args.shard_shampoo_across != "2d"
        else PartitionSpec(None, "dp", "mp")
    )

    opt = distributed_shampoo(
        learning_rate_fn,
        block_size=training_args.block_size,
        beta1=training_args.beta1,
        beta2=training_args.beta2,
        diagonal_epsilon=1e-10,
        matrix_epsilon=1e-6,
        weight_decay=training_args.weight_decay,
        start_preconditioning_step=max(
            training_args.preconditioning_compute_steps + 1, 101
        ),
        preconditioning_compute_steps=training_args.preconditioning_compute_steps,
        statistics_compute_steps=1,
        best_effort_shape_interpretation=True,
        graft_type=graft_type,
        nesterov=training_args.nesterov,
        exponent_override=0,
        statistics_partition_spec=statistics_partition_spec,
        preconditioner_partition_spec=PartitionSpec(
            training_args.shard_shampoo_across,
            None,
            None
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
        moving_average_for_momentum=True,
        skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=None,
        precision=jax.lax.Precision.HIGHEST,
        best_effort_memory_usage_reduction=training_args.optim_quantized,
    )

    update_fn = opt.update
    optimizer = {}
    opt_fn = {}

    # initialize optimizer with training parameters
    for k, p in split_params(trainable_params_shape).items():
        if "scanned" in k:
            p = jax.eval_shape(
                lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p
            )

        optimizer[k] = opt.init(p)
        opt_fn[k] = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
            optimizer[k].pspec_fn, optimizer[k].shape_and_dtype_fn
        )
        optimizer[k] = optax.GradientTransformation(optimizer[k].init_fn, update_fn)

    def get_opt_state_spec_and_shape():
        """ Get PartitionSpec for optimizer state """

        opt_state_shape = {}

        for k, p in split_params(trainable_params_shape).items():
            if "scanned" not in k:
                opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)
            else:
                opt_state_shape[k] = jax.eval_shape(jax.vmap(optimizer[k].init), p)

        split_spec = split_params(set_partitions(trainable_params_shape, False))
        opt_state_spec = {}

        for k, p in split_params(trainable_params_shape).items():
            if "scanned" in k:
                p = jax.eval_shape(
                    lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p
                )

            opt_state_spec[k] = opt_fn[k].pspec_fn(
                p,
                split_spec[k],
                statistics_partition_spec,
            )

            # Add dimension for scanned params
            if "scanned" in k:
                opt_state_spec[k] = jax.tree_util.tree_map(
                    lambda x: PartitionSpec(*(None,) + x)
                    if x is not None
                    else None,
                    opt_state_spec[k],
                    is_leaf=lambda x: isinstance(x, PartitionSpec),
                )

        return freeze(opt_state_spec), freeze(opt_state_shape)

    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape()

    # Represent data as matrix
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))

    class TrainState(struct.PyTreeNode):
        """ Define training's state """

        step: int
        params: core.FrozenDict[str, Any]
        opt_state: optax.OptState
        apply_fn: Callable = struct.field(pytree_node=False)
        tx: optax.GradientTransformation = struct.field(pytree_node=False)
        dropout_rng: jnp.ndarray = None
        epoch: int = 0
        train_time: float = 0.0
        train_samples: int = 0

        def apply_gradients(self, *, grads, **kwargs):
            """ One step of training """

            grads = split_params(
                trainable_params(grads)
            )
            params = split_params(
                trainable_params(self.params)
            )
            opt_state = {}

            # Loop over keys: "standard", "scanned_encoder", "scanned_decoder"
            for k, param in params.items():
                update_fn = self.tx[k].update

                if "scanned" in k:
                    update_fn = jax.vmap(update_fn, in_axes=(0, 0, 0), out_axes=(0, 0))

                updates, new_opt_state = update_fn(grads[k], self.opt_state[k], param)
                params[k] = optax.apply_updates(param, updates)
                opt_state[k] = new_opt_state

            params = unsplit_params(params)
            params, new_params = traverse_util.flatten_dict(
                unfreeze(self.params)
            ), traverse_util.flatten_dict(unfreeze(params))
            params.update(new_params)
            params = freeze(traverse_util.unflatten_dict(params))

            return self.replace(
                step=self.step + 1,
                params=params,
                opt_state=freeze(opt_state),
                **kwargs,
            )

        @classmethod
        def create(cls, *, apply_fn, params, tx, **kwargs):
            """ Creating new training step """

            opt_state = {}

            for k, p in split_params(
                trainable_params(params)
            ).items():
                init_fn = tx[k].init

                if "scanned" in k:
                    init_fn = jax.vmap(init_fn)

                opt_state[k] = init_fn(p)

            return cls(
                step=0,
                apply_fn=apply_fn,
                params=params,
                tx=tx,
                opt_state=freeze(opt_state),
                **kwargs,
            )

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

    def maybe_init_params(params):
        """ Init params if not available yet """

        if params is not None:
            return params
        else:
            return model.init_weights(model.key, model.input_shape)

    # Create state
    with mesh:
        # Restore state
        attr_state = {}
        keys = ["train_time", "train_samples"]
        if model_args.restore_state:
            keys += ["step", "epoch"]
        attr_state = {k: v for k, v in model_metadata.items() if k in keys}

        if not model_args.restore_state:
            def init_state(params):
                """ Initialize new state """

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
            opt_state = from_bytes(opt_state_shape, model_args.get_opt_state())

            def restore_state(params, opt_state):
                """ Load state """

                return TrainState(
                    apply_fn=model.__call__,
                    tx=optimizer,
                    params=params,
                    opt_state=opt_state,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                restore_state,
                in_axis_resources=(
                    param_spec,
                    opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(params, opt_state)

            # Free stuff from CPU memory
            del opt_state

    del params, opt_state_spec, opt_state_shape

    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    def loss_fn(logits, labels):
        """ Define loss """

        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Avoid crash
    use_vmap_trick = training_args.use_vmap_trick

    # Make grad_param_spec for vmap
    if use_vmap_trick:
        grad_param_spec = jax.tree_util.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            param_spec,
        )

    def train_step(state, batch, train_time):
        """ Update gradient each step """

        def get_minibatch(batch, grad_idx):
            """ Get a minibatch (one gradient accumulation slice) """

            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        def compute_loss(params, minibatch, dropout_rng):
            """ Loss function """

            minibatch, labels = minibatch.pop("labels")
            logits = state.apply_fn(
                **minibatch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            return loss_fn(logits, labels)

        grad_fn = jax.value_and_grad(compute_loss)

        def loss_and_grad(grad_idx, dropout_rng):
            """ Get loss, gradient, and dropout for each step """

            minibatch = (
                get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            )
            minibatch = with_sharding_constraint(minibatch, batch_spec)

            dropout_rng, _ = jax.random.split(dropout_rng)

            if use_vmap_trick:
                # Get values per device
                loss, grads = jax.vmap(
                    grad_fn, in_axes=(None, 0, None), out_axes=(0, 0)
                )(state.params, minibatch, dropout_rng)
                loss = with_sharding_constraint(loss, batch_spec)
                grads = with_sharding_constraint(grads, grad_param_spec)
                # Average values across all devices
                loss, grads = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0), (loss, grads)
                )
            else:
                loss, grads = grad_fn(state.params, minibatch, dropout_rng)

            grads = with_sharding_constraint(grads, param_spec)

            return loss, grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, dropout_rng = loss_and_grad(None, state.dropout_rng)
        else:
            # Create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                0.0,
                with_sharding_constraint(
                    jax.tree_util.tree_map(jnp.zeros_like, state.params), param_spec
                ),
                state.dropout_rng,
            )

            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                """ Accumulate gradients """

                cumul_loss, cumul_grads, dropout_rng = cumul_loss_grad_dropout
                loss, grads, dropout_rng = loss_and_grad(grad_idx, dropout_rng)
                cumul_loss, cumul_grads = jax.tree_util.tree_map(
                    jnp.add, (cumul_loss, cumul_grads), (loss, grads)
                )
                cumul_grads = with_sharding_constraint(cumul_grads, param_spec)

                return cumul_loss, cumul_grads, dropout_rng

            # Loop over gradients
            loss, grads, dropout_rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step,
            )
            grads = with_sharding_constraint(grads, param_spec)

            loss, grads = jax.tree_util.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps, (loss, grads)
            )

        grads = with_sharding_constraint(grads, param_spec)

        # Uspdate state
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

        def maybe_fn(function, val, zeros, freq):
            """ Log update """

            return jax.lax.cond(
                state.step % freq == 0,
                function,
                lambda _: zeros,
                val,
            )

        params = trainable_params(state.params)
        grads = trainable_params(grads)
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

        return state, metrics

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

    # Keep local copy of state
    local_state = {
        k: jax.device_get(getattr(state, k)).item()
        for k in ["step", "epoch", "train_time", "train_samples"]
    }
    start_time = time.perf_counter() - local_state["train_time"]
    save_model_ran = False
    epochs = tqdm(
        range(local_state["epoch"], num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    def run_save_model(state):
        """ Save model """

        if jax.process_index() == 0:
            output_dir = training_args.output_dir

            params = jax.device_get(state.params)
            model.save_pretrained(
                output_dir,
                params=params,
            )

            tokenizer.save_pretrained(output_dir)

            opt_state = jax.device_get(state.opt_state)
            with (Path(output_dir) / "opt_state.msgpack").open("wb") as file:
                file.write(to_bytes(opt_state))

            if training_args.log_model:
                metadata = {
                    k: jax.device_get(getattr(state, k)).item()
                    for k in ["step", "epoch", "train_time", "train_samples"]
                }
                metadata["num_params"] = num_params

                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="DalleBart_model",
                    metadata=metadata,
                )

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

                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="DalleBart_state",
                    metadata=metadata,
                )
                artifact_state.add_file(
                    f"{Path(training_args.output_dir) / 'opt_state.msgpack'}"
                )

                wandb.run.log_artifact(artifact_state)

    # Start training
    with mesh:
        for epoch in epochs:
            state = state.replace(epoch=epoch)
            local_state["epoch"] = epoch

            # Load data
            node_groups = max(
                1, training_args.mp_devices // jax.local_device_count()
            )
            loader_bs = batch_size_per_node * node_groups
            train_loader = dataset.dataloader(
                "train",
                loader_bs,
                epoch,
            )

            for batch in tqdm(
                train_loader,
                desc="Training...",
                position=1,
                leave=False,
                total=steps_per_epoch,
                disable=jax.process_index() > 0,
            ):
                train_time = time.perf_counter() - start_time

                save_model_ran = False

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

                if training_args.gradient_accumulation_steps > 1:
                    bs_shape = (
                        training_args.gradient_accumulation_steps,
                    ) + bs_shape

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape(bs_shape + x.shape[1:]),
                    batch,
                )
                batch = freeze(batch)

                state = p_train_step(state, batch, train_time)
                local_state["step"] += 1
                local_state["train_time"] = train_time
                local_state["train_samples"] += batch_size_per_step

                if local_state["step"] % training_args.save_steps == 0:
                    run_save_model(state)
                    save_model_ran = True

            # Save each epoch
            if not save_model_ran:
                run_save_model(state)


if __name__ == "__main__":
    main()
