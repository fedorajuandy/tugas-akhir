""" Main training file """
# pylint: disable=too-many-lines
# pylint: disable=line-too-long
# pylint: disable=f-string-without-interpolation
# pylint: disable=invalid-name

import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple
import datasets
import flax
import jax
import jax.numpy as jnp
import jaxlib # pylint: disable=import-error # type: ignore
import numpy as np
import optax # pylint: disable=import-error # type: ignore
import transformers
import wandb
from datasets import Dataset
from flax import traverse_util
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.training.common_utils import onehot
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from scalable_shampoo.distributed_shampoo import GraftingType, distributed_shampoo #pylint: disable=import-error
from tqdm import tqdm
from transformers import HfArgumentParser

import dalle_mini
from dalle_mini.data import Dataset # pylint: disable=import-error # pylint: disable=reimported
from dalle_mini.model import (
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer,
    set_partitions,
)
from .arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from .train_state import TrainState
from .metrics_logger import MetricsLogger

cc.initialize_cache("jax_cache")


def split_params(data):
    """Split params between scanned and non-scanned"""

    flat = traverse_util.flatten_dict(unfreeze(data))
    split = {"standard": {}, "scanned_encoder": {}, "scanned_decoder": {}}

    for k, v in flat.items(): # pylint: disable=invalid-name
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
    """" Unsplitting parameterse """

    flat = {}

    for k in ["standard", "scanned_encoder", "scanned_decoder"]:
        if k in data:
            flat.update(traverse_util.flatten_dict(unfreeze(data[k])))

    return freeze(traverse_util.unflatten_dict(flat))


def trainable_params(data):
    """Keep only trainable parameters"""

    data = unfreeze(data)

    trainable = {
        "lm_head": data["lm_head"],
        "model": {
            "decoder": {
                layer: data["model"]["decoder"][layer]
                for layer in [
                    "embed_positions",
                    "embed_tokens",
                    "final_ln",
                    "layernorm_embedding",
                ]
            }
        },
    }

    return freeze(trainable)


def init_embeddings(model, params):
    """Reinitialize trainable embeddings"""

    trainable_keypaths = [
        "lm_head.kernel",
        "model.decoder.embed_positions.embedding",
        "model.decoder.embed_tokens.embedding",
        "model.decoder.final_ln.bias",
        "model.decoder.layernorm_embedding.bias",
        "model.decoder.layernorm_embedding.scale",
    ]

    init_keys = {tuple(k.split(".")) for k in trainable_keypaths}
    model._missing_keys = init_keys # pylint: disable=protected-access

    return model.init_weights(model.key, model.input_shape, params=params)


def main():
    """ Main function """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses() # pylint: disable=unbalanced-tuple-unpacking

    if training_args.mp_devices > jax.local_device_count():
        assert (
            data_args.seed_dataset is not None
        ), "Seed dataset must be provided when model is split over multiple hosts"

    dataset = Dataset(
        **asdict(data_args),
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )

    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=parser.parse_args(),
        )

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
    param_spec = set_partitions(params_shape, model.config.use_scan)
    params_shape = freeze(params_shape)

    if params is not None:
        params = freeze(params)

    tokenizer = DalleBartTokenizer.from_pretrained(
        model_args.tokenizer_name, use_fast=True
    )

    dataset.preprocess(tokenizer=tokenizer, config=model.config)
    dropout_rng = jax.random.PRNGKey(training_args.seed_model)
    num_epochs = training_args.num_train_epochs
    batch_size_per_node_per_grad_step = (
        training_args.per_device_train_batch_size
        * jax.local_device_count()
        // training_args.mp_devices
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
        wandb.define_metric("*", step_metric="train/step")
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
        """Create the learning rate function."""

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=training_args.learning_rate,
            transition_steps=training_args.warmup_steps + 1,  # ensure not 0
        )
        last_boundary = training_args.warmup_steps

        if training_args.lr_offset:
            # appends a constant schedule of 0 before the warm-up schedule, and updates the last boundary (for resuming training from a checkpoint)
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
            ), "linear decay requires knowing the dataset length"
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
        moving_average_for_momentum=True,
        skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=None,
        precision=jax.lax.Precision.HIGHEST,
        best_effort_memory_usage_reduction=training_args.optim_quantized,
    )

    update_fn = opt.update
    optimizer = {}
    opt_fn = {}

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
        opt_state_shape = {}

        for k, p in split_params(trainable_params_shape).items():
            if "scanned" not in k:
                opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)
            else:
                opt_state_shape[k] = jax.eval_shape(jax.vmap(optimizer[k].init), p)

        if training_args.optim in ["adam", "distributed_shampoo"]:
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

        return freeze(opt_state_spec), freeze(opt_state_shape)

    print(f"Ra's here. Mesh stuff here..?")
    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape()

    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))

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
        if params is not None:
            return params
        else:
            return model.init_weights(model.key, model.input_shape)

    with mesh:
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

            state = pjit(
                restore_state,
                in_axis_resources=(
                    param_spec,
                    opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(params, opt_state)

            del opt_state

    del params, opt_state_spec, opt_state_shape

    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

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

        params = trainable_params(state.params)
        grads = trainable_params(grads)
        if training_args.log_norm_steps:
            zeros_norm = jax.tree_util.tree_map(lambda _: jnp.float32(0), params)

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val) # pylint: disable=unnecessary-lambda

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


    # keep local copy of state
    local_state = {
        k: jax.device_get(getattr(state, k)).item()
        for k in ["step", "epoch", "train_time", "train_samples"]
    }
    start_time = time.perf_counter() - local_state["train_time"]
    save_model_ran = False
    metrics_logger = MetricsLogger(local_state["step"])
    epochs = tqdm(
        range(local_state["epoch"], num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    def run_save_model(state):
        if jax.process_index() == 0:
            output_dir = training_args.output_dir

            params = jax.device_get(state.params)
            model.save_pretrained(
                output_dir,
                params=params,
            )

            tokenizer.save_pretrained(output_dir)

            opt_state = jax.device_get(state.opt_state)
            with (Path(output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(opt_state))

            if training_args.log_model:
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("20GB"))

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

    with mesh:
        for epoch in epochs:
            state = state.replace(epoch=epoch)
            local_state["epoch"] = epoch
            metrics_logger.update_state_metrics(local_state)
            metrics_logger.log({})

            if training_args.do_train:
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
                        lambda x: x.reshape(bs_shape + x.shape[1:]), # pylint: disable=cell-var-from-loop
                        batch,
                    )
                    batch = freeze(batch)

                    state, train_metrics = p_train_step(state, batch, train_time) # KAGGLE
                    local_state["step"] += 1
                    local_state["train_time"] = train_time
                    local_state["train_samples"] += batch_size_per_step

                    if local_state["step"] % training_args.save_steps == 0:
                        run_save_model(state)
                        save_model_ran = True


            if not save_model_ran:
                run_save_model(state)


if __name__ == "__main__":
    main()
