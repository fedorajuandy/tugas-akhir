""" Training maybe """
# pylint: disable=line-too-long
# pylint: disable=abstract-method

import os
import sys
from typing import Any, Callable
from flax import core, struct, traverse_util
from flax.core.frozen_dict import freeze, unfreeze
import jax
import jax.numpy as jnp
import optax # pylint: disable=import-error # type: ignore
from transformers import HfArgumentParser
from .main import split_params, trainable_params, unsplit_params
from .arguments import ModelArguments, DataTrainingArguments, TrainingArguments

parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_json_file(
    json_file=os.path.abspath(sys.argv[1])
)

# define TrainState
class TrainState(struct.PyTreeNode):
    """ Training """

    step: int
    params: core.FrozenDict[str, Any]
    opt_state: optax.OptState
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def apply_gradients(self, *, grads, **kwargs):
        """ One step of training """

        grads = split_params(
            trainable_params(grads)
        )
        params = split_params(
            trainable_params(self.params)
        )
        opt_state = {}

        for k, param in params.items():
            update_fn = self.tx[k].update # pylint: disable=unsubscriptable-object

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

        for k, p in split_params( # pylint: disable=invalid-name
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
