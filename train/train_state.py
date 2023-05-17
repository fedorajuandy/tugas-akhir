from flax import core, struct, traverse_util

# define TrainState
class TrainState(struct.PyTreeNode):
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
        print(f"Ra's here. Start applying gradients...")

        grads = split_params(trainable_params(grads, training_args.embeddings_only))
#             print(f"RA: grads = {grads}")
        params = split_params(
            trainable_params(self.params, training_args.embeddings_only)
        )
#             print(f"RA: params = {params}")
        opt_state = {}
        
        # we loop over keys: "standard", "scanned_encoder", "scanned_decoder"
        print(f"Ra's here. Start looping...")
        for k, param in params.items():
            update_fn = self.tx[k].update
#                 print(f"RA: update_fn = {update_fn}")
            if "scanned" in k:
                update_fn = jax.vmap(update_fn, in_axes=(0, 0, 0), out_axes=(0, 0))
#                     print(f"RA: update_fn = {update_fn}")
            updates, new_opt_state = update_fn(grads[k], self.opt_state[k], param)
#                 print(f"RA: update_fn = {update_fn}")
#                 print(f"RA: update_fn = {update_fn}")
            params[k] = optax.apply_updates(param, updates)
#                 print(f"RA: params[k] = {params[k]}")
            opt_state[k] = new_opt_state
#                 print(f"RA: opt_state[k] = {opt_state[k]}")
        print(f"Ra's here. Loop ends.")
        
        print(f"Ra's here. Params stuff here.")
        params = unsplit_params(params)
#             print(f"RA: params = {params}")
        # merge with non-trainable params
        params, new_params = traverse_util.flatten_dict(
            unfreeze(self.params)
        ), traverse_util.flatten_dict(unfreeze(params))
#             print(f"RA: params = {params}")
#             print(f"RA: new_params = {new_params}")
        params.update(new_params)
#             print(f"RA: params = {params}")
        params = freeze(traverse_util.unflatten_dict(params))
#             print(f"RA: params = {params}")

#             print(f"RA: step = {self.step + 1}")
#             print(f"RA: opt_state = {freeze(opt_state)}")
#             print(f"RA: kwargs = {kwargs}")

        return self.replace(
            step=self.step + 1,
            params=params,
            opt_state=freeze(opt_state),
            **kwargs,
        )

    # DILUC
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = {}
        for k, p in split_params(
            trainable_params(params, training_args.embeddings_only)
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