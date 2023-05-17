# define metrics logger
class MetricsLogger:
    def __init__(self, step):
        # keep state
        self.state_dict = {}
        # estimate speed
        self.step = step
        self.time = time.perf_counter()
        self.offset_time = 0.0

    def update_state_metrics(self, state):
        """Update internal state metrics (logged at each call to be used as x-axis)"""
        self.state_dict = {
            f'train/{k.split("_")[-1]}': state[k]
            for k in ["step", "epoch", "train_time", "train_samples"]
        }
        # timing metrics
        new_step = int(state["step"])
        new_time = time.perf_counter()
        if new_step > self.step:
            # remove time for eval & save
            delta_time = new_time - self.time - self.offset_time
            self.offset_time = 0
            time_per_step = delta_time / (new_step - self.step)
            self.step = new_step
            self.time = new_time
            self.log_time("train_per_step", time_per_step, offset=False)
            self.log_time("train_per_log", delta_time, offset=False)

    def log_time(self, key, duration, offset=True):
        if jax.process_index() == 0:
            wandb.log({f"time/{key}": duration, **self.state_dict})
        if offset:
            self.offset_time += duration

    def log(self, metrics, prefix=None):
        if jax.process_index() == 0:
            log_metrics = {}
            for k, v in metrics.items():
                if "_norm" in k:
                    if self.step % training_args.log_norm_steps == 0:
                        log_metrics[f"{k}/"] = unfreeze(v)
                elif "_hist" in k:
                    if self.step % training_args.log_histogram_steps == 0:
                        v = jax.tree_util.tree_map(
                            lambda x: jax.device_get(x), unfreeze(v)
                        )
                        v = jax.tree_util.tree_map(
                            lambda x: wandb.Histogram(np_histogram=x),
                            v,
                            is_leaf=lambda x: isinstance(x, tuple),
                        )
                        log_metrics[f"{k}/"] = v
                else:
                    if prefix is not None:
                        k = f"{prefix}/{k}"
                    log_metrics[k] = v
            wandb.log({**log_metrics, **self.state_dict})