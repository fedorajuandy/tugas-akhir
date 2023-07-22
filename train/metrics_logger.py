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
""" Logging """
# pylint: disable=line-too-long

import os
import sys
import time
import jax
import wandb
from flax.core.frozen_dict import unfreeze
from transformers import HfArgumentParser
from .arguments import ModelArguments, DataTrainingArguments, TrainingArguments

parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_json_file(
    json_file=os.path.abspath(sys.argv[1])
)

class MetricsLogger:
    """ For WANDB """
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
            f'train/{k.rsplit("_", maxsplit=1)[-1]}': state[k]
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
        """ time """
        if jax.process_index() == 0:
            wandb.log({f"time/{key}": duration, **self.state_dict})
        if offset:
            self.offset_time += duration

    def log(self, metrics, prefix=None):
        """ metrics """
        if jax.process_index() == 0:
            log_metrics = {}
            for k, v in metrics.items(): # pylint: disable=invalid-name
                if "_norm" in k:
                    if self.step % training_args.log_norm_steps == 0:
                        log_metrics[f"{k}/"] = unfreeze(v)
                elif "_hist" in k:
                    if self.step % training_args.log_histogram_steps == 0:
                        v = jax.tree_util.tree_map( # pylint: disable=invalid-name
                            lambda x: jax.device_get(x), unfreeze(v)
                        )
                        v = jax.tree_util.tree_map( # pylint: disable=invalid-name
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
