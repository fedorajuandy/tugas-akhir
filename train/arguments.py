""" Training, model, and data arguments """
# pylint: disable=line-too-long

from dataclasses import dataclass, field
import tempfile
import os
from pathlib import Path
from typing import Optional
import jax
import wandb


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # CONFIG STUFF
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Don't set if you want to train a model from scratch. "
            "W&B artifact references are supported in addition to the sources supported by `PreTrainedModel`."
        },
    )
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
            "help": "Floating-point format in which the computations will be performed (not the model weights). Choose one of `[float32, float16, bfloat16]`."
            # How much memory will be used (common -> less memory but sacrifices precision -> middle of the two)
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Restore optimizer and training state. Can be True (will retrieve associated wandb artifact), a local directory or a Google bucket path."
            # save optimiser and training state or not (not = train from scratch)
        },
    )
    dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout rate. Overwrites config."},
        # Prevent overfitting (too complex, memorise rather than understand #human'shere) -> more on Kaggle ML lesson
    )
    activation_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Activation dropout rate. Overwrites config."},
        # layer's output (not neuron), prevent overfitting
    )
    attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Attention dropout rate. Overwrites config."},
        # attention mechanicsm in transformers (relation between tokens in seq to learn contextual representation), prevent overfitting
    )

    # TOKENIZER
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
            assert (
                self.tokenizer_name is not None
            ), "Tokenizer name or model name/path needs to be specified"
        if self.restore_state:
            assert self.model_name_or_path is not None and (
                "/model-" in self.model_name_or_path
            ), "Restoring state only available with W&B artifact reference"

    # PRE-TRAINED MODEL [DELETE LATER]
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

    # PROB DELETE LATER TOO
    def get_opt_state(self):
        """ get state """

        # immediately deleted after execution
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if self.restore_state is True:
                # wandb artifact
                state_artifact = self.model_name_or_path.replace(
                    "/model-", "/state-", 1
                )
                # if first jax process
                if jax.process_index() == 0:
                    artifact = wandb.run.use_artifact(state_artifact)
                else:
                    artifact = wandb.Api().artifact(state_artifact)
                if artifact.metadata.get("bucket_path"):
                    # we will read directly file contents
                    self.restore_state = artifact.metadata["bucket_path"]
                else:
                    artifact_dir = artifact.download(tmp_dir)
                    self.restore_state = str(Path(artifact_dir) / "opt_state.msgpack")

            with Path(self.restore_state).open("rb") as f:
                return f.read()
                # return optimiser state from those args in binary (bytes)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
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
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file (glob & braceexpand acceptable)."
        },
    )
    # data loading should not be a bottleneck so we use "streaming" mode by default
    streaming: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to stream the dataset."},
        # read and processed in small chunks incrementally; slowly into memory
    )
    # DELETE LATER
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use the authentication token for private datasets."
        },
    )
    shard_by_host: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to shard data files by host in multi-host environments."
            # spread data into multiple devices; ah, yes, good ol teamwork that does not exist between humans- (distribution and synchronisation)
        },
    )
    # DELETE LATER (?) using managed dataset
    blank_caption_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of removing some captions for classifier-free guidance."
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
    # DELETE LATER
    multi_eval_ds: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to look for multiple validation datasets (local support only)."
        },
    )
    # DELETE LATER
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples."
        },
    )
    # DELETE LATER
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples."
        },
    )
    # DELETE LATER
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing. Not used in streaming mode."
        },
    )
    # DELETE LATER
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets. Not used in streaming mode."
        },
    )
    # default seed of None ensures we don't repeat the same items if script was interrupted during an epoch
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
    Arguments pertaining to training parameters.
    """

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    # DELETE LATER
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the validation set."}
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per data parallel device for training."},
    )
    # DELETE LATER
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size per data parallel device for evaluation. Same as training batch size if not set."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing an update pass."
        },
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing."}
        # reduce memory usage (selectively recomputing intermediate actvations during backward pass (computing loss then update model); time++)
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate."}
    )
    optim: str = field(
        default="distributed_shampoo",
        metadata={
            "help": 'The optimizer to use. Can be "distributed_shampoo" (default), "adam" or "adafactor"'
            # for learning rate and update; based on task, arch, or experiments
            # shampoo: for distributed, based on Hessioan matrix (second order partial derivaties; those H(f) from high scholl)
            # adam: Adaptive Moment Estimation = update weights each iteration
            # adafactor: adaptive learning (per parameter), address Adam's limitation (sensitive: wrong val = fail, biased, cannot use large scale distribution, suboptimal solution)
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay applied to parameters."}
        # regularisation, prevent overfitting (reduce weigthts'magnitude)
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
        # hyperparameters; exponential decay rates (each x epochs, learning rate reduces by y) for first moment (gradient moving  average)
        # Hessian and its inverse
        # smaller value = slower (more stable)
        # bigger = faster but can fluctuate and overshoot the optimal parameter value and swing back (like running too fast and past the intended point, need to go back and end up going back to far)
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
        # second moment (squared gradient average)
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
        # small value added to denominator so it would not be divided by zsero; default = common
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for Adafactor."}
        # max value prevent instabilities (too large, too small)
    )
    block_size: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
        # reduce memory req
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
        # steps = each iteration of optimization (more steps, more optimization; depending on dataset, model, criteria)
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": "The type of grafting to use. Can be 'rmsprop_normalized' (default), 'rmsprop', 'adagrad', 'adagrad_normalized', 'sgd' or 'sqrt_n'"
        },
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Use Nesterov momentum for Distributed Shampoo."},
        # improves convergence
    )
    optim_quantized: bool = field(
        default=True,
        metadata={
            "help": "Whether to quantize optimizer (only supported with Distributed Shampoo)."
            # shard optimizer across devices
        },
    )
    shard_shampoo_across: str = field(
        default="dp",
        metadata={
            "help": "Whether to shard the optimizer across data devices (dp), model devices (mp) or both (2d)."
        },
    )

    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
        # an epoch = complete iteration (takes input, produces output, backward pass) until dataset is complete
    )

    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
        # Gradually increases lr (very small, avoid negative impact to performance) until max val over certain number of steps/epochs [said 5-20% of training steps]
    )
    lr_decay: str = field(
        default=None,
        metadata={
            "help": "Decay to be used in the learning rate scheduler. Can be None (default), linear or exponential."
            # linear is reduces at a fixed number, exponential using some factor (yk, the usual)
            # exponential more sensitive, yet more effective
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": "Number of transition steps associated with learning rate decay when using exponential decay."
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={
            "help": "Decay rate associated with learning rate when using exponential decay."
            # the point of decay occurs (nth step)
        },
    )
    lr_staircase: bool = field(
        default=False,
        metadata={
            "help": "Whether to use staircase or continuous learning rate when using exponential decay."
            # continuous: each each epoch/nth step
            # staircase: discrete steps (determinded my decay rate; can be at 10 epochs or sth)
        },
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
        # keep lr at 0 at the beginning (if has many parameteres; uncertain to initialise weights to obtain optimal perfrom; model explore parameter without large changes at the beginning then lr increses slowly)
    )
    logging_steps: int = field(
        default=40, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=400, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=1, metadata={"help": "Save checkpoint every X updates steps."}
    )
    log_model: bool = field(
        default=True,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )
    log_norm_steps: int = field(
        default=True,
        metadata={"help": "Log parameters and gradients norm at this frequency."},
    )
    log_histogram_steps: int = field(
        default=False,
        metadata={
            "help": "Log parameters and gradients histograms at this frequency. Slows down training."
        },
    )

    seed_model: int = field(
        default=42,
        # 42 is the answer to life, universe, and everything~ #jk
        metadata={
            "help": "Random seed for the model that will be set at the beginning of training."
            # small, constant to be reproducible
            # initialise weights and break symetry, avoid stuck in local minima (loss func raches local min val)
            # same one in diff exp/model resulting in biased result
        },
    )

    embeddings_only: bool = field(
        default=False, metadata={"help": "Train only embedding layers."}
        # map encoded word (high dimensional sparse vector; where the val is 1 in corresponding word and 0 in others) to word embedding (low dimensional dense vector; represents the attributes) -> more in report
        # to capture the meaning and context of words (semantic relationship; hyponym etc., seek high school stuff)
        # not including hidden and output; useful in transfer learning (fine tuned on specific task)
    )
    init_embeddings: bool = field(
        default=False,
        metadata={"help": "When training embedding layers, initialize them."},
        # weights initialised before training; using pret-trained is more common
    )

    # DELETE LATER
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity to use (for teams)."},
    )
    # CHANGE LATER, DELETE LATER
    wandb_project: str = field(
        default="dalle-mini",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="Seq2Seq",
        metadata={"help": "The name of the wandb job type."},
    )

    # DELETE LATER
    assert_tpu_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    use_vmap_trick: bool = field(
        default=True,
        metadata={"help": "EDITED: DL framework to apply same operation to many at the same time for parallel processing"},
    )

    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of devices required for model parallelism. The other dimension of available devices is used for data parallelism."
        },
    )

    dp_devices: int = field(init=False)

    def __post_init__(self):
        # DELETE LATER
        if self.assert_tpu_available:
            assert (
                jax.local_device_count() == 8
            ), "TPUs in use, please check running processes"

        assert self.optim in [
            "distributed_shampoo",
            "adam",
            "adafactor",
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
        ), "Number of devices for model parallelism must be > 0"

        assert (
            jax.device_count() % self.mp_devices == 0
        ), f"Number of available devices ({jax.device_count()} must be divisible by number of devices used for model parallelism ({self.mp_devices})."
        self.dp_devices = jax.device_count() // self.mp_devices
