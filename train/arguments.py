""" Training, model, and data arguments """
# pylint: disable=line-too-long

from dataclasses import dataclass, field
import tempfile
from pathlib import Path
from typing import Optional
import jax
import wandb


@dataclass
class ModelArguments:
    """ Train from zero or from checkpoint """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for weights initialization."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Computations format."
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Restore optimizer and training state."
        },
    )
    dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Rate to prevent overfitting by neuron's output. Overwrites config."},
    )
    activation_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Rate to prevent overfitting by layer's output. Overwrites config."},
    )
    attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Rate to prevent overfitting by transformers. Overwrites config."},
    )


    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
            assert (
                self.tokenizer_name is not None
            ), "tokenizer_name or model_name_or_path needs to be specified."
        if self.restore_state:
            assert self.model_name_or_path is not None and (
                "/model-" in self.model_name_or_path
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
        """ get state """

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
    """ Data's detail for training and evaluation """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "Dataset's column name for caption."
        },
    )
    encoding_column: Optional[str] = field(
        default="encoding",
        metadata={
            "help": "Dataset's column name for image encoding."
        },
    )
    dataset_repo_or_path: str = field(
        default=None,
        metadata={"help": "Dataset's location."},
    )
    streaming: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to stream the dataset to prevent bottleneck."},
    )
    blank_caption_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of removing some captions for classifier-free guidance."
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
    """ Training parameters """

    output_dir: str = field(
        metadata={
            "help": "Directory to store model predictions and checkpoints."
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
            "help": "To reduce memory usage."
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
            "help": "Weight dermscay applied to parameters."
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
            "help": "Maintain moving average of the square of gradients; divide the gradient by the root of the average."
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
            "help": "Quantize optimizer to map infinite values to a smaller set of discrete finite values."
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
            "help": "Staircase (discrete steps) or continuous learning rate (each epoch) when using exponential decay."
        },
    )
    lr_offset: int = field(
        default=0,
        metadata={
            "help": "Number of steps to offset learning rate and keep it at 0."
        },
    )
    save_steps: int = field(
        default=1,
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
        default="dalle-mini",
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
            ), "TPUs in use, please check running processes"

        assert self.optim in [
            "distributed_shampoo",
        ], f"Selected optimizer not supported: {self.optim}"

        assert self.graft_type in [
            "rmsprop_normalized",
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
        ), f"Number of available devices ({jax.device_count()} must be divisible by number of devices used for model parallelism ({self.mp_devices})."

        self.dp_devices = jax.device_count() // self.mp_devices
