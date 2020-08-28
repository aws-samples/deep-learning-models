"""
Since arguments are duplicated in run_pretraining.py and launch_sagemaker.py, they have
been abstracted into this file. It also makes the training scripts much shorter.

Using `transformers.HfArgumentParser` we can turn these classes into argparse arguments.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class PathArguments:
    train_dir: str = field(metadata={"help": "A folder containing TFRecords"})
    val_dir: str = field(metadata={"help": "A folder containing TFRecords"})

    filesystem_prefix: str = field(
        default="/fsx", metadata={"help": "Change to '/opt/ml/input/data/training' on SageMaker",},
    )
    log_dir: str = field(
        default="logs/default", metadata={"help": "For example, 'logs/albert' or 'logs/squad'"},
    )
    checkpoint_dir: str = field(
        default="checkpoints/default",
        metadata={"help": "For example, 'checkpoints/albert' or 'checkpoints/squad'"},
    )


@dataclass
class TrainingArguments:
    """ Related to the training loop. """

    seed: int = field(default=42)
    # TODO: Change this to per_gpu_train_batch_size
    per_gpu_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    warmup_steps: int = field(default=3125)
    total_steps: int = field(default=125000)
    optimizer: str = field(default="lamb", metadata={"choices": ["lamb", "adamw"]})
    learning_rate: float = field(default=0.00176)
    weight_decay: float = field(default=0.01)
    end_learning_rate: float = field(default=3e-5)
    learning_rate_decay_power: float = field(default=1.0)
    beta_1: float = field(default=0.9)
    beta_2: float = field(default=0.999)
    epsilon: float = field(default=1e-6)

    max_grad_norm: float = field(default=1.0)
    name: str = field(default="", metadata={"help": "Additional info to append to metadata"})

    skip_xla: str = field(default=None, metadata={"choices": ["true"]})
    eager: str = field(default=None, metadata={"choices": ["true"]})
    skip_sop: str = field(default=None, metadata={"choices": ["true"]})
    skip_mlm: str = field(default=None, metadata={"choices": ["true"]})

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


@dataclass
class ModelArguments:
    """
    ModelArguments is the subset of arguments relating to the model instantiation.
    So config options such as dropout fall under this, but skip_xla does not because it is
    used at training time.
    """

    model_dir: str = field(default=None, metadata={"help": "Unused, but passed by SageMaker"})
    model_type: str = field(default="albert", metadata={"choices": ["albert", "bert", "electra"]})
    model_size: str = field(default="base", metadata={"choices": ["small", "base", "large"]})
    load_from: str = field(
        default="scratch", metadata={"choices": ["scratch", "checkpoint", "huggingface"]}
    )
    # TODO: Move this to PathArguments?
    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "For example, `/fsx/checkpoints/albert/2020..step125000`. No .ckpt on the end."
        },
    )
    load_optimizer_state: str = field(default="true", metadata={"choices": ["true", "false"]})
    hidden_dropout_prob: float = field(default=0.0)
    attention_probs_dropout_prob: float = field(default=0.0)
    electra_tie_weights: str = field(default="true", metadata={"choices": ["true", "false"]})
    # TODO: Pre-layer norm is not yet supported in transformers. PR is at https://github.com/huggingface/transformers/pull/3929, but maintainers are unresponsive.
    # The difficulty of keeping a parallel fork means we'll disable this option temporarily.
    pre_layer_norm: str = field(
        default=None,
        metadata={
            "choices": [],
            "help": "Place layer normalization before the attention & FFN, rather than after adding the residual connection. https://openreview.net/pdf?id=B1x8anVFPr",
        },
    )

    @property
    def model_desc(self) -> str:
        if self.model_type == "albert":
            return f"albert-{self.model_size}-v2"
        elif self.model_type == "bert":
            return f"bert-{self.model_size}-uncased"
        elif self.model_type == "electra":
            # assert False, "Not yet supported since there are two ELECTRA models"
            return f"google/electra-{self.model_size}-discriminator"
        else:
            assert False


@dataclass
class DataTrainingArguments:
    """ Arguments related to the dataset preparation.

    Task name, sequence length, and filepath fall under this category, but batch size does not.
    """

    squad_version: str = field(default="squadv2", metadata={"choices": ["squadv1", "squadv2"]})
    # For BERT/ALBERT the only valid combos are [512,20] and [128,80]
    # For ELECTRA we use dynamic masking, so all combos are valid
    max_seq_length: int = field(default=512)
    max_predictions_per_seq: int = field(default=20)


@dataclass
class LoggingArguments:
    """ Related to validation and finetuning evaluation.

    This can have a significant impact on runtime (squad_frequency), so it seems a tad disingenuous
    to call them logging arguments. Maybe change later.
    """

    log_frequency: int = field(default=10)
    validation_frequency: int = field(default=2000)
    checkpoint_frequency: int = field(default=5000)
    evaluate_frequency: int = field(default=5000)
    run_name: str = field(
        default=None,
        metadata={
            "help": "Name of saved checkpoints and logs during training. For example, bert-phase-1."
        },
    )

    # TODO: Remove these since they're a little too specific
    squad_frequency: int = field(default=0)
    fast_squad: str = field(default=None, metadata={"choices": ["true"]})
    dummy_eval: str = field(default=None, metadata={"choices": ["true"]})


@dataclass
class SageMakerArguments:
    """ Related to SageMaker infrastructure, unused on EC2 or EKS. """

    source_dir: str = field(
        metadata={
            "help": "For example, /Users/myusername/Desktop/deep-learning-models/models/nlp/albert"
        }
    )
    entry_point: str = field(metadata={"help": "For example, run_pretraining.py or run_squad.py"})
    sm_job_name: str = field(default="albert")

    role: str = field(default=None)
    image_name: str = field(default=None)
    fsx_id: str = field(default=None)
    fsx_mount_name: str = field(default="fsx")
    subnet_ids: str = field(default=None, metadata={"help": "Comma-separated string"})
    security_group_ids: str = field(default=None, metadata={"help": "Comma-separated string"})
    instance_type: str = field(
        default="ml.p3dn.24xlarge",
        metadata={"choices": ["ml.p3dn.24xlarge", "ml.p3.16xlarge", "ml.g4dn.12xlarge"]},
    )
    instance_count: int = field(default=1)

    def __post_init__(self):
        # Dataclass are evaluated at import-time, so we need to wrap these in a post-init method
        # in case the env-vars don't exist.
        self.role = self.role or os.environ["SAGEMAKER_ROLE"]
        self.image_name = self.image_name or os.environ["SAGEMAKER_IMAGE_NAME"]
        self.fsx_id = self.fsx_id or os.environ["SAGEMAKER_FSX_ID"]
        self.fsx_mount_name = self.fsx_mount_name or os.environ["SAGEMAKER_FSX_MOUNT_NAME"]
        self.subnet_ids = self.subnet_ids or os.environ["SAGEMAKER_SUBNET_IDS"]
        self.security_group_ids = (
            self.security_group_ids or os.environ["SAGEMAKER_SECURITY_GROUP_IDS"]
        )

        self.subnet_ids = self.subnet_ids.replace(" ", "").split(",")
        self.security_group_ids = self.security_group_ids.replace(" ", "").split(",")
