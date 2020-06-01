"""
Since arguments are duplicated in run_pretraining.py and sagemaker_pretraining.py, they have
been abstracted into this file. It also makes the training scripts much shorter.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    ModelArguments is the subset of arguments relating to the model instantiation.
    So config options such as dropout fall under this, but skip_xla does not because it is
    used at training time.
    """

    model_type: str = field(default="albert", metadata={"choices": ["albert", "bert"]})
    model_size: str = field(default="base", metadata={"choices": ["base", "large"]})
    load_from: str = field(
        default="scratch", metadata={"choices": ["scratch", "checkpoint", "huggingface"]}
    )
    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "For example, `/fsx/checkpoints/albert/2020..step125000`. No .ckpt on the end."
        },
    )
    pre_layer_norm: str = field(
        default=None,
        metadata={
            "choices": ["true"],
            "help": "Place layer normalization before the attention & FFN, rather than after adding the residual connection. https://openreview.net/pdf?id=B1x8anVFPr",
        },
    )
    hidden_dropout_prob: float = field(default=0.0)
    attention_probs_dropout_prob: float = field(default=0.0)

    @property
    def model_desc(self) -> str:
        if self.model_type == "albert":
            return f"albert-{self.model_size}-v2"
        elif self.model_type == "bert":
            return f"bert-{self.model_size}-uncased"
        else:
            assert False


@dataclass
class DataTrainingArguments:
    task_name: str = field(default="squadv2", metadata={"choices": ["squadv1", "squadv2"]})
    max_seq_length: int = field(default=512, metadata={"choices": [128, 512]})
    max_predictions_per_seq: int = field(default=20, metadata={"choices": [20, 80]})
    fsx_prefix: str = field(
        default="/fsx",
        metadata={
            "choices": ["/fsx", "/opt/ml/input/data/training"],
            "help": "Change to /opt/ml/input/data/training on SageMaker",
        },
    )


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # model_dir: str = field(default=None, metadata={"help": "Unused, but passed by SageMaker"})
    seed: int = field(default=42)
    # TODO: Change this to per_gpu_train_per_gpu_batch_size
    per_gpu_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    warmup_steps: int = field(default=3125)
    total_steps: int = field(default=125000)
    optimizer: str = field(default="lamb", metadata={"choices": ["lamb", "adamw", "adam"]})
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
class LoggingArguments:
    log_frequency: int = field(default=1000)
    validation_frequency: int = field(default=2000)
    checkpoint_frequency: int = field(default=5000)
    squad_frequency: int = field(default=40000)
    fast_squad: str = field(default=None, metadata={"choices": ["true"]})
    dummy_eval: str = field(default=None, metadata={"choices": ["true"]})


@dataclass
class SageMakerArguments:
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
        self.subnet_ids = self.subnet_ids or os.environ["SAGEMAKER_SUBNET_IDS"]
        self.security_group_ids = (
            self.security_group_ids or os.environ["SAGEMAKER_SECURITY_GROUP_IDS"]
        )

        self.subnet_ids = self.subnet_ids.replace(" ", "").split(",")
        self.security_group_ids = self.security_group_ids.replace(" ", "").split(",")
