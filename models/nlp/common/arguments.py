""" Since arguments are duplicated in run_pretraining.py and sagemaker_pretraining.py, they have
been abstracted into this file. It also makes the training scripts much shorter.
"""

import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_dir: str = field(default=None, metadata={"help": "Unused, but passed by SageMaker"})
    model_type: str = field(default="albert", metadata={"choices": ["albert", "bert"]})
    model_size: str = field(default="base", metadata={"choices": ["base", "large"]})
    # TODO: Change this to per_gpu_train_batch_size
    batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    max_seq_length: int = field(default=512, metadata={"choices": [128, 512]})
    max_predictions_per_seq: int = field(default=20, metadata={"choices": [20, 80]})
    warmup_steps: int = field(default=3125)
    total_steps: int = field(default=125000)
    learning_rate: float = field(default=0.00176)
    end_learning_rate: float = field(default=3e-5)
    learning_rate_decay_power: float = field(default=1.0)
    hidden_dropout_prob: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    optimizer: str = field(default="lamb", metadata={"choices": ["lamb", "adam"]})
    name: str = field(default="", metadata={"help": "Additional info to append to metadata"})
    log_frequency: int = field(default=1000)
    load_from: str = field(
        default="scratch", metadata={"choices": ["scratch", "checkpoint", "huggingface"]}
    )
    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "For example, `/fsx/checkpoints/albert/2020..step125000`. No .ckpt on the end."
        },
    )
    fsx_prefix: str = field(
        default="/fsx",
        metadata={
            "choices": ["/fsx", "/opt/ml/input/data/training"],
            "help": "Change to /opt/ml/input/data/training on SageMaker",
        },
    )
    skip_xla: str = field(default=None, metadata={"choices": ["true"]})
    eager: str = field(default=None, metadata={"choices": ["true"]})
    skip_sop: str = field(default=None, metadata={"choices": ["true"]})
    skip_mlm: str = field(default=None, metadata={"choices": ["true"]})
    pre_layer_norm: str = field(
        default=None,
        metadata={
            "choices": ["true"],
            "help": "Place layer normalization before the attention & FFN, rather than after adding the residual connection. https://openreview.net/pdf?id=B1x8anVFPr",
        },
    )
    extra_squad_steps: str = field(default=None)
    fast_squad: str = field(default=None, metadata={"choices": ["true"]})
    dummy_eval: str = field(default=None, metadata={"choices": ["true"]})
    seed: int = field(default=42)

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


def populate_squad_parser(parser: argparse.ArgumentParser) -> None:
    # Model loading
    parser.add_argument("--model_type", default="albert", choices=["albert", "bert"])
    parser.add_argument("--model_size", default="base", choices=["base", "large"])
    parser.add_argument("--load_from", required=True)
    parser.add_argument("--load_step", type=int)
    parser.add_argument("--skip_xla", choices=["true"])
    parser.add_argument("--eager", choices=["true"])
    parser.add_argument(
        "--pre_layer_norm",
        choices=["true"],
        help="See https://github.com/huggingface/transformers/pull/3929",
    )
    parser.add_argument(
        "--fsx_prefix",
        default="/fsx",
        choices=["/fsx", "/opt/ml/input/data/training"],
        help="Change to /opt/ml/input/data/training on SageMaker",
    )
    # Hyperparameters from https://arxiv.org/pdf/1909.11942.pdf#page=17
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--total_steps", default=8144, type=int)
    parser.add_argument("--warmup_steps", default=814, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--dataset", default="squadv2")
    parser.add_argument("--seed", type=int, default=42)
    # Logging information
    parser.add_argument("--name", default="default")
    parser.add_argument("--validate_frequency", default=1000, type=int)
    parser.add_argument("--checkpoint_frequency", default=500, type=int)
    parser.add_argument("--model_dir", help="Unused, but passed by SageMaker")


def populate_sagemaker_parser(parser: argparse.ArgumentParser) -> None:
    # SageMaker parameters
    parser.add_argument(
        "--source_dir",
        help="For example, /Users/myusername/Desktop/deep-learning-models/models/nlp/albert",
    )
    parser.add_argument("--entry_point", default="run_pretraining.py")
    parser.add_argument("--role", default=os.environ["SAGEMAKER_ROLE"])
    parser.add_argument("--image_name", default=os.environ["SAGEMAKER_IMAGE_NAME"])
    parser.add_argument("--fsx_id", default=os.environ["SAGEMAKER_FSX_ID"])
    parser.add_argument(
        "--subnet_ids", help="Comma-separated string", default=os.environ["SAGEMAKER_SUBNET_IDS"]
    )
    parser.add_argument(
        "--security_group_ids",
        help="Comma-separated string",
        default=os.environ["SAGEMAKER_SECURITY_GROUP_IDS"],
    )
    # Instance specs
    parser.add_argument(
        "--instance_type",
        type=str,
        default="ml.p3dn.24xlarge",
        choices=["ml.p3dn.24xlarge", "ml.p3.16xlarge", "ml.g4dn.12xlarge"],
    )
    parser.add_argument("--instance_count", type=int, default=1)
