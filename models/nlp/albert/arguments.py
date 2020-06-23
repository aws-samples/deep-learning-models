""" Contains many default training hyperparameters specific to the model.

While many of the fields are the same, the default hyperparameters correspond to a single model.
For example, 125k steps and 0.00176 learning rate refers to ALBERT pretraining. This could lead to
users improperly training ELECTRA if they forget to set all the hyperparameters correctly.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
