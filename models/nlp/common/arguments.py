""" Since arguments are duplicated in run_pretraining.py and sagemaker_pretraining.py, they have
been abstracted into this file. It also makes the training scripts much shorter.
"""

import argparse
import os


def populate_pretraining_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_dir", help="Unused, but passed by SageMaker")
    parser.add_argument("--model_type", default="albert", choices=["albert", "bert"])
    parser.add_argument("--model_size", default="base", choices=["base", "large"])
    parser.add_argument("--batch_size", type=int, default=32, help="per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512, choices=[128, 512])
    parser.add_argument("--max_predictions_per_seq", type=int, default=20, choices=[20, 80])
    parser.add_argument("--warmup_steps", type=int, default=3125)
    parser.add_argument("--total_steps", type=int, default=125000)
    parser.add_argument("--learning_rate", type=float, default=0.00176)
    parser.add_argument("--end_learning_rate", type=float, default=3e-5)
    parser.add_argument("--learning_rate_decay_power", type=float, default=1.0)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", default="lamb", choices=["lamb", "adam"])
    parser.add_argument("--name", default="", help="Additional info to append to metadata")
    parser.add_argument("--log_frequency", type=int, default=1000)
    parser.add_argument(
        "--load_from", default="scratch", choices=["scratch", "checkpoint", "huggingface"],
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="For example, `/fsx/checkpoints/albert/2020..step125000`. No .ckpt on the end.",
    )
    parser.add_argument(
        "--fsx_prefix",
        default="/fsx",
        choices=["/fsx", "/opt/ml/input/data/training"],
        help="Change to /opt/ml/input/data/training on SageMaker",
    )
    # SageMaker does not work with 'store_const' args, since it parses into a dictionary
    # We will treat any value not equal to None as True, and use --skip_xla=true
    parser.add_argument(
        "--skip_xla",
        choices=["true"],
        help="For debugging. Faster startup time, slower runtime, more GPU vRAM.",
    )
    parser.add_argument(
        "--eager",
        choices=["true"],
        help="For debugging. Faster launch, slower runtime, more GPU vRAM.",
    )
    parser.add_argument(
        "--skip_sop", choices=["true"], help="Only use MLM loss, and exclude the SOP loss.",
    )
    parser.add_argument(
        "--skip_mlm", choices=["true"], help="Only use SOP loss, and exclude the MLM loss.",
    )
    parser.add_argument(
        "--pre_layer_norm",
        choices=["true"],
        help="Place layer normalization before the attention & FFN, rather than after adding the residual connection. https://openreview.net/pdf?id=B1x8anVFPr",
    )
    parser.add_argument("--extra_squad_steps", type=str)
    parser.add_argument("--fast_squad", choices=["true"])
    parser.add_argument("--dummy_eval", choices=["true"])
    parser.add_argument("--seed", type=int, default=42)


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
