import collections
import math
from typing import Dict, List

import tensorflow as tf
import tqdm
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    SquadResult,
    SquadV2Processor,
)

from common.utils import get_dataset


def get_evaluation_metrics(
    model,
    tokenizer,
    data_dir: str,
    filename: str,
    per_gpu_batch_size: int = 32,
    num_batches: int = None,
    disable_tqdm: bool = False,
) -> Dict[str, "Number"]:
    """
    Return an OrderedDict in the format:
    {
    'exact': 0.8169797018445212,
    'f1': 4.4469722448269335,
    'total': 11873,
    'HasAns_exact': 0.15182186234817813,
    'HasAns_f1': 7.422216845956518,
    'HasAns_total': 5928,
    'NoAns_exact': 1.4802354920100924,
    'NoAns_f1': 1.4802354920100924,
    'NoAns_total': 5945,
    'best_exact': 50.07159100480081,
    'best_exact_thresh': 0.0,
    'best_f1': 50.0772059855695,
    'best_f1_thresh': 0.0
    }
    """
    # These are not used in inference, only for scoring in `compute_predictions_logits()`.
    processor = SquadV2Processor()
    examples: List[SquadExample] = processor.get_dev_examples(data_dir, filename=filename)
    features: List[SquadFeatures] = get_dataset(
        tokenizer=tokenizer,
        processor=processor,
        data_dir=data_dir,
        filename=filename,
        per_gpu_batch_size=per_gpu_batch_size,
        shard=False,
        shuffle=False,
        drop_remainder=False,
        return_raw_features=True,
    )

    # Here we get the dataset instead of just the features, with return_raw_features=False.
    dataset: tf.data.Dataset = get_dataset(
        tokenizer=tokenizer,
        processor=processor,
        data_dir=data_dir,
        filename=filename,
        per_gpu_batch_size=per_gpu_batch_size,
        shard=False,
        shuffle=False,
        drop_remainder=False,
        return_raw_features=False,
    )
    results: List[SquadResult] = get_squad_results(
        model=model,
        dataset=dataset,
        features=features,
        per_gpu_batch_size=per_gpu_batch_size,
        num_batches=num_batches,
        disable_tqdm=disable_tqdm,
    )

    write_prediction_files = False
    if write_prediction_files:
        output_predictions_file = f"/fsx/{args.checkpoint}_predictions.json"
        output_nbest_file = f"/fsx/{args.checkpoint}_nbest_predictions.json"
        output_null_log_odds_file = f"/fsx/{args.checkpoint}_null_odds.json"
    else:
        output_predictions_file = None
        output_nbest_file = None
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=results,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file=output_predictions_file,
        output_nbest_file=output_nbest_file,
        output_null_log_odds_file=output_null_log_odds_file,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer,
    )

    results: collections.OrderedDict = squad_evaluate(examples, predictions)
    return results


def get_squad_results(
    model,
    dataset: tf.data.Dataset,
    features: List[SquadFeatures],
    per_gpu_batch_size: int,
    num_batches: int,
    disable_tqdm: bool,
) -> List[SquadResult]:
    results = []

    total_steps = math.ceil(len(features) / per_gpu_batch_size)
    pbar = tqdm.tqdm(total=total_steps, disable=disable_tqdm)
    pbar.set_description(f"Evaluating with batch size {per_gpu_batch_size}")

    if num_batches:
        dataset = dataset.take(num_batches)

    for step, batch in enumerate(dataset):
        input_dict = {
            "input_ids": batch[0]["input_ids"],
            "attention_mask": batch[0]["attention_mask"],
            "token_type_ids": batch[0]["token_type_ids"],
        }
        outputs = model(input_dict, training=False)
        start_logits, end_logits = outputs[0], outputs[1]

        per_gpu_batch_size = len(batch[1]["start_positions"])
        for i in range(per_gpu_batch_size):
            feature_index = batch[0]["feature_index"][i].numpy().item()
            unique_id = int(features[feature_index].unique_id)
            result = SquadResult(
                unique_id=unique_id,
                start_logits=start_logits[i].numpy().tolist(),
                end_logits=end_logits[i].numpy().tolist(),
            )
            results.append(result)

        pbar.update(1)
    pbar.close()

    return results
