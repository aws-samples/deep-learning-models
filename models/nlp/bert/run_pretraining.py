"""
This will be a dynamic run




Unit testing requires a few pieces:
- A stable API (functions to call for pretraining & squad).
- Synthetic data.
- Flexibility to change.

"""

from transformers import AlbertConfig, TFAlbertForPreTraining

from common.datasets import (
    gen_model_inputs,
    get_synthetic_mlm_batch,
    get_synthetic_mlm_dataset,
    parse_text,
)


def test_forward_pass():
    parse_text("/fsx/wikitext/wikitext-2-raw/wiki.test.raw")

    dataset = get_synthetic_mlm_dataset(per_gpu_batch_size=2)
    config = AlbertConfig.from_pretrained("albert-base-v2")
    model = TFAlbertForPreTraining(config)
    batch = get_synthetic_mlm_batch(2)
    inputs = next(gen_model_inputs(2))
    outputs = model(inputs)


if __name__ == "__main__":
    test_forward_pass()
