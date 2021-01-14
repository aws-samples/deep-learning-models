"""
Launch with
`python -m t5.mini_t5`
"""

import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from transformers import T5ForConditionalGeneration


def gen_synthetic_batch(batch_size: int, sequence_length: int):
    data = np.zeros((batch_size, sequence_length))
    tensor = torch.tensor(data, dtype=torch.int64)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        default="t5-small",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)

    args = parser.parse_args()

    print(
        f"Training {args.model_name} for {args.steps} steps with batch size {args.batch_size} and sequence length {args.sequence_length}"
    )

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_frequency = 10
    start_time = time.perf_counter()
    for i in range(args.steps):
        inputs = gen_synthetic_batch(
            batch_size=args.batch_size, sequence_length=args.sequence_length
        )

        optimizer.zero_grad()

        outputs = model(inputs, decoder_input_ids=inputs)
        last_hidden_states = (
            outputs.encoder_last_hidden_state
        )  # [batch_size, sequence_length, vocab_size]
        loss = torch.mean(last_hidden_states ** 2)
        loss.backward()
        optimizer.step()

        if i % log_frequency == 0:
            elapsed_time = time.perf_counter() - start_time
            it_s = log_frequency / elapsed_time
            start_time = time.perf_counter()
            print(f"Step {i}, Loss: {loss.item():.3f}, It/s: {it_s:.3f}")


if __name__ == "__main__":
    main()
