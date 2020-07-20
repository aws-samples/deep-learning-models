import nlp
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

dset_size = 10000
max_seq_length = 512
dset = nlp.Dataset.from_dict(
    {"examples": [[str(i) for i in range(max_seq_length)] for _ in range(dset_size)]}
)

dset = dset.map(
    lambda batch: tokenizer(
        batch["examples"], is_pretokenized=True,  # rather than [ex for ex in batch["examples"]]
    ),
    batched=True,
    remove_columns=["examples"],
)
