"""
Step 1: Load Primock_med Dataset
=================================
Downloads and loads the Na0s/Primock_med dataset from HuggingFace.
Outputs: DatasetDict with train / validation / test splits.
"""

import os
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio


REPO_ID = "Na0s/Primock_med"


def load_primock_med() -> "DatasetDict":
    print("=" * 60)
    print("STEP 1: Loading Na0s/Primock_med from HuggingFace")
    print("=" * 60)

    parquet_files = {}
    for split in ("train", "validation", "test"):
        print(f"  Downloading {split} split ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        parquet_files[split] = path
        print(f"    cached at {path}")

    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False)
    )

    print(f"\nSplits     : {list(dataset.keys())}")
    print(f"  train      : {len(dataset['train'])} examples")
    print(f"  validation : {len(dataset['validation'])} examples")
    print(f"  test       : {len(dataset['test'])} examples")
    print(f"Columns    : {dataset['train'].column_names}")

    row = dataset["train"][0]
    print(f"\nfile_name  : {row['file_name']}")
    print(f"sentence   : {row['sentence'][:200]}")

    return dataset


if __name__ == "__main__":
    dataset = load_primock_med()
