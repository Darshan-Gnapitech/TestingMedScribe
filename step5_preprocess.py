"""
Step 5: Preprocess the Dataset
================================
Converts raw audio arrays + corrected transcripts into model-ready tensors.

Input columns expected from Step 2:
    audio      : dict {"array": np.float32 array, "sampling_rate": int}
    transcript : str  (doctor-corrected ground truth)

Output columns (replaces all raw columns):
    input_features : log-Mel spectrogram, shape (80, 3000)
    labels         : token ids, padding replaced with -100
"""

from datasets import DatasetDict
from transformers import WhisperProcessor


class MedicalWhisperPreprocessor:
    """
    Converts raw audio arrays + corrected transcripts into model-ready tensors.

    Input columns expected from Member 1:
        audio      : dict {"array": np.float32 array, "sampling_rate": int}
        transcript : str  (doctor-corrected ground truth)

    Output columns (replaces all raw columns):
        input_features : log-Mel spectrogram, shape (80, 3000)
        labels         : token ids, padding replaced with -100
    """

    TARGET_SR        = 16_000
    MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        processor: WhisperProcessor,
        audio_column: str = "audio_array",
        transcript_column: str = "sentence",
    ):
        self.processor         = processor
        self.audio_column      = audio_column
        self.transcript_column = transcript_column

    def _audio_to_features(self, batch):
        audio_arrays = [a if isinstance(a, list) else a["array"] for a in batch[self.audio_column]]
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        batch["input_features"] = inputs.input_features
        return batch

    def _tokenize_labels(self, batch):
        labels = self.processor.tokenizer(
            batch[self.transcript_column],
            max_length=self.MAX_LABEL_TOKENS,
            truncation=True,
            padding="max_length",
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        batch["labels"] = [
            [(t if t != pad_id else -100) for t in seq] for seq in labels
        ]
        batch["raw_text"] = batch[self.transcript_column]
        return batch

    def __call__(self, dataset_dict: DatasetDict, batch_size: int = 8) -> DatasetDict:
        print("[Preprocess] Extracting log-Mel features ...")
        dataset_dict = dataset_dict.map(
            self._audio_to_features, batched=True,
            batch_size=batch_size, desc="Audio -> log-Mel"
        )

        print("[Preprocess] Tokenizing transcripts ...")
        dataset_dict = dataset_dict.map(
            self._tokenize_labels, batched=True,
            batch_size=batch_size, desc="Text -> tokens"
        )

        for split in dataset_dict:
            drop = [
            c for c in dataset_dict[split].column_names
            if c not in {
                "input_features",
                "labels",
                "raw_text",
                }
            ]
            dataset_dict[split] = dataset_dict[split].remove_columns(drop)

        print("[Preprocess] Done.")
        for split, ds in dataset_dict.items():
            print(f"  {split:12s}: {len(ds):,} examples")
        return dataset_dict


def preprocess_dataset(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    skip_preprocessing: bool = False,
) -> DatasetDict:
    """
    Convenience wrapper used by main.py and prepare_for_member3.

    skip_preprocessing=True  -> dataset already has input_features + labels
    skip_preprocessing=False -> dataset has raw audio + transcript (production)
    """
    if skip_preprocessing:
        print("\n-- Step 5: Skipped (data already pre-processed) --")
        return dataset

    print("\n-- Step 5: Preprocessing dataset --")
    return MedicalWhisperPreprocessor(processor)(dataset)


if __name__ == "__main__":
    import numpy as np
    from datasets import Dataset, DatasetDict
    from transformers import WhisperProcessor

    MODEL_NAME  = "openai/whisper-small"
    SAMPLE_TEXT = "Patient was prescribed 500 mg amoxicillin twice daily."
    SR          = 16_000

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="English", task="transcribe"
    )

    def make_split(n):
        silence = np.zeros(SR * 3, dtype=np.float32)
        feats = processor.feature_extractor(
            [silence] * n, sampling_rate=SR,
            return_tensors="np", padding="max_length", truncation=True,
        ).input_features
        tok = processor.tokenizer(
            [SAMPLE_TEXT] * n, max_length=448, truncation=True, padding="max_length"
        ).input_ids
        pad_id = processor.tokenizer.pad_token_id
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in tok]
        return {"input_features": [feats[i] for i in range(n)], "labels": labels}

    dataset = DatasetDict({
        "train"     : Dataset.from_dict(make_split(6)),
        "validation": Dataset.from_dict(make_split(3)),
        "test"      : Dataset.from_dict(make_split(3)),
    })

    processed = preprocess_dataset(dataset, processor, skip_preprocessing=True)
    print("Step 5 standalone test complete.")
    print(f"  columns: {processed['train'].column_names}")