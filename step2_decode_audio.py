"""
Step 2: Decode Audio at 16 kHz
================================
Decodes raw audio bytes/paths from the dataset using soundfile.
No torchcodec / FFmpeg dependency.
Outputs: DatasetDict with added columns  audio_array  and  sampling_rate.
"""

import io
import numpy as np
import soundfile as sf
from datasets import DatasetDict


TARGET_SR = 16_000


def decode_audio(example: dict) -> dict:
    audio_field = example["audio"]
    raw_bytes = audio_field.get("bytes")
    if raw_bytes is not None and len(raw_bytes) > 0:
        arr, sr = sf.read(io.BytesIO(raw_bytes),
                          dtype="float32", always_2d=False)
    elif audio_field.get("path"):
        arr, sr = sf.read(audio_field["path"],
                          dtype="float32", always_2d=False)
    else:
        raise ValueError(
            f"Audio entry has neither bytes nor a valid path: {audio_field}")


def decode_dataset(dataset: DatasetDict) -> tuple:
    """
    Applies decode_audio to every split.

    Returns
    -------
    dataset : DatasetDict  (with audio_array + sampling_rate columns added)
    sample  : dict         first training example (for quick inspection)
    arr     : np.ndarray   float32 waveform of the first training example
    """
    print("\n" + "=" * 60)
    print("STEP 2: Decoding audio with soundfile (16 kHz)")
    print("=" * 60)

    for split in ("train", "validation", "test"):
        print(f"  Decoding {split} ...")
        dataset[split] = dataset[split].map(decode_audio)

    sample = dataset["train"][0]
    arr = np.array(sample["audio_array"], dtype=np.float32)

    print(f"\n  audio shape   : {arr.shape}")
    print(f"  sampling rate : {sample['sampling_rate']} Hz  OK")
    print(f"  ground truth  : {sample['sentence'][:200]}")

    print(f"\nSplit sizes:")
    for split in dataset:
        print(f"  {split:12s}: {len(dataset[split])} examples")

    return dataset, sample, arr


if __name__ == "__main__":
    from step1_load_dataset import load_primock_med
    dataset = load_primock_med()
    dataset, sample, arr = decode_dataset(dataset)
