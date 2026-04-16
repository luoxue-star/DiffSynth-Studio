import argparse
import pandas as pd
import sys
import os
import imageio
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.abspath('.'))
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

# File columns that reference video or image files
FILE_COLUMNS = ["video", "vace_video", "vace_video_mask", "vace_reference_image"]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def check_file(filepath):
    """Check if a file exists and can be read correctly.
    Returns None if OK, or an error message string if the file is broken.
    """
    if not os.path.exists(filepath):
        return f"file not found: {filepath}"

    ext = os.path.splitext(filepath)[1].lower()

    if ext in VIDEO_EXTENSIONS:
        try:
            reader = imageio.get_reader(filepath)
            num_frames = reader.count_frames()
            if num_frames <= 0:
                reader.close()
                return f"video has 0 frames: {filepath}"
            reader.get_data(0)
            reader.close()
        except Exception as e:
            return f"video read error: {filepath} ({e})"

    elif ext in IMAGE_EXTENSIONS:
        try:
            Image.open(filepath).convert("RGB")
        except Exception as e:
            return f"image read error: {filepath} ({e})"

    else:
        return f"unsupported file extension '{ext}': {filepath}"

    return None


def _validate_row(args):
    """Validate a single row (designed for multiprocessing).
    Returns (idx, error_reason_or_None).
    """
    idx, row_dict, tokenizer_path = args

    prompt = row_dict['prompt']

    # === Prompt checks ===
    if pd.isna(prompt) or isinstance(prompt, float):
        return (idx, "prompt: NaN/float")
    if not isinstance(prompt, str):
        return (idx, "prompt: non-string type")
    try:
        tokenizer = _get_worker_tokenizer(tokenizer_path)
        tokenizer(prompt)
    except Exception:
        return (idx, "prompt: text encoding failed")

    # === File integrity checks ===
    for col in FILE_COLUMNS:
        if col not in row_dict:
            continue
        filepath = row_dict[col]
        if pd.isna(filepath):
            continue
        error = check_file(filepath)
        if error is not None:
            reason_key = error.split(":")[0]
            return (idx, f"{col}: {reason_key}")

    return (idx, None)


# Per-worker cached tokenizer (avoid re-loading per row)
_worker_tokenizer = None
_worker_tokenizer_path = None


def _get_worker_tokenizer(tokenizer_path):
    global _worker_tokenizer, _worker_tokenizer_path
    if _worker_tokenizer is None or _worker_tokenizer_path != tokenizer_path:
        from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
        _worker_tokenizer = HuggingfaceTokenizer(name=tokenizer_path, clean="canonicalize")
        _worker_tokenizer_path = tokenizer_path
    return _worker_tokenizer


def process_and_clean(file_path, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 32)

    print(f"Loading metadata from {file_path}...")
    df = pd.read_csv(file_path)

    tokenizer_path = "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "google/umt5-xxl"

    # Verify tokenizer can load before spawning workers
    print("Loading text encoder HuggingfaceTokenizer...")
    try:
        _get_worker_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Please ensure you have internet access or the model is downloaded locally.")
        return

    invalid_indices = []
    error_reasons = defaultdict(list)

    # Build task list
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((idx, row.to_dict(), tokenizer_path))

    print(f"Processing {len(df)} rows with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_validate_row, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, reason = future.result()
            if reason is not None:
                invalid_indices.append(idx)
                error_reasons[reason].append(idx)

    # === Report ===
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"Total rows:   {len(df)}")
    print(f"Invalid rows: {len(invalid_indices)}")
    print(f"Valid rows:   {len(df) - len(invalid_indices)}")

    if error_reasons:
        print(f"\nBreakdown by reason:")
        print("-" * 60)
        for reason, indices in sorted(error_reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {reason}: {len(indices)}")
    print("=" * 60)

    if len(invalid_indices) > 0:
        print("\nRemoving the identified erroneous data...")
        df_clean = df.drop(index=invalid_indices)
        df_clean.to_csv(file_path, index=False)
        print(f"Done. Successfully saved cleaned data to {file_path}.")
    else:
        print("\nNo erroneous data found in the current CSV file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean metadata CSV by validating prompts and file integrity.")
    parser.add_argument("--file_path", type=str, default="demo/metadata_vace.csv", help="Path to metadata CSV file.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: cpu_count, max 32).")
    args = parser.parse_args()
    process_and_clean(args.file_path, num_workers=args.num_workers)
