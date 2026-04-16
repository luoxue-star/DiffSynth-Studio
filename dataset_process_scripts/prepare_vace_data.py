import os
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Create VACE dataset metadata CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with processed videos (e.g. demo/processed).")
    parser.add_argument("--output_csv", type=str, required=True, help="Output metadata CSV file path.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the relative paths. If omitted, uses current directory.")
    args = parser.parse_args()

    # Collect all subdirectories first so we can show a progress bar
    all_dirs = []
    for root, dirs, files in os.walk(args.input_dir):
        all_dirs.append((root, files))

    data = []
    skipped = 0
    base = args.base_dir if args.base_dir else os.getcwd()

    for root, files in tqdm(all_dirs, desc="Scanning folders"):
        # Check required video/image files
        missing = [f for f in ("orig.mp4", "src.mp4", "ref_image.png") if f not in files]
        if missing:
            if any(f in files for f in ("orig.mp4", "src.mp4", "ref_image.png")):
                # Only report if the folder looks like a partial data folder
                skipped += 1
                tqdm.write(f"  [SKIP] {root}  -- missing: {', '.join(missing)}")
            continue

        # Check for mask
        mask_file = "mask.mp4" if "mask.mp4" in files else None

        # Try to read prompt
        prompt_file = f"{os.path.basename(root)}.txt"
        if prompt_file in files:
            with open(os.path.join(root, prompt_file), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        elif "prompt.txt" in files:
            with open(os.path.join(root, "prompt.txt"), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            skipped += 1
            tqdm.write(f"  [SKIP] {root}  -- missing prompt file (no {prompt_file} or prompt.txt)")
            continue

        row = {
            "video": os.path.relpath(os.path.join(root, "orig.mp4"), base),
            "vace_video": os.path.relpath(os.path.join(root, "src.mp4"), base),
            "vace_reference_image": os.path.relpath(os.path.join(root, "ref_image.png"), base),
            "prompt": prompt
        }
        if mask_file:
            row["vace_video_mask"] = os.path.relpath(os.path.join(root, mask_file), base)

        data.append(row)

    print(f"\nProcessed {len(all_dirs)} folders: {len(data)} valid, {skipped} skipped.")

    if not data:
        print(f"No valid video folders found in {args.input_dir}.")
        return

    df = pd.DataFrame(data)
    
    # Ensure columns are ordered consistently
    cols = ["video", "vace_video"]
    if "vace_video_mask" in df.columns:
        cols.append("vace_video_mask")
    cols.extend(["vace_reference_image", "prompt"])
    
    df = df[cols]
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Generated {args.output_csv} with {len(df)} entries.")
    
    # Print the first row as an example
    print("\nExample entry:")
    for k, v in df.iloc[0].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
