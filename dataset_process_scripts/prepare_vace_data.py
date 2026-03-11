import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create VACE dataset metadata CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with processed videos (e.g. demo/processed).")
    parser.add_argument("--output_csv", type=str, required=True, help="Output metadata CSV file path.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the relative paths. If omitted, uses current directory.")
    args = parser.parse_args()

    data = []
    
    # Traverse input directory to find video folders
    for root, dirs, files in os.walk(args.input_dir):
        # A valid folder should contain these files
        if "orig.mp4" in files and "src.mp4" in files and "ref_image.png" in files:
            # Check for mask: prefer src_tracked_mask.mp4 over mask.mp4
            mask_file = "src_tracked_mask.mp4" if "src_tracked_mask.mp4" in files else "mask.mp4"
            if mask_file not in files:
                mask_file = None
                
            # Try to read prompt from prompt.txt, or default to folder name
            prompt = os.path.basename(root)
            prompt_file = f"{os.path.basename(root)}.txt"
            if prompt_file in files:
                with open(os.path.join(root, prompt_file), "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            elif "prompt.txt" in files:
                with open(os.path.join(root, "prompt.txt"), "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            
            # Use relative paths from the base_dir
            base = args.base_dir if args.base_dir else os.getcwd()
            
            row = {
                "video": os.path.relpath(os.path.join(root, "orig.mp4"), base),
                "vace_video": os.path.relpath(os.path.join(root, "src.mp4"), base),
                "vace_reference_image": os.path.relpath(os.path.join(root, "ref_image.png"), base),
                "prompt": prompt
            }
            if mask_file:
                row["vace_video_mask"] = os.path.relpath(os.path.join(root, mask_file), base)
            
            data.append(row)

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
