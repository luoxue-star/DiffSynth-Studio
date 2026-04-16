import argparse
import pandas as pd
import os


def extract_category(video_path):
    """Extract category from video path.

    Path structure: .../{category}/{video_id}/orig.mp4
    Category is the parent directory of the video folder.
    """
    parts = video_path.replace("\\", "/").split("/")
    # parts[-1] = "orig.mp4", parts[-2] = video_id, parts[-3] = category
    if len(parts) >= 3:
        return parts[-3]
    return None


def split_train_val(file_path, val_per_category, seed, output_dir):
    print(f"Loading metadata from {file_path}...")
    df = pd.read_csv(file_path)

    df['_category'] = df['video'].apply(extract_category)

    categories = sorted(df['_category'].dropna().unique())
    print(f"Found {len(categories)} categories: {categories}")

    val_indices = []
    for cat in categories:
        cat_df = df[df['_category'] == cat]
        n = min(val_per_category, len(cat_df))
        sampled = cat_df.sample(n=n, random_state=seed)
        val_indices.extend(sampled.index.tolist())
        print(f"  {cat}: {len(cat_df)} total, {n} selected for val")

    val_df = df.loc[val_indices].drop(columns=['_category'])
    train_df = df.drop(index=val_indices).drop(columns=['_category'])

    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "metadata_train.csv")
    val_path = os.path.join(output_dir, "metadata_val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nTrain: {len(train_df)} rows -> {train_path}")
    print(f"Val:   {len(val_df)} rows -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split metadata CSV into train/val sets by category.")
    parser.add_argument("--file_path", type=str, default="demo/metadata_vace.csv", help="Path to metadata CSV file.")
    parser.add_argument("--val_per_category", type=int, default=20, help="Number of val samples per category.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to same dir as input CSV.")
    args = parser.parse_args()
    split_train_val(args.file_path, args.val_per_category, args.seed, args.output_dir)
