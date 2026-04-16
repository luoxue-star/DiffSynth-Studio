# -*- coding: utf-8 -*-
"""
Remove generated files so that mask_annotations_from_existing.py
will re-process every video subdirectory.

Files removed per subdirectory:
  - ref_image.png  (skip-condition checked by the script)
  - src.mp4        (generated inpainted source)
  - mask.mp4       (generated mask video)

Files kept:
  - orig.mp4              (required input)
  - src_tracked_mask.mp4  (required input)
  - *.txt                 (metadata, not touched)

Usage:
    python dataset_process_scripts/segmentation/clean_for_reprocess.py \
        --data_dir /path/to/VACE
"""

import os
import argparse

FILES_TO_REMOVE = ["ref_image.png", "src.mp4", "mask.mp4"]


def clean(data_dir, dry_run=False):
    removed = 0
    # data_dir contains category dirs (shoes, human, …),
    # each category dir contains per-video subdirs.
    for category in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        for video_name in sorted(os.listdir(category_path)):
            video_path = os.path.join(category_path, video_name)
            if not os.path.isdir(video_path):
                continue
            for fname in FILES_TO_REMOVE:
                fpath = os.path.join(video_path, fname)
                if os.path.exists(fpath):
                    if dry_run:
                        print(f"[dry-run] would remove: {fpath}")
                    else:
                        os.remove(fpath)
                        print(f"removed: {fpath}")
                    removed += 1

    action = "would remove" if dry_run else "removed"
    print(f"\nDone. {action} {removed} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean generated files for reprocessing.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root VACE data directory (contains category subdirs).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be deleted, without actually deleting.")
    args = parser.parse_args()
    clean(args.data_dir, dry_run=args.dry_run)
