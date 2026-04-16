# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Preprocessing script for videos that already have pre-computed masks.
# Applies mask augmentation and saves all components in the same format
# as mask_annotations.py, but skips SAM3 detection/tracking.

"""
python dataset_process_scripts/segmentation/mask_annotations_from_existing.py \
       --video_dir demo/processed/backup/ \
       --maskaug_mode "hull_expand:0.4,bbox_expand:0.1,dilate:0.5" \
       --maskaug_ratio 0.2
"""

import os
import random
import cv2
import copy
import numpy as np
import argparse

from utils.utils import read_video_frames, save_one_video, save_one_image
from utils.maskaug import MaskAugAnnotator


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process videos with existing masks: apply mask augmentation and save all components."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Root directory containing per-video subdirectories with orig.mp4 and src_tracked_mask.mp4.")
    parser.add_argument(
        "--ref_crop",
        action="store_true",
        default=True,
        help="Crop reference image to subject bounding box (default: True).")
    parser.add_argument(
        "--maskaug_mode",
        type=str,
        default=None,
        help="Mask augmentation mode, e.g. original, original_expand, hull, hull_expand, bbox, bbox_expand, dilate. "
             "You can also specify probabilities, e.g. original:0.8,bbox:0.2.")
    parser.add_argument(
        "--maskaug_ratio",
        type=float,
        default=None,
        help="Ratio of mask augmentation.")
    parser.add_argument(
        "--maskaug_iters",
        type=int,
        default=7,
        help="Number of dilation iterations for mask augmentation (default: 5).")
    parser.add_argument(
        "--save_fps",
        type=int,
        default=16,
        help="FPS for saved video output.")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index (inclusive) for subdirectory processing slice.")
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (exclusive) for subdirectory processing slice.")
    return parser


def remove_background(image_np, mask):
    """Apply mask to remove background: set background to white, crop to subject."""
    mask_uint8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_uint8, 1, 255, cv2.THRESH_BINARY)

    # Set background pixels to white
    out_image = image_np.copy()
    out_image[binary_mask == 0] = 255

    return out_image, binary_mask


def crop_to_subject(image_np, mask):
    """Crop image to the subject bounding box."""
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if binary_mask is None or binary_mask.size == 0 or cv2.countNonZero(binary_mask) == 0:
        return image_np, mask
    x, y, w, h = cv2.boundingRect(binary_mask)
    return image_np[y:y+h, x:x+w], mask[y:y+h, x:x+w]


def load_mask_frames(mask_video_path):
    """Load mask video and return list of binary mask frames (uint8, single-channel)."""
    frames, fps, width, height, num_frames = read_video_frames(
        mask_video_path, use_type='cv2', is_rgb=False, info=True
    )
    if frames is None:
        return None, fps, width, height, 0

    mask_frames = []
    for frame in frames:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # Binarize: anything > 0 becomes 255
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_frames.append(binary.astype(np.uint8))

    return mask_frames, fps, width, height, num_frames


def build_mask_list_from_frames(mask_frames):
    """
    Build a mask_list dict from loaded mask frames.
    The mask_list format matches SAM3 track output: {obj_id: [mask_per_frame, ...]}.
    Since we have a merged binary mask, we treat it as a single object (obj_id=0).
    """
    mask_list = {0: mask_frames}
    return mask_list


def parse_maskaug_config(args):
    """Parse mask augmentation configuration from args."""
    mask_cfg = None
    if args.maskaug_mode is not None:
        modes = []
        weights = []
        for item in args.maskaug_mode.split(","):
            if ":" in item:
                mode, weight = item.split(":")
                modes.append(mode)
                weights.append(float(weight))
            else:
                modes.append(item)
                weights.append(1.0)
        maskaug_mode = random.choices(modes, weights=weights, k=1)[0]
        if args.maskaug_ratio is not None:
            mask_cfg = {"mode": maskaug_mode, "kwargs": {'expand_ratio': args.maskaug_ratio, 'expand_iters': args.maskaug_iters}}
        else:
            mask_cfg = {"mode": maskaug_mode}
    return mask_cfg


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args

    video_dir = args.video_dir
    save_fps = args.save_fps
    ref_crop = args.ref_crop

    maskaug_annotator = MaskAugAnnotator(cfg={})
    mask_cfg = parse_maskaug_config(args)

    # Discover all per-video subdirectories
    if not os.path.isdir(video_dir):
        print(f"Error: {video_dir} is not a directory.")
        return

    subdirs = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])

    if not subdirs:
        print(f"No subdirectories found in {video_dir}")
        return

    total = len(subdirs)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total
    subdirs = subdirs[start_idx:end_idx]
    print(f"Total subdirectories: {total}, processing [{start_idx}:{end_idx}] ({len(subdirs)} items)")

    for subdir in subdirs:
        pre_save_dir = os.path.join(video_dir, subdir)
        orig_path = os.path.join(pre_save_dir, "orig.mp4")
        tracked_mask_path = os.path.join(pre_save_dir, "src_tracked_mask.mp4")

        # Validate required files exist
        if not os.path.exists(orig_path):
            print(f"Skipping {subdir}: orig.mp4 not found.")
            continue
        if not os.path.exists(tracked_mask_path):
            print(f"Skipping {subdir}: src_tracked_mask.mp4 not found.")
            continue

        # Check if already fully processed (ref_image.png exists)
        if os.path.exists(os.path.join(pre_save_dir, "ref_image.png")):
            print(f"Skipping {subdir}: already processed (ref_image.png exists).")
            continue

        print(f"Processing: {subdir}")

        # ----- Step 1: Load original video frames -----
        frames, fps, width, height, num_frames = read_video_frames(
            orig_path, use_type='cv2', info=True
        )
        if frames is None:
            print(f"  Error reading orig.mp4 for {subdir}")
            continue

        # ----- Step 2: Load pre-existing tracked masks -----
        mask_frames, _, _, _, mask_num_frames = load_mask_frames(tracked_mask_path)
        if mask_frames is None:
            print(f"  Error reading src_tracked_mask.mp4 for {subdir}")
            continue

        # Ensure frame counts match (truncate to minimum)
        min_frames = min(num_frames, mask_num_frames)
        frames = frames[:min_frames]
        mask_frames = mask_frames[:min_frames]

        # Build mask_list in the same format as SAM3 tracking output
        mask_list = build_mask_list_from_frames(mask_frames)

        # ----- Step 3: Apply mask augmentation -----
        try:
            inp_results = maskaug_annotator.apply_seg_mask(
                mask_list, copy.deepcopy(frames),
                mask_color=(128, 128, 128), mask_cfg=mask_cfg
            )
        except Exception as e:
            print(f"  Mask augmentation error for {subdir}: {e}")
            continue

        # ----- Step 4: Generate reference image (random frame, masked + cropped) -----
        ref_img = None
        if len(mask_list) > 0:
            all_frame_indices = list(range(min_frames))

            if len(all_frame_indices) > 0:
                sel_idx = random.choice(all_frame_indices)
                sel_frame = frames[sel_idx].copy()

                # Build a union mask across all objects for the selected frame
                h, w = sel_frame.shape[:2]
                union_mask = np.zeros((h, w), dtype=np.uint8)
                for obj_id, masks in mask_list.items():
                    if sel_idx < len(masks):
                        obj_mask = masks[sel_idx]
                        union_mask = np.maximum(union_mask, obj_mask)

                # Remove background (set to white) and crop to union bounding box
                ref_img, ref_binary_mask = remove_background(sel_frame, union_mask)
                if ref_crop:
                    ref_img, _ = crop_to_subject(ref_img, ref_binary_mask)

        # ----- Step 5: Save all components -----
        cur_save_fps = fps if fps is not None else save_fps

        # save original raw video (already exists, but re-save for consistency)
        orig_save_path = os.path.join(pre_save_dir, 'orig.mp4')
        save_one_video(orig_save_path, frames, fps=cur_save_fps)
        print(f"  Saved original video to {orig_save_path}")

        # save inpainted frames (src.mp4)
        save_path = os.path.join(pre_save_dir, 'src.mp4')
        save_one_video(save_path, inp_results[0], fps=cur_save_fps)
        print(f"  Saved src frames to {save_path}")

        # save masks (mask.mp4)
        save_path = os.path.join(pre_save_dir, 'mask.mp4')
        save_one_video(save_path, inp_results[1], fps=cur_save_fps)
        print(f"  Saved masks to {save_path}")

        # save tracked masks (src_tracked_mask.mp4)
        tracked_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(min_frames)]
        for masks in mask_list.values():
            for i, m in enumerate(masks[:min_frames]):
                tracked_masks[i] |= m
        tracked_save_path = os.path.join(pre_save_dir, 'src_tracked_mask.mp4')
        save_one_video(tracked_save_path, tracked_masks, fps=cur_save_fps)
        print(f"  Saved tracked masks to {tracked_save_path}")

        # save reference image
        if ref_img is not None:
            save_path = os.path.join(pre_save_dir, 'ref_image.png')
            save_one_image(save_path, ref_img, use_type='pil')
            print(f"  Saved reference image to {save_path}")

        print(f"  Done processing {subdir}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
