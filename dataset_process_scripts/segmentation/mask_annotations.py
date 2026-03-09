# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Standalone preprocessing script for the swap_anything task,
# extracted from vace_preproccess.py.

import os
import random
import cv2
import torch
import copy
import time
import numpy as np
import argparse
from transformers import Sam3Processor, Sam3Model, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from utils.utils import read_image, read_mask, read_video_frames, save_one_video, save_one_image, convert_to_numpy
from utils.sam3 import detect_bboxes_using_sam3, track_bboxes_using_sam3
from utils.maskaug import MaskAugAnnotator


def parse_bboxes(s):
    bboxes = []
    for bbox_str in s.split():
        coords = list(map(float, bbox_str.split(',')))
        if len(coords) != 4:
            raise ValueError(f"The bounding box requires 4 values, but the input is {len(coords)}.")
        bboxes.append(coords)
    return bboxes


def get_parser():
    parser = argparse.ArgumentParser(
        description="Swap Anything preprocessing for VACE"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="The path of the video to be processed.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="The specific mode of the task, such as firstframe, mask, bboxtrack, label...")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="The path of the mask image.")
    parser.add_argument(
        "--bbox",
        type=parse_bboxes,
        default=None,
        help="Bounding box: four numbers separated by commas (x1,y1,x2,y2). Multiple boxes separated by spaces.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Labels to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Caption to be processed.")
    parser.add_argument(
        "--ref_crop",
        action="store_true",
        default=True,
        help="Crop reference image to subject bounding box (default: True).")
    parser.add_argument(
        "--maskaug_mode",
        type=str,
        default=None,
        help="Mask augmentation mode, e.g. original, original_expand, hull, hull_expand, bbox, bbox_expand.")
    parser.add_argument(
        "--maskaug_ratio",
        type=float,
        default=None,
        help="Ratio of mask augmentation.")
    parser.add_argument(
        "--pre_save_dir",
        type=str,
        default=None,
        help="The directory to save processed data.")
    parser.add_argument(
        "--save_fps",
        type=int,
        default=16,
        help="FPS for saved video output.")
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


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args

    video_path = args.video
    mask_path = args.mask
    bbox = args.bbox
    caption = args.caption
    label = args.label
    save_fps = args.save_fps
    ref_crop = args.ref_crop

    device = f'cuda:{os.getenv("RANK", 0)}'

    # init processor, detect model, track model and mask augment annotator
    detect_processor = Sam3Processor.from_pretrained("../../checkpoints/sam3")
    detector = Sam3Model.from_pretrained("../../checkpoints/sam3").to(device)
    track_processor = Sam3TrackerVideoProcessor.from_pretrained("../../checkpoints/sam3")
    tracker = Sam3TrackerVideoModel.from_pretrained("../../checkpoints/sam3").to(device, dtype=torch.bfloat16)
    maskaug_annotator = MaskAugAnnotator(cfg={})

    mask_cfg = None
    if args.maskaug_mode is not None:
        if args.maskaug_ratio is not None:
            mask_cfg = {"mode": args.maskaug_mode, "kwargs": {'expand_ratio': args.maskaug_ratio, 'expand_iters': 5}}
        else:
            mask_cfg = {"mode": args.maskaug_mode}

    assert video_path is not None, "Please set --video"
    if os.path.isdir(video_path):
        from pathlib import Path
        video_paths = [str(p) for p in Path(video_path).rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']]
    else:
        video_paths = [video_path]

    for v_path in video_paths:
        print(f"Processing video: {v_path}")
        assert args.pre_save_dir is not None, "Please set --pre_save_dir"
        base_name = os.path.basename(v_path).split('.')[0]
        pre_save_dir = os.path.join(args.pre_save_dir, label, base_name)
        if os.path.exists(os.path.join(pre_save_dir, "ref_image.png")):
            print(f"Video {v_path} has been processed, skip.")
            continue
        # ----- Step 1: Video inpainting (detect + mask subject in video) -----
        fps = None
        frames, fps, width, height, num_frames = read_video_frames(v_path.split(",")[0], use_type='cv2', info=True)
        if frames is None:
            print(f"Video read error for {v_path}")
            continue

        inp_kwargs = dict(path=frames, model=detector, processor=detect_processor)
        if label is not None:
            inp_kwargs['prompt'] = label.split(',')

        try:
            print("Processing video inpainting...")
            boxes, scores, frames = detect_bboxes_using_sam3(**inp_kwargs)
            valid_idx, valid_bboxes, mask_list = track_bboxes_using_sam3(frames, boxes, device, tracker, track_processor)
            inp_results = maskaug_annotator.apply_seg_mask(mask_list, copy.deepcopy(frames), mask_color=(128, 128, 128), mask_cfg=mask_cfg)
        except Exception as e:
            print(f"Video inpainting error for {v_path}: {e}")
            continue

        # ----- Step 2: Generate reference image (random frame, masked + cropped) -----
        ref_img = None
        if len(mask_list) > 0:
            # Determine valid frame indices (frames that have at least one object mask)
            all_frame_indices = set()
            for obj_id, masks in mask_list.items():
                for fidx in range(len(masks)):
                    all_frame_indices.add(fidx)
            all_frame_indices = sorted(all_frame_indices)

            if len(all_frame_indices) > 0:
                # Randomly select a frame index
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

        # ----- Output -----
        cur_save_fps = fps if fps is not None else save_fps
        os.makedirs(pre_save_dir, exist_ok=True)

        # save original raw video
        orig_save_path = os.path.join(pre_save_dir, f'orig.mp4')
        save_one_video(orig_save_path, frames, fps=cur_save_fps)
        print(f"Save original video result to {orig_save_path}")

        # save inpainted frames
        save_path = os.path.join(pre_save_dir, f'src.mp4')
        save_one_video(save_path, inp_results[0], fps=cur_save_fps)
        print(f"Save frames result to {save_path}")

        # save masks
        save_path = os.path.join(pre_save_dir, f'mask.mp4')
        save_one_video(save_path, inp_results[1], fps=cur_save_fps)
        print(f"Save masks result to {save_path}")

        # save tracked masks (merging multiple objects if they exist)
        tracked_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
        for masks in mask_list.values():
            for i, m in enumerate(masks[:num_frames]):
                tracked_masks[i] |= m
        tracked_save_path = os.path.join(pre_save_dir, f'src_tracked_mask.mp4')
        save_one_video(tracked_save_path, tracked_masks, fps=cur_save_fps)
        print(f"Save tracked masks result to {tracked_save_path}")

        # save reference images
        if ref_img is not None:
            save_path = os.path.join(pre_save_dir, f'ref_image.png')
            save_one_image(save_path, ref_img, use_type='pil')
            print(f"Save reference image to {save_path}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
