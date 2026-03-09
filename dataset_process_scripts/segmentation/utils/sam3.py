import os
import numpy as np
import torch
from typing import List, Union
from transformers import Sam3Processor, Sam3Model, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from PIL import Image
import imageio.v2 as imageio


def load_data(path: str):
	if path.endswith(".mp4"):
		video = imageio.get_reader(path)
		frames = []
		for frame in video:
			frames.append(frame)
		video.close()
	elif path.endswith(".jpg") or path.endswith(".png"):
		frame = imageio.imread(path)
		frames = [frame]
	else:
		if os.path.isdir(path):
			frames = [imageio.imread(os.path.join(path, f)) for f in sorted(os.listdir(path))]
		else:
			raise ValueError(f"Unsupported file type: {path}")
	return np.stack(frames, axis=0)


def detect_bboxes_using_sam3(
	path: Union[str, List[np.ndarray]], 
	prompt: Union[str, List[str]], 
	device: str = "cuda",
	model: Sam3Model = None,
	processor: Sam3Processor = None,
):
	"""
	Detect the bboxes of first frame using SAM3
	Args:
		path: str, path to the video or image
		prompt: str or list of str, prompt to detect the bboxes
		device: str, device to use
	Returns:
		bbox: np.ndarray, detected bbox
		frames: np.ndarray, All frames of the video/image/directory
	"""
	if model is None:
		model = Sam3Model.from_pretrained("facebook/sam3").to(device)
	if processor is None:
		processor = Sam3Processor.from_pretrained("facebook/sam3")

	if isinstance(prompt, str):
		text_prompts = [prompt]
	else:
		text_prompts = list(prompt)

	if isinstance(path, str):
		frames = load_data(path)
	else:
		frames = path
	inputs = processor(images=[Image.fromarray(frames[0])], text=text_prompts, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	results = processor.post_process_instance_segmentation(
			outputs,
			threshold=0.5,
			mask_threshold=0.5,
			target_sizes=inputs.get("original_sizes").tolist()
	)
	scores = results[0]['scores'].cpu().numpy()
	boxes = results[0]['boxes'].cpu().numpy()
	return boxes, scores, frames


def track_bboxes_using_sam3(video_frames: Union[np.ndarray, List[np.ndarray], str],
							boxes: np.ndarray = None,
							device: str = "cuda",
							tracker: Sam3TrackerVideoModel = None,
							processor: Sam3TrackerVideoProcessor = None,
							):
	"""
	Track the bboxes of the video using SAM3
	Args:
		boxes: np.ndarray, detected bboxes
		video_frames: np.ndarray, video frames
		device: str, device to use
	Returns:
		valid_idx: list of int, indices of valid frames
		valid_bboxes: list of np.ndarray, detected bboxes
		mask_list: list of np.ndarray, detected masks
	"""
	# 1. Initialize the tracker
	if tracker is None:
		tracker = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
	if processor is None:
		processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

	num_boxes = len(boxes)
	if num_boxes == 0:
		raise ValueError("No bounding boxes detected on the first frame for the given prompt.")
	obj_ids = list(range(num_boxes))

	# 2. Initialize the tracker and track the bboxes of the video
	inference_session = processor.init_video_session(
						video=video_frames,
						inference_device=device,
						dtype=torch.bfloat16,
					)
	processor.add_inputs_to_inference_session(
		inference_session=inference_session,
		frame_idx=0,
		obj_ids=obj_ids,
		# SAM3 expects boxes in shape [image, boxes, coords]; keep single image batch.
		input_boxes=[boxes.tolist()],
		# input_labels=[1 for _ in range(len(boxes))],
	)

	valid_idx, valid_bboxes, mask_list = {i: [] for i in range(len(boxes))}, {i: [] for i in range(len(boxes))}, {i: [] for i in range(len(boxes))}
	for sam3_tracker_video_output in tracker.propagate_in_video_iterator(
		inference_session,
		start_frame_idx=0,  # first conditioning frame already populated via add_inputs_to_inference_session
	):
		video_res_masks = processor.post_process_masks(
			[sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
		)[0]

		video_res_masks = (video_res_masks.float().cpu().numpy() > 0.0).astype(np.uint8) * 255
		for i, mask in enumerate(video_res_masks):
			y, x = np.where(mask[0] > 0)
			if len(y) > 0 and len(x) > 0:
				boxes = np.array([x.min(), y.min(), x.max(), y.max()])
				valid_idx[i].append(sam3_tracker_video_output.frame_idx)
				valid_bboxes[i].append(boxes)
				mask_list[i].append(mask[0])
			else:
				continue

	return valid_idx, valid_bboxes, mask_list


if __name__ == "__main__":
	valid_bboxes, detected_bboxes, mask_list = track_bboxes("model.mp4", [60, 0, 1380, 1440])
	video_frames = [np.zeros_like(mask) for mask in mask_list]
	video_frames = load_data("model.mp4")
	for i, bbox in enumerate(detected_bboxes):
		x_min, y_min, x_max, y_max = bbox
		video_frames[i][y_min: y_max, x_min: x_max, :] = 255

	import cv2
	def write_video(frames, video_path, fps=30):
		height, width, _ = frames[0].shape
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
		for frame in frames:
			video.write(frame[:, :, ::-1])
		video.release()

	# write_video(video_frames, "model_mask.mp4", fps=24)
	imageio.mimsave("model_mask.mp4", video_frames, fps=24)

