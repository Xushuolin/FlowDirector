import argparse
import inspect
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an object mask video with GroundingDINO boxes and optional SAM masks.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Input video path.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output binary mask video path. White pixels mark the object/removal region.")
    parser.add_argument("--text_prompt", type=str, required=True,
                        help="Object text prompt for detection, e.g. 'large brown bear'.")
    parser.add_argument("--backend", type=str, default="sam", choices=["sam", "box"],
                        help="Use SAM masks from GroundingDINO boxes, or save box masks only.")
    parser.add_argument("--grounding_model", type=str, default="IDEA-Research/grounding-dino-base",
                        help="Hugging Face GroundingDINO model id.")
    parser.add_argument("--sam_model", type=str, default="facebook/sam-vit-base",
                        help="Hugging Face SAM model id used when --backend sam.")
    parser.add_argument("--box_threshold", type=float, default=0.25,
                        help="GroundingDINO box confidence threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="GroundingDINO text confidence threshold.")
    parser.add_argument("--max_boxes", type=int, default=3,
                        help="Maximum detected boxes to segment per frame.")
    parser.add_argument("--dilate", type=int, default=9,
                        help="Optional dilation kernel size for expanding the final mask. Use 0 to disable.")
    parser.add_argument("--blur", type=int, default=0,
                        help="Optional Gaussian blur kernel size for soft mask edges. Use 0 to disable.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device for open-source detection/segmentation models.")
    return parser.parse_args()


def _normalize_grounding_prompt(text):
    text = text.strip()
    if not text.endswith("."):
        text = text + "."
    return text


def _open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, frame_count


def _open_writer(output_path, fps, width, height):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise ValueError(f"Cannot open output mask video for writing: {output_path}")
    return writer


def _detect_boxes(image, prompt, processor, model, device, box_threshold, text_threshold, max_boxes):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post_process = processor.post_process_grounded_object_detection
    post_process_params = inspect.signature(post_process).parameters
    threshold_kwargs = {
        "text_threshold": text_threshold,
        "target_sizes": [image.size[::-1]],
    }
    if "box_threshold" in post_process_params:
        threshold_kwargs["box_threshold"] = box_threshold
    else:
        threshold_kwargs["threshold"] = box_threshold

    results = post_process(
        outputs,
        inputs.input_ids,
        **threshold_kwargs)[0]
    boxes = results["boxes"]
    scores = results.get("scores")
    if boxes.numel() == 0:
        return boxes.cpu()
    if scores is not None:
        order = torch.argsort(scores, descending=True)[:max_boxes]
        boxes = boxes[order]
    else:
        boxes = boxes[:max_boxes]
    return boxes.detach().cpu()


def _box_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes.tolist():
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _sam_mask(image, boxes, sam_processor, sam_model, device, width, height):
    if boxes.numel() == 0:
        return np.zeros((height, width), dtype=np.uint8)

    input_boxes = [boxes.tolist()]
    inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]
    iou_scores = outputs.iou_scores.detach().cpu()[0]
    best_mask_ids = iou_scores.argmax(dim=-1)
    box_ids = torch.arange(masks.shape[0])
    best_masks = masks[box_ids, best_mask_ids]
    combined = best_masks.any(dim=0).numpy().astype(np.uint8) * 255
    return combined


def _postprocess_mask(mask, dilate, blur):
    if dilate and dilate > 0:
        kernel_size = dilate if dilate % 2 == 1 else dilate + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur and blur > 0:
        kernel_size = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return mask


def main():
    args = _parse_args()
    device = torch.device(args.device)
    prompt = _normalize_grounding_prompt(args.text_prompt)

    grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model).to(device).eval()

    sam_processor = None
    sam_model = None
    if args.backend == "sam":
        sam_processor = SamProcessor.from_pretrained(args.sam_model)
        sam_model = SamModel.from_pretrained(args.sam_model).to(device).eval()

    cap, fps, width, height, frame_count = _open_video(args.video_path)
    writer = _open_writer(args.output_path, fps, width, height)

    progress = tqdm(total=frame_count if frame_count > 0 else None, desc="Generating masks")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        boxes = _detect_boxes(
            image,
            prompt,
            grounding_processor,
            grounding_model,
            device,
            args.box_threshold,
            args.text_threshold,
            args.max_boxes)
        if args.backend == "sam":
            mask = _sam_mask(image, boxes, sam_processor, sam_model, device, width, height)
        else:
            mask = _box_mask(boxes, width, height)
        mask = _postprocess_mask(mask, args.dilate, args.blur)
        writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        progress.update(1)

    progress.close()
    cap.release()
    writer.release()
    print(f"Saved mask video to {args.output_path}")


if __name__ == "__main__":
    main()
