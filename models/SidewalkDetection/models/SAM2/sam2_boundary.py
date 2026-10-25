"""SAM 2.1 Small boundary detection, independent of SegFormer.

Automatic prompts assume a sidewalk enters the lower middle of the photograph.
SAM is class agnostic: geometric selection is a heuristic, not semantic proof.
Boundary coordinates are (x, y) in the original raster, sorted by increasing y.
"""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

MODEL_ID = "facebook/sam2.1-hiera-small"
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "weights"


def _image(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert("RGB")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


@lru_cache(maxsize=1)
def _load_model():
    from transformers import Sam2Config, Sam2Model, Sam2Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
    # The checkpoint also contains video metadata; retain its image parameters.
    config_data, _ = Sam2Config.get_config_dict(MODEL_PATH, local_files_only=True)
    config_data.update(model_type="sam2", architectures=["Sam2Model"])
    config = Sam2Config.from_dict(config_data)
    model = Sam2Model.from_pretrained(MODEL_PATH, config=config, local_files_only=True).to(device).eval()
    return processor, model


def automatic_prompts(width, height):
    """Three independent hypotheses, with foreground and background points."""
    groups = []
    for center in (0.5, 0.35, 0.65):
        normalized = [(center, 0.85), (center, 0.68),
                      (0.02, 0.65), (0.98, 0.65), (0.5, 0.08)]
        groups.append([[round(x * (width - 1)), round(y * (height - 1))]
                       for x, y in normalized])
    return np.array(groups, dtype=np.int32), np.array([[1, 1, 0, 0, 0]] * 3)


def _seed_component(mask, seed):
    """Discard disconnected objects; never relabel another region as sidewalk."""
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    index = labels[seed[1], seed[0]]
    return labels == index if index else np.zeros_like(mask, dtype=bool)


def _select_mask(masks, scores, points):
    height, width = masks.shape[-2:]
    best, best_score = np.zeros((height, width), dtype=bool), -np.inf
    for group, group_scores, prompts in zip(masks, scores, points):
        for mask, quality in zip(group, group_scores):
            component = _seed_component(mask, prompts[0])
            ys, xs = np.nonzero(component)
            area = len(xs) / (height * width)
            if not 0.005 <= area <= 0.8 or float(quality) < 0.5:
                continue
            span = (ys.max() - ys.min() + 1) / height
            if span < 0.25 or not component[prompts[1, 1], prompts[1, 0]]:
                continue
            # Prefer vertically extended regions, penalize whole-frame spill.
            edge_fraction = np.mean(component[:, [0, -1]])
            score = float(quality) + 0.2 * span - 0.3 * edge_fraction
            if score > best_score:
                best, best_score = component, score
    return best.astype(np.uint8)


@torch.inference_mode()
def segment_sidewalk(image):
    """Return an original-size 0/1 mask, or an empty mask if no candidate passes."""
    image = _image(image)
    points, labels = automatic_prompts(*image.size)
    processor, model = _load_model()
    inputs = processor(images=image, input_points=[points.tolist()],
                       input_labels=[labels.tolist()], return_tensors="pt").to(model.device)
    outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(),
                                        inputs["original_sizes"].cpu())[0].numpy()
    scores = outputs.iou_scores[0].cpu().numpy()
    return _select_mask(masks, scores, points)


def extract_boundaries(mask):
    """Outer left/right pixel on each occupied row; holes are not side boundaries.

    Missing rows are omitted. No extrapolation or smoothing fabricates points.
    Empty masks return two arrays with shape (0, 2).
    """
    mask = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(mask.any(axis=1))
    left_x = mask[rows].argmax(axis=1)
    right_x = mask.shape[1] - 1 - mask[rows, ::-1].argmax(axis=1)
    return (np.column_stack((left_x, rows)).astype(np.int32),
            np.column_stack((right_x, rows)).astype(np.int32))


def detect_boundaries(image):
    """Return (left_boundary, right_boundary), each an int32 (N, 2) array."""
    return extract_boundaries(segment_sidewalk(image))


def save_boundaries(image, output_dir=None):
    """Save boundaries.png, left_boundary.npy and right_boundary.npy."""
    image = _image(image)
    mask = segment_sidewalk(image)
    left, right = extract_boundaries(mask)
    directory = Path(output_dir) if output_dir is not None else PROJECT_DIR / "output"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "left_boundary.npy", left, allow_pickle=False)
    np.save(directory / "right_boundary.npy", right, allow_pickle=False)
    Image.fromarray(mask * 255).save(directory / "sidewalk_mask.png")
    drawing = image.copy()
    pen = ImageDraw.Draw(drawing)
    for boundary, color in [(left, "red"), (right, "cyan")]:
        # Break at missing rows rather than drawing across an undetected gap.
        for section in np.split(boundary, np.flatnonzero(np.diff(boundary[:, 1]) > 1) + 1):
            if len(section) >= 2:
                pen.line([tuple(point) for point in section.tolist()], fill=color, width=3)
    pen.text((10, 10), "LEFT: red | RIGHT: cyan" if len(left) else "No candidate found", fill="yellow")
    drawing.save(directory / "boundaries.png")
    return left, right


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", default=PROJECT_DIR / "images/sidewalk.jpg")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output")
    args = parser.parse_args()
    left, right = save_boundaries(args.image, args.output_dir)
    print(f"Saved {len(left)} boundary pairs to {args.output_dir.resolve()}")
    if not len(left):
        print("No geometrically plausible candidate; automatic prompts may miss the sidewalk.")
