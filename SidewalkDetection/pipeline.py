"""Independent SAM 2 -> GeoCalib -> ground projection -> width pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from models.SAM2.sam2_boundary import save_boundaries
from models.GeoCalib.calibration import calibrate_image
from geometry import measure_width
from wheelchair_check import compare_width

ROOT = Path(__file__).resolve().parent


def _save_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _horizon_segment(line, width, height):
    a, b, c = line
    points = []
    if abs(b) > 1e-12:
        points.extend((x, -(a*x+c)/b) for x in (0,width-1))
    if abs(a) > 1e-12:
        points.extend((-(b*y+c)/a, y) for y in (0,height-1))
    inside = [(float(x),float(y)) for x,y in points if 0 <= x < width and 0 <= y < height]
    return inside[:2]


def _visualize(directory, calibration, report):
    with Image.open(directory / "boundaries.png") as source:
        image = source.convert("RGB")
    drawing = ImageDraw.Draw(image)
    endpoints = _horizon_segment(calibration["horizon_line_abc"], *image.size)
    if len(endpoints) == 2:
        drawing.line(endpoints, fill="orange", width=5)
    width = report["width_m"]
    text = f"Selected-region median width: {width:.3f} m" if width is not None else "No valid width"
    font = ImageFont.load_default(size=32)
    drawing.rectangle((5,30,1600,135),fill="black")
    drawing.text((15,37),text,fill="white",font=font)
    drawing.text((15,82),"Orange: calibrated horizon | Red/cyan: SAM 2 boundaries",fill="orange",font=font)
    image.save(directory / "measurement.png")


def run_pipeline(image_path, height_m, output_dir=None, wheelchair_width_m=None):
    """Save all intermediate artifacts and return the measurement dictionary."""
    image_path = Path(image_path).resolve()
    directory = Path(output_dir) if output_dir is not None else ROOT / "output"
    directory.mkdir(parents=True,exist_ok=True)

    print("Running SAM 2: segmenting the sidewalk and extracting boundaries.",flush=True)
    left,right = save_boundaries(image_path,directory)

    print("Running GeoCalib: estimating camera intrinsics and orientation.",flush=True)
    calibration = calibrate_image(image_path)
    _save_json(directory / "calibration.json",calibration)

    print("Running width calculation: projecting boundaries and measuring width.",flush=True)
    report,samples = measure_width(left,right,calibration,height_m)
    report.update(image=str(image_path),
                  calibration="GeoCalib pinhole; principal point fixed at image center",
                  validated_against_ground_truth=False,
                  assumptions=["Caller-provided camera height is accurate", "Ground is flat",
                               "Selected boundaries represent approximately parallel passage edges"],
                  sample_columns=["depth_m","left_x_m","right_x_m","horizontal_distance_m","width_m"])
    if wheelchair_width_m is not None:
        report["passability"] = compare_width(report["width_m"], wheelchair_width_m)
        report["wheelchair_width_m"] = float(wheelchair_width_m)
    np.save(directory / "width_samples.npy",samples,allow_pickle=False)
    np.savetxt(directory / "width_samples.csv",samples,delimiter=",",
               header=",".join(report["sample_columns"]),comments="")
    _save_json(directory / "measurement.json",report)
    _visualize(directory,calibration,report)
    # 测试输出
    # print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print(report["width_m"],flush=True)
    if wheelchair_width_m is not None:
        print(report["passability"],flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path,nargs="?",default=ROOT / "images/image8.jpg")
    parser.add_argument("--height",type=float,required=True,help="Camera height in meters, e.g. 1.24")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--wheelchair-width",type=float,help="Wheelchair width in meters, e.g. 0.65")
    args = parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    run_pipeline(args.image,args.height,args.output_dir,wheelchair_width_m=args.wheelchair_width)
