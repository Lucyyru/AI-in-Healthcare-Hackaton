"""Original-raster pinhole calibration using the official GeoCalib model."""
from functools import lru_cache
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image
import torch

MODEL_DIR = Path(__file__).resolve().parent
SOURCE_REVISION = "1dedb29f94fc11b896c5f572005978aeebccd67f"


@lru_cache(maxsize=1)
def _model():
    # Keep the upstream source and checkpoint inside this independent project.
    sys.path.insert(0, str(MODEL_DIR / "source"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=r"`torch\.jit\.script` is deprecated\..*",
                                module=r"torch\.jit\._script")
        from geocalib import GeoCalib
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GeoCalib(weights=str(MODEL_DIR / "weights/pinhole.tar")).to(device).eval(), device


def calibrate_image(image_path):
    model, device = _model()
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
    # Upstream NMF builds random bases even at inference; make repeats comparable.
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        prediction = model.calibrate(tensor, camera_model="pinhole")
    camera, gravity = prediction["camera"], prediction["gravity"]
    k = camera.K[0].detach().cpu().numpy().astype(float)
    # Upstream gives (0,-1,0) for a level OpenCV camera: this vector points UP.
    up = gravity.vec3d[0].detach().cpu().numpy().astype(float)
    up /= np.linalg.norm(up)
    horizon = np.linalg.solve(k.T, up)
    horizon /= np.linalg.norm(horizon[:2])
    roll, pitch = np.degrees(gravity.rp[0].detach().cpu().numpy())
    uncertainty = {}
    for name in ("focal_uncertainty", "roll_uncertainty", "pitch_uncertainty"):
        if name in prediction:
            value = float(prediction[name].detach().cpu().reshape(-1)[0])
            uncertainty[name + ("_deg" if name != "focal_uncertainty" else "_px")] = (
                float(np.degrees(value)) if name != "focal_uncertainty" else value
            )
    return {
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "K": k.tolist(),
        "fx": float(k[0, 0]), "fy": float(k[1, 1]),
        "cx": float(k[0, 2]), "cy": float(k[1, 2]),
        "up_camera": up.tolist(),
        "horizon_line_abc": horizon.tolist(),
        "roll_deg_geocalib": float(roll), "pitch_deg_geocalib": float(pitch),
        "principal_point_estimated": False,
        "principal_point_note": "Assumed at image center, not optimized by GeoCalib",
        "pose_note": "Angles use GeoCalib conventions; projection uses up_camera directly",
        "camera_model": "pinhole",
        "source_revision": SOURCE_REVISION,
        "random_seed": 0,
        "uncertainty": uncertainty,
    }
