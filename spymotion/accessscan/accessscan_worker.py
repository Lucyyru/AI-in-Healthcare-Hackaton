"""Keep the AccessScan models and the last image's measurements in memory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


class ImageMeasurements:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.image_key = None
        self.boundaries = None
        self.calibration = None

    def recognize(self, image):
        from models.SAM2.sam2_boundary import detect_boundaries
        from models.GeoCalib.calibration import calibrate_image

        start = time.perf_counter()
        boundaries = detect_boundaries(image)
        sam_seconds = time.perf_counter() - start
        start = time.perf_counter()
        calibration = calibrate_image(image)
        return boundaries, calibration, {
            "sam2_s": sam_seconds, "geocalib_s": time.perf_counter() - start,
        }

    def analyze(self, request):
        import numpy as np
        from geometry import measure_width
        from wheelchair_check import compare_width

        start = time.perf_counter()
        image = Path(request["image"])
        key = hashlib.sha256(image.read_bytes()).hexdigest()
        reused = key == self.image_key
        timings = {"sam2_s": 0.0, "geocalib_s": 0.0}
        if not reused:
            boundaries, calibration, timings = self.recognize(image)
            self.boundaries, self.calibration, self.image_key = boundaries, calibration, key

        left, right = self.boundaries
        report, samples = measure_width(left, right, self.calibration, request["height_m"])
        report.update(
            image=str(image), wheelchair_width_m=request["wheelchair_width_m"],
            passability=compare_width(report["width_m"], request["wheelchair_width_m"]),
            calibration="GeoCalib pinhole; principal point fixed at image center",
            validated_against_ground_truth=False,
            measurement_note="Median width of the selected visible region; not verified physical passage width",
            image_cache_hit=reused, worker_pid=os.getpid(),
            visualizations_generated=False,
            sample_columns=["depth_m", "left_x_m", "right_x_m", "horizontal_distance_m", "width_m"],
        )
        np.save(self.output / "left_boundary.npy", left, allow_pickle=False)
        np.save(self.output / "right_boundary.npy", right, allow_pickle=False)
        np.save(self.output / "width_samples.npy", samples, allow_pickle=False)
        np.savetxt(self.output / "width_samples.csv", samples, delimiter=",",
                   header=",".join(report["sample_columns"]), comments="")
        (self.output / "calibration.json").write_text(
            json.dumps(self.calibration, indent=2, allow_nan=False), encoding="utf-8"
        )
        # These older visualizations do not represent this number-only run.
        for filename in ("sidewalk_mask.png", "boundaries.png", "measurement.png"):
            (self.output / filename).unlink(missing_ok=True)
        timings["total_s"] = time.perf_counter() - start
        report["timings_s"] = timings
        (self.output / "measurement.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        return report


def main():
    backend = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(backend))
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    measurements = ImageMeasurements(backend / "output")
    for line in sys.stdin:
        request = json.loads(line)
        try:
            response = {"ok": True, "report": measurements.analyze(request)}
        except Exception as error:
            traceback.print_exc()
            response = {"ok": False, "error": str(error)}
        # Publish the complete response atomically for the frontend reader.
        destination = Path(request["response"])
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(response, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)


if __name__ == "__main__":
    main()
