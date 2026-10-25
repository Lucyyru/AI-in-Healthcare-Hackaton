# SidewalkDetection

English | [中文](README.md)

## How to use

Place your photo in `images`, open a terminal in the `SidewalkDetection` folder, and run this command using a Python environment with the required packages installed:

```powershell
python pipeline.py images/image8.jpg --height 1.24 --wheelchair-width 0.65
```

For each test, change three values:

- `image8.jpg`: your image filename, including the correct `.jpg` or `.png` extension.
- `--height 1.24`: vertical camera height above the ground, in meters.
- `--wheelchair-width 0.65`: wheelchair width in meters; enter `0.65` for 65 cm.

The terminal shows progress in English, followed by the estimated passage width and comparison result:

```text
1.5848951751805072
Can pass!
```

- `Can pass!`: the measured passage width is greater than the wheelchair width.
- `Cannot pass`: the measured passage width is less than or equal to the wheelchair width.
- `Unable to determine: no valid passage width.`: no valid measurement was obtained.

Include `--wheelchair-width` to enable the comparison. Remove this argument and its value to output only the passage width.

Results are saved in `output`. Each run overwrites files with the same names:

- Open `measurement.png` to view the boundaries and measurement.
- Open `measurement.json` for `width_m`, wheelchair width, the comparison result and measurement warnings.

## How it works

1. **SAM 2**: uses automatic prompts to select the passage region and extract its left and right boundaries.
2. **GeoCalib**: estimates the camera focal length, pitch and roll.
3. **Width calculation**: uses camera height and pose to project boundaries onto the ground, measures widths at multiple depths and returns their median.
4. **Wheelchair comparison**: compares the measured passage width with the supplied wheelchair width.

Models and their code are in `models`, input photos are in `images`, and geometry tests are in `tests`.
This is a prototype: the comparison uses only the estimated width and does not separately assess obstacles. Segmentation and calibration errors affect the result.

## Models and attribution

- **Meta SAM 2.1 Small**: uses the pretrained [`facebook/sam2.1-hiera-small`](https://huggingface.co/facebook/sam2.1-hiera-small) checkpoint for prompted segmentation and boundary extraction. The model is licensed under Apache-2.0; [official project](https://github.com/facebookresearch/sam2).
- **GeoCalib (ECCV 2024, ETH Zurich / CVG)**: uses the pretrained pinhole model to estimate focal length and camera orientation; [official project](https://github.com/cvg/GeoCalib). The code is licensed under Apache-2.0 and the model weights under CC BY 4.0. Both the [code license](models/GeoCalib/source/LICENSE) and [weights license](models/GeoCalib/source/LICENSE-WEIGHTS) are retained.

This project uses these pretrained models without retraining them. The GeoCalib source revision is `1dedb29f94fc11b896c5f572005978aeebccd67f`.
SAM 2 has no semantic "sidewalk" label; automatic prompts select the region. GeoCalib assumes that the principal point is at the image center.
