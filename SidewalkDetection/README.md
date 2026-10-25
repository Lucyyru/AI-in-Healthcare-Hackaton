# SidewalkDetection

中文 | [English](README_EN.md)

## 怎么使用

把照片放进 `images` 文件夹，在 `SidewalkDetection` 文件夹中打开终端，使用已安装好依赖的 Python 环境运行：

```powershell
python pipeline.py images/image8.jpg --height 1.24 --wheelchair-width 0.65
```

每次测试只需修改三个地方：

- `image8.jpg`：你的图片文件名，注意 `.jpg` 或 `.png`。
- `--height 1.24`：相机镜头中心离地高度，单位米。
- `--wheelchair-width 0.65`：轮椅宽度，单位米；65 厘米填 `0.65`。

运行后，终端先显示英文进度，最后显示通道宽度和比较结果：

```text
1.5848951751805072
Can pass!
```

- `Can pass!`：测得的通道宽度大于轮椅宽度。
- `Cannot pass`：通道宽度小于或等于轮椅宽度。
- `Unable to determine: no valid passage width.`：未获得有效测量。

必须带上 `--wheelchair-width` 才会比较；删掉这个参数和它的数值，就只输出通道宽度。

结果保存在 `output`，每次运行覆盖同名文件：

- 打开 `measurement.png` 看边界和测量结果。
- 打开 `measurement.json` 看 `width_m`、轮椅宽度、判断结果及测量警告。

## 怎么工作

1. **SAM 2**：自动提示模型选取通道区域，提取左右边界。
2. **GeoCalib**：估计相机焦距、俯仰角和横滚角。
3. **宽度计算**：结合相机高度，将边界投影到地面，在多个深度测宽并取中位数。
4. **轮椅比较**：将测得宽度与传入的轮椅宽度比较，输出结论。

模型和相关代码放在 `models`，测试图片放在 `images`，几何测试放在 `tests`。
当前为原型：比较只看测得的宽度，不额外判断障碍物；分割和标定误差也会影响结果。

## 使用的模型与声明

- **Meta SAM 2.1 Small**：使用预训练模型 [`facebook/sam2.1-hiera-small`](https://huggingface.co/facebook/sam2.1-hiera-small)，通过自动提示分割区域、提取边界。模型采用 Apache-2.0 许可证；[官方项目](https://github.com/facebookresearch/sam2)。
- **GeoCalib（ECCV 2024，ETH Zurich / CVG）**：使用预训练 pinhole 模型估计焦距和相机姿态；[官方项目](https://github.com/cvg/GeoCalib)。代码采用 Apache-2.0，权重采用 CC BY 4.0；[代码许可证](models/GeoCalib/source/LICENSE)、[权重许可证](models/GeoCalib/source/LICENSE-WEIGHTS)均已保留。

本项目调用以上预训练模型，没有重新训练模型。GeoCalib 源码版本为 `1dedb29f94fc11b896c5f572005978aeebccd67f`。
SAM 2 本身没有“人行道”语义标签，当前由自动提示选择区域；GeoCalib 默认主点位于图像中心。
