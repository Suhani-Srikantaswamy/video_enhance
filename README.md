# Video Enhancement Pipeline

FastDVDnet + MILDCLAHE + Light Sharpening for video enhancement.

## Pipeline Overview

1. **FastDVDnet Denoising**: Temporal video denoising using 5-frame windows
2. **MILDCLAHE**: Mild contrast enhancement on Y channel (YCrCb color space)
3. **Light Sharpening**: Subtle unsharp mask for detail enhancement

## Usage

```bash
python enhance.py input.mp4 output.mp4 --weights fastdvdnet_repo/model.pth
```

### Optional Parameters

```bash
python enhance.py input.mp4 output.mp4 \
  --weights fastdvdnet_repo/model.pth \
  --noise-sigma 10 \
  --clahe-clip 1.5 \
  --sharp-amount 0.3
```

- `--noise-sigma`: Estimated noise level (0-50, default: 10)
- `--clahe-clip`: CLAHE clip limit (default: 1.5 for mild enhancement)
- `--sharp-amount`: Sharpening strength (default: 0.3 for light sharpening)

## Parameters Tuning

### MILDCLAHE (Mild Contrast Enhancement)
- **Clip Limit**: 1.5 (lower = milder, typical range: 1.0-3.0)
- **Tile Size**: 8x8 (larger = smoother transitions)

### Light Sharpening
- **Amount**: 0.3 (lower = subtler, typical range: 0.2-0.5)
- **Sigma**: 1.0 (Gaussian blur radius)

### FastDVDnet
- **Temporal Window**: 5 frames (fixed for FastDVDnet)
- **Noise Sigma**: 10/255 (adjust based on input video noise level)

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy

## Model Weights

Place FastDVDnet pretrained weights at `fastdvdnet_repo/model.pth` or specify custom path with `--weights`.
