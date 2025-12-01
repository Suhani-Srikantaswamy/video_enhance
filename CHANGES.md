# Changes Made to enhance.py

## Key Updates

### 1. Fixed FastDVDnet Integration
- Corrected import to use `from models import FastDVDnet` from the fastdvdnet_repo
- Fixed model forward call to include required `noise_map` parameter
- Updated preprocessing to concatenate 5 frames into [1, 15, H, W] tensor format
- Added proper noise map creation and expansion

### 2. Implemented MILDCLAHE
- Renamed function to `apply_mildclahe_bgr()` for clarity
- Adjusted default parameters for mild enhancement:
  - Clip limit: 1.5 (down from 2.0)
  - Tile size: 8x8 (up from 4x4 for smoother transitions)

### 3. Implemented Light Sharpening
- Renamed function to `light_unsharp_mask()` for clarity
- Adjusted default sharpening amount to 0.3 (down from 0.9) for subtle enhancement

### 4. Enhanced CLI
- Added default weights path: `fastdvdnet_repo/model.pth`
- Added command-line arguments for tuning:
  - `--noise-sigma`: Control denoising strength
  - `--clahe-clip`: Control contrast enhancement
  - `--sharp-amount`: Control sharpening strength
- Added informative startup banner showing all parameters

### 5. Code Quality Improvements
- Added sys.path manipulation to properly import from fastdvdnet_repo
- Improved documentation and comments
- Added proper tensor shape handling
- Better error messages

## Pipeline Flow

```
Input Video
    ↓
[Frame Buffer: 5 frames]
    ↓
FastDVDnet Denoising (temporal, 5-frame window)
    ↓
MILDCLAHE (mild contrast on Y channel)
    ↓
Light Sharpening (subtle unsharp mask)
    ↓
Output Video
```

## Testing

Run with:
```bash
python enhance.py input.mp4 output.mp4
```

Or with custom parameters:
```bash
python enhance.py input.mp4 output.mp4 \
  --noise-sigma 15 \
  --clahe-clip 2.0 \
  --sharp-amount 0.5
```
