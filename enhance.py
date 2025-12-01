#!/usr/bin/env python3
"""
enhance.py - Video enhancement with FastDVDnet + MILDCLAHE + Light Sharpening
Usage:
  python enhance.py input.mp4 output.mp4 --weights fastdvdnet_repo/model.pth
  python enhance.py input.mp4 output.mp4 --fast  # Skip denoising (CLAHE + sharpening only)
  python enhance.py input.mp4 output.mp4 --scale 0.5  # Process at half resolution
"""

import argparse
import cv2
import numpy as np
import torch
from collections import deque
import os
import sys
import time

# Add fastdvdnet_repo to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fastdvdnet_repo'))

# ========== TUNABLE PARAMETERS ==========
TEMPORAL_WINDOW = 5   # Must be 5 for FastDVDnet
NOISE_SIGMA = 10.0 / 255.0  # Estimated noise level (10/255 is mild)
# MILDCLAHE parameters (mild contrast enhancement)
CLAHE_CLIP = 1.5      # Lower clip limit for mild enhancement
CLAHE_TILE = (8, 8)   # Larger tiles for smoother enhancement
# Sharpening parameters
SHARP_AMOUNT = 0.5    # Overall sharpening (increased from 0.3)
SHARP_SIGMA = 1.0
# Face enhancement parameters
FACE_SHARP_AMOUNT = 0.8   # Extra sharpening for faces
FACE_CLAHE_CLIP = 2.0     # Stronger CLAHE for faces
FACE_DETAIL_BOOST = 1.3   # Detail enhancement multiplier for faces
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =======================================

# Global face detector (loaded once)
FACE_CASCADE = None


# ---- Load FastDVDnet model
def load_fastdvdnet(weights_path, device=DEVICE):
    from models import FastDVDnet

    model = FastDVDnet(num_input_frames=5).to(device)
    state = torch.load(weights_path, map_location=device)

    # Handle different state dict formats
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # Remove 'module.' prefix if present (from DataParallel)
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '') if k.startswith('module.') else k
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.eval()
    return model


# ---- MILDCLAHE on Y channel (mild contrast enhancement)
def apply_mildclahe_bgr(img_bgr, clip_limit, tile_size):
    """Apply mild CLAHE for subtle contrast enhancement"""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    y_eq = clahe.apply(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ---- Unsharp mask sharpening
def unsharp_mask(img_bgr, amount, sigma):
    """Apply sharpening using unsharp mask"""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    img_float = img_bgr.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    result = cv2.addWeighted(img_float, 1.0 + amount, blurred_float, -amount, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ---- Detail enhancement (edge-aware, no hallucination)
def enhance_details(img_bgr, strength=1.2):
    """Enhance fine details using bilateral filter + detail layer extraction"""
    # Bilateral filter preserves edges while smoothing
    base = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    
    # Extract detail layer
    img_float = img_bgr.astype(np.float32)
    base_float = base.astype(np.float32)
    detail = img_float - base_float
    
    # Boost details and recombine
    enhanced = base_float + detail * strength
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ---- Load face detector
def get_face_detector():
    """Load OpenCV's Haar cascade face detector"""
    global FACE_CASCADE
    if FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return FACE_CASCADE


# ---- Detect faces in image
def detect_faces(img_bgr, scale_factor=1.1, min_neighbors=4, min_size=(30, 30)):
    """Detect faces and return list of (x, y, w, h) rectangles"""
    detector = get_face_detector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces


# ---- Enhance face region
def enhance_face_region(img_bgr, x, y, w, h, clahe_clip, sharp_amount, detail_boost):
    """Apply stronger enhancement to a face region with smooth blending"""
    # Expand region slightly for better blending
    pad = int(min(w, h) * 0.15)
    img_h, img_w = img_bgr.shape[:2]
    
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    
    # Extract face region
    face_region = img_bgr[y1:y2, x1:x2].copy()
    
    # Apply stronger CLAHE to face
    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
    y_ch, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
    y_eq = clahe.apply(y_ch)
    face_enhanced = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
    
    # Apply detail enhancement
    face_enhanced = enhance_details(face_enhanced, detail_boost)
    
    # Apply stronger sharpening
    face_enhanced = unsharp_mask(face_enhanced, sharp_amount, 0.8)
    
    # Create smooth blending mask (feathered edges)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    inner_pad = pad // 2
    mask[inner_pad:-inner_pad if inner_pad > 0 else None, 
         inner_pad:-inner_pad if inner_pad > 0 else None] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad/2)
    mask = np.stack([mask] * 3, axis=-1)
    
    # Blend enhanced face with original
    original_region = img_bgr[y1:y2, x1:x2].astype(np.float32)
    face_enhanced = face_enhanced.astype(np.float32)
    blended = original_region * (1 - mask) + face_enhanced * mask
    
    # Put back
    result = img_bgr.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ---- Full face-aware enhancement
def enhance_with_faces(img_bgr, clahe_clip, sharp_amount, 
                       face_clahe_clip, face_sharp_amount, face_detail_boost,
                       enable_face_enhance=True):
    """Apply enhancement with extra processing for detected faces"""
    # Base enhancement for whole image
    enhanced = apply_mildclahe_bgr(img_bgr, clahe_clip, CLAHE_TILE)
    enhanced = enhance_details(enhanced, 1.1)  # Mild detail boost
    enhanced = unsharp_mask(enhanced, sharp_amount, SHARP_SIGMA)
    
    if not enable_face_enhance:
        return enhanced
    
    # Detect and enhance faces
    faces = detect_faces(img_bgr)
    for (x, y, w, h) in faces:
        enhanced = enhance_face_region(
            enhanced, x, y, w, h,
            face_clahe_clip, face_sharp_amount, face_detail_boost
        )
    
    return enhanced


# ---- Preprocess frames for FastDVDnet
def preprocess_for_model(frames, device):
    """
    frames: list of 5 HxWx3 BGR uint8 arrays
    Returns: torch Tensor [1, 15, H, W] (5 frames * 3 channels concatenated)
    """
    frame_tensors = []
    for f in frames:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(f_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        frame_tensors.append(t)

    stacked = torch.cat(frame_tensors, dim=0).unsqueeze(0)
    return stacked.to(device)


# ---- Postprocess model output: tensor [1, 3, H, W] -> BGR uint8
def postprocess_from_model(tensor_out):
    """Convert model output tensor to BGR image"""
    out = tensor_out.squeeze(0).cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_bgr


# ---- Call FastDVDnet model on a window of frames
def denoise_fastdvdnet_window(model, frames_window, noise_sigma, device):
    """
    Denoise a window of 5 frames using FastDVDnet
    Returns: denoised center frame as BGR uint8
    """
    inp = preprocess_for_model(frames_window, device)

    # Pad to make dimensions divisible by 4
    _, _, h, w = inp.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4

    if pad_h > 0 or pad_w > 0:
        inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, h_pad, w_pad = inp.shape
    noise_map = torch.FloatTensor([noise_sigma]).to(device)
    noise_map = noise_map.view(1, 1, 1, 1).expand(1, 1, h_pad, w_pad)

    with torch.no_grad():
        out = model(inp, noise_map)

    if pad_h > 0 or pad_w > 0:
        out = out[:, :, :h, :w]

    return postprocess_from_model(out)


def format_time(seconds):
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ---- Main processing loop
def process_video(input_path, output_path, weights_path, scale, fast_mode,
                  noise_sigma, clahe_clip, clahe_tile, sharp_amount, sharp_sigma,
                  face_enhance=True, face_clahe_clip=FACE_CLAHE_CLIP, 
                  face_sharp_amount=FACE_SHARP_AMOUNT, face_detail_boost=FACE_DETAIL_BOOST):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + input_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate processing dimensions
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    # Use avc1 codec for better macOS compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    
    # Fallback to mp4v if avc1 fails
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    model = None
    if not fast_mode:
        if not os.path.exists(weights_path):
            raise FileNotFoundError("Weights file not found: " + weights_path)
        print("Loading FastDVDnet model...")
        model = load_fastdvdnet(weights_path)
        print(f"Model loaded to {DEVICE}")

    print(f"\nProcessing: {orig_w}x{orig_h} -> {proc_w}x{proc_h} ({total_frames} frames)")
    if fast_mode:
        print("Mode: FAST (CLAHE + Sharpening only)\n")
    else:
        print("Mode: FULL (FastDVDnet + CLAHE + Sharpening)\n")

    buf = deque(maxlen=TEMPORAL_WINDOW)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Empty input video.")

    # Resize first frame if needed
    if scale != 1.0:
        first_proc = cv2.resize(first, (proc_w, proc_h))
    else:
        first_proc = first

    for _ in range(TEMPORAL_WINDOW - 1):
        buf.append(first_proc.copy())
    buf.append(first_proc.copy())

    frame_idx = 0
    start_time = time.time()
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Resize for processing if needed
        if scale != 1.0:
            frame_proc = cv2.resize(frame, (proc_w, proc_h))
        else:
            frame_proc = frame

        buf.append(frame_proc.copy())
        if len(buf) < TEMPORAL_WINDOW:
            continue

        window_frames = list(buf)

        # 1) FastDVDnet denoising (skip in fast mode)
        if fast_mode:
            denoised = window_frames[TEMPORAL_WINDOW // 2]  # Use center frame
        else:
            denoised = denoise_fastdvdnet_window(model, window_frames, noise_sigma, DEVICE)

        # 2) Face-aware enhancement (CLAHE + detail boost + sharpening)
        final = enhance_with_faces(
            denoised, 
            clahe_clip, sharp_amount,
            face_clahe_clip, face_sharp_amount, face_detail_boost,
            enable_face_enhance=face_enhance
        )

        # Resize back to original if needed
        if scale != 1.0:
            final = cv2.resize(final, (orig_w, orig_h))

        out.write(final)

        frame_idx += 1
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        # Progress update every 10 frames or every frame if slow
        if frame_idx % 10 == 0 or frame_time > 2:
            avg_time = sum(frame_times[-20:]) / len(frame_times[-20:])
            remaining = (total_frames - frame_idx) * avg_time
            elapsed = time.time() - start_time
            print(f"Frame {frame_idx}/{total_frames} | "
                  f"{frame_time:.2f}s/frame | "
                  f"Elapsed: {format_time(elapsed)} | "
                  f"ETA: {format_time(remaining)}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\nFinished! Processed {frame_idx} frames in {format_time(total_time)}")
    print(f"Output: {output_path}")


# ---- CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video enhancement with FastDVDnet + MILDCLAHE + Light Sharpening"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--weights", default="fastdvdnet_repo/model.pth",
                        help="Path to FastDVDnet weights (.pth)")
    parser.add_argument("--noise-sigma", type=float, default=10.0,
                        help="Noise sigma (0-50, default: 10)")
    parser.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP,
                        help="CLAHE clip limit (default: 1.5)")
    parser.add_argument("--sharp-amount", type=float, default=SHARP_AMOUNT,
                        help="Sharpening amount (default: 0.3)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for processing (0.5 = half res, faster)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip FastDVDnet, only CLAHE + sharpening")
    parser.add_argument("--no-face", action="store_true",
                        help="Disable face detection and enhancement")
    parser.add_argument("--face-sharp", type=float, default=FACE_SHARP_AMOUNT,
                        help="Face sharpening amount (default: 0.8)")
    parser.add_argument("--face-clahe", type=float, default=FACE_CLAHE_CLIP,
                        help="Face CLAHE clip limit (default: 2.0)")

    args = parser.parse_args()

    print("=== Video Enhancement Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {DEVICE}")
    print(f"Scale: {args.scale}")
    print(f"Fast mode: {args.fast}")
    if not args.fast:
        print(f"Weights: {args.weights}")
        print(f"Noise Sigma: {args.noise_sigma}/255")
    print(f"CLAHE Clip: {args.clahe_clip}")
    print(f"Sharpening: {args.sharp_amount}")
    print(f"Face Enhancement: {not args.no_face}")
    if not args.no_face:
        print(f"Face Sharpening: {args.face_sharp}")
        print(f"Face CLAHE: {args.face_clahe}")
    print("==================================")

    process_video(
        args.input, args.output, args.weights,
        scale=args.scale,
        fast_mode=args.fast,
        noise_sigma=args.noise_sigma / 255.0,
        clahe_clip=args.clahe_clip,
        clahe_tile=CLAHE_TILE,
        sharp_amount=args.sharp_amount,
        sharp_sigma=SHARP_SIGMA,
        face_enhance=not args.no_face,
        face_clahe_clip=args.face_clahe,
        face_sharp_amount=args.face_sharp,
        face_detail_boost=FACE_DETAIL_BOOST
    )
