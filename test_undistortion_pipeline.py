#!/usr/bin/env python3
"""
Test the actual undistortion pipeline with a real image.
"""

import numpy as np
import cv2
import json
from PIL import Image

scene_dir = "/scratch/runw/project/colmap/DL3DV-10K-Benchmark/5c8dafad7d782c76ffad8c14e9e1244ce2b83aa12324c54a3cc10176964acf04/gaussian_splat"

print("="*80)
print("Testing Undistortion Pipeline")
print("="*80)

# Load transforms.json
with open(f"{scene_dir}/transforms.json", 'r') as f:
    transforms = json.load(f)

# Load first image
test_image_path = f"{scene_dir}/images_4/frame_00001.png"
img = cv2.imread(test_image_path)
h_dist, w_dist = img.shape[:2]

print(f"\n1. Input image: {test_image_path}")
print(f"   Size: {w_dist}x{h_dist}")

# Build distorted camera matrix for images_4/
scale_factor = w_dist / transforms['w']
K_dist = np.array([
    [transforms['fl_x'] * scale_factor, 0, transforms['cx'] * scale_factor],
    [0, transforms['fl_y'] * scale_factor, transforms['cy'] * scale_factor],
    [0, 0, 1]
])

dist_coeffs = np.array([
    transforms.get('k1', 0),
    transforms.get('k2', 0),
    transforms.get('p1', 0),
    transforms.get('p2', 0),
    0  # k3
])

print(f"\n2. Distorted camera matrix K_dist:")
print(K_dist)
print(f"   Distortion coefficients: {dist_coeffs}")

# Build undistorted camera matrix (from COLMAP)
colmap_width = 3819
colmap_height = 2147
fx_colmap = 1627.77
fy_colmap = 1631.18
cx_colmap = 1909.5
cy_colmap = 1073.5

downscale = colmap_width / w_dist
K_undist = np.array([
    [fx_colmap / downscale, 0, cx_colmap / downscale],
    [0, fy_colmap / downscale, cy_colmap / downscale],
    [0, 0, 1]
])

undist_w = int(colmap_width / downscale)
undist_h = int(colmap_height / downscale)

print(f"\n3. Undistorted camera matrix K_undist:")
print(K_undist)
print(f"   Target size: {undist_w}x{undist_h}")

# Perform undistortion
print(f"\n4. Running cv2.undistort()...")
img_undist = cv2.undistort(img, K_dist, dist_coeffs, None, K_undist)
h_out, w_out = img_undist.shape[:2]

print(f"   Output size: {w_out}x{h_out}")
print(f"   Expected: {undist_w}x{undist_h}")
print(f"   Match: {'✅' if w_out == undist_w and h_out == undist_h else '❌'}")

# Crop to target size if needed
if w_out != undist_w or h_out != undist_h:
    print(f"\n5. Cropping to target size...")
    start_x = (w_out - undist_w) // 2
    start_y = (h_out - undist_h) // 2
    img_undist = img_undist[start_y:start_y+undist_h, start_x:start_x+undist_w]
    h_out, w_out = img_undist.shape[:2]
    print(f"   After crop: {w_out}x{h_out}")

# Central crop to square
crop_size = min(w_out, h_out)
crop_left = (w_out - crop_size) // 2
crop_top = 0
img_crop = img_undist[crop_top:crop_top+crop_size, crop_left:crop_left+crop_size]

print(f"\n6. Central crop to square:")
print(f"   Crop size: {crop_size}x{crop_size}")
print(f"   Crop offset: left={crop_left}, top={crop_top}")

# Resize to 448x448
img_resized = cv2.resize(img_crop, (448, 448), interpolation=cv2.INTER_LANCZOS4)

print(f"\n7. Resize to 448x448:")
print(f"   Final size: {img_resized.shape[1]}x{img_resized.shape[0]}")

# Compute final intrinsics
fx_undist = K_undist[0, 0]
fy_undist = K_undist[1, 1]
cx_undist = K_undist[0, 2]
cy_undist = K_undist[1, 2]

cx_crop = cx_undist - crop_left
cy_crop = cy_undist - crop_top

resize_scale = 448 / crop_size
fx_final = fx_undist * resize_scale
fy_final = fy_undist * resize_scale
cx_final = cx_crop * resize_scale
cy_final = cy_crop * resize_scale

print(f"\n8. Final intrinsics:")
print(f"   fx={fx_final:.6f}, fy={fy_final:.6f}")
print(f"   cx={cx_final:.6f}, cy={cy_final:.6f}")
print(f"   Expected cameras.txt:")
print(f"   1 PINHOLE 448 448 {fx_final:.6f} {fy_final:.6f} {cx_final:.6f} {cy_final:.6f}")

# Sanity checks
print(f"\n9. Sanity Checks:")
success = True
if not (abs(cx_final - 224) < 5):
    print(f"   ❌ cx_final={cx_final:.2f} is not near 224 (center)")
    success = False
else:
    print(f"   ✅ cx_final={cx_final:.2f} is near 224")

if not (abs(cy_final - 224) < 5):
    print(f"   ❌ cy_final={cy_final:.2f} is not near 224 (center)")
    success = False
else:
    print(f"   ✅ cy_final={cy_final:.2f} is near 224")

if not (abs(fx_final - fy_final) < 5):
    print(f"   ⚠️  fx_final and fy_final differ by {abs(fx_final - fy_final):.2f} (might be OK)")
else:
    print(f"   ✅ fx_final ≈ fy_final (diff={abs(fx_final - fy_final):.2f})")

print(f"\n{'='*80}")
if success:
    print(f"✅ PIPELINE TEST PASSED - Undistortion working correctly!")
else:
    print(f"❌ PIPELINE TEST FAILED - Check calculations")
print(f"{'='*80}\n")

# Save sample outputs for visual inspection
try:
    cv2.imwrite("/tmp/dl3dv_distorted.png", img)
    cv2.imwrite("/tmp/dl3dv_undistorted.png", img_undist)
    cv2.imwrite("/tmp/dl3dv_final_448.png", img_resized)
    print(f"📁 Saved test images to /tmp/dl3dv_*.png for visual inspection")
except Exception as e:
    print(f"⚠️  Could not save test images: {e}")

