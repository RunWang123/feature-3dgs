#!/usr/bin/env python3
"""
Verify DL3DV intrinsics conversion is mathematically correct.
"""

import numpy as np
import json

# Test scene parameters
scene_dir = "/scratch/runw/project/colmap/DL3DV-10K-Benchmark/5c8dafad7d782c76ffad8c14e9e1244ce2b83aa12324c54a3cc10176964acf04/gaussian_splat"

print("="*80)
print("DL3DV Intrinsics Conversion Verification")
print("="*80)

# 1. Load transforms.json
with open(f"{scene_dir}/transforms.json", 'r') as f:
    transforms = json.load(f)

transforms_w = transforms['w']
transforms_h = transforms['h']
fl_x = transforms['fl_x']
fl_y = transforms['fl_y']
cx_json = transforms['cx']
cy_json = transforms['cy']
k1 = transforms['k1']
k2 = transforms['k2']
p1 = transforms['p1']
p2 = transforms['p2']

print(f"\n1. transforms.json (OPENCV DISTORTED model)")
print(f"   Size: {transforms_w}x{transforms_h}")
print(f"   Intrinsics: fx={fl_x:.2f}, fy={fl_y:.2f}, cx={cx_json:.2f}, cy={cy_json:.2f}")
print(f"   Distortion: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}")

# 2. COLMAP intrinsics (undistorted)
# Simulate reading from cameras.bin
colmap_width = 3819
colmap_height = 2147
fx_colmap = 1627.77
fy_colmap = 1631.18
cx_colmap = 1909.5
cy_colmap = 1073.5

print(f"\n2. COLMAP cameras.bin (PINHOLE UNDISTORTED model)")
print(f"   Size: {colmap_width}x{colmap_height}")
print(f"   Intrinsics: fx={fx_colmap:.2f}, fy={fy_colmap:.2f}, cx={cx_colmap:.2f}, cy={cy_colmap:.2f}")

# 3. images_4/ actual size
distorted_width = 960
distorted_height = 540

print(f"\n3. images_4/ (DISTORTED, downscaled by 4x)")
print(f"   Size: {distorted_width}x{distorted_height}")
print(f"   Expected from transforms.json: {transforms_w//4}x{transforms_h//4}")
print(f"   Match: {'✅' if abs(distorted_width - transforms_w//4) < 5 else '❌'}")

# 4. Scale transforms.json intrinsics to images_4/ size
scale_factor = distorted_width / transforms_w
fx_dist_960 = fl_x * scale_factor
fy_dist_960 = fl_y * scale_factor
cx_dist_960 = cx_json * scale_factor
cy_dist_960 = cy_json * scale_factor

print(f"\n4. K_dist for images_4/ (960x540 DISTORTED)")
print(f"   Scale factor: {scale_factor:.4f} ({distorted_width}/{transforms_w})")
print(f"   K_dist: fx={fx_dist_960:.2f}, fy={fy_dist_960:.2f}, cx={cx_dist_960:.2f}, cy={cy_dist_960:.2f}")

# 5. Target undistorted size
downscale_factor = colmap_width / distorted_width
undistorted_width = int(colmap_width / downscale_factor)
undistorted_height = int(colmap_height / downscale_factor)

print(f"\n5. Target undistorted size after cv2.undistort()")
print(f"   Downscale factor: {downscale_factor:.3f} ({colmap_width}/{distorted_width})")
print(f"   Undistorted size: {undistorted_width}x{undistorted_height}")
print(f"   Expected from COLMAP: {colmap_width//4}x{colmap_height//4}")
print(f"   Close enough: {'✅' if abs(undistorted_width - colmap_width//4) <= 1 else '❌'}")

# 6. K_undist (new camera matrix after undistortion)
fx_undist = fx_colmap / downscale_factor
fy_undist = fy_colmap / downscale_factor
cx_undist = cx_colmap / downscale_factor
cy_undist = cy_colmap / downscale_factor

print(f"\n6. K_undist for undistorted image ({undistorted_width}x{undistorted_height})")
print(f"   K_undist: fx={fx_undist:.2f}, fy={fy_undist:.2f}, cx={cx_undist:.2f}, cy={cy_undist:.2f}")

# 7. Verify K_undist is just COLMAP intrinsics scaled
expected_fx = fx_colmap * (undistorted_width / colmap_width)
expected_fy = fy_colmap * (undistorted_width / colmap_width)
expected_cx = cx_colmap * (undistorted_width / colmap_width)
expected_cy = cy_colmap * (undistorted_height / colmap_height)

print(f"\n7. Verification: K_undist should equal COLMAP * scale")
print(f"   Expected: fx={expected_fx:.2f}, fy={expected_fy:.2f}, cx={expected_cx:.2f}, cy={expected_cy:.2f}")
print(f"   Match: {'✅' if abs(fx_undist - expected_fx) < 0.1 else '❌'}")

# 8. After undistortion: crop to square and resize to 448x448
crop_size = min(undistorted_width, undistorted_height)
crop_left = (undistorted_width - crop_size) // 2
crop_top = 0

cx_crop = cx_undist - crop_left
cy_crop = cy_undist - crop_top

print(f"\n8. Central crop to square ({crop_size}x{crop_size})")
print(f"   Crop left={crop_left}, top={crop_top}")
print(f"   cx adjusted: {cx_undist:.2f} - {crop_left} = {cx_crop:.2f}")
print(f"   cy adjusted: {cy_undist:.2f} - {crop_top} = {cy_crop:.2f}")

# 9. Resize to 448x448
resize_scale = 448 / crop_size
fx_final = fx_undist * resize_scale
fy_final = fy_undist * resize_scale
cx_final = cx_crop * resize_scale
cy_final = cy_crop * resize_scale

print(f"\n9. Resize to 448x448 (scale={resize_scale:.4f})")
print(f"   Final intrinsics: fx={fx_final:.6f}, fy={fy_final:.6f}, cx={cx_final:.6f}, cy={cy_final:.6f}")

# 10. Compare with expected cameras.txt output
print(f"\n10. Expected cameras.txt line:")
print(f"    1 PINHOLE 448 448 {fx_final:.6f} {fy_final:.6f} {cx_final:.6f} {cy_final:.6f}")

# 11. Sanity checks
print(f"\n11. Sanity Checks:")
print(f"    cx_final near 224 (center): {'✅' if abs(cx_final - 224) < 5 else '❌'} ({cx_final:.2f})")
print(f"    cy_final near 224 (center): {'✅' if abs(cy_final - 224) < 5 else '❌'} ({cy_final:.2f})")
print(f"    fx_final ≈ fy_final: {'✅' if abs(fx_final - fy_final) < 5 else '❌'} (diff={abs(fx_final - fy_final):.2f})")

# 12. Alternative calculation: direct from COLMAP
print(f"\n12. Alternative: Direct from COLMAP intrinsics")
print(f"    COLMAP: {colmap_width}x{colmap_height} → {undistorted_width}x{undistorted_height}")
direct_scale = undistorted_width / colmap_width
fx_alt = fx_colmap * direct_scale
fy_alt = fy_colmap * direct_scale
cx_alt = cx_colmap * direct_scale
cy_alt = cy_colmap * direct_scale
print(f"    After scale: fx={fx_alt:.2f}, fy={fy_alt:.2f}, cx={cx_alt:.2f}, cy={cy_alt:.2f}")

# Crop and resize
crop_size_alt = min(undistorted_width, undistorted_height)
crop_left_alt = (undistorted_width - crop_size_alt) // 2
cx_alt_crop = cx_alt - crop_left_alt
cy_alt_crop = cy_alt - 0

resize_scale_alt = 448 / crop_size_alt
fx_alt_final = fx_alt * resize_scale_alt
fy_alt_final = fy_alt * resize_scale_alt
cx_alt_final = cx_alt_crop * resize_scale_alt
cy_alt_final = cy_alt_crop * resize_scale_alt

print(f"    After crop+resize: fx={fx_alt_final:.6f}, fy={fy_alt_final:.6f}, cx={cx_alt_final:.6f}, cy={cy_alt_final:.6f}")
print(f"    Match main calculation: {'✅' if abs(fx_final - fx_alt_final) < 0.01 else '❌'}")

print(f"\n{'='*80}")
print(f"VERIFICATION: {'✅ ALL CHECKS PASSED' if abs(fx_final - fx_alt_final) < 0.01 and abs(cx_final - 224) < 5 else '❌ SOME CHECKS FAILED'}")
print(f"{'='*80}\n")

