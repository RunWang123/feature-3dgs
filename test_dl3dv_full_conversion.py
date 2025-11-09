#!/usr/bin/env python3
"""
Test the full DL3DV conversion pipeline to verify coordinate systems are correct.

This compares:
1. Original COLMAP cameras (from images.bin) - OpenCV convention
2. transforms.json cameras - OpenGL/NeRF convention  
3. Our converted cameras - back to OpenCV/COLMAP convention

They should match!
"""

import numpy as np
import sys

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

print("="*80)
print("DL3DV Coordinate Conversion Test")
print("="*80)

# Simulate a test camera in COLMAP format (OpenCV convention)
print("\n1. ORIGINAL COLMAP CAMERA (OpenCV: Y down, Z forward)")
print("-"*80)

# Camera at (1, 2, 3), looking down +Z (forward in OpenCV)
qvec_colmap = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
tvec_colmap = np.array([-1.0, -2.0, -3.0])     # COLMAP stores -R^T * camera_position

R_w2c_colmap = qvec2rotmat(qvec_colmap)
t_w2c_colmap = tvec_colmap

w2c_colmap = np.eye(4)
w2c_colmap[:3, :3] = R_w2c_colmap
w2c_colmap[:3, 3] = t_w2c_colmap

c2w_colmap = np.linalg.inv(w2c_colmap)

print(f"W2C (COLMAP format):")
print(f"  qvec: {qvec_colmap}")
print(f"  tvec: {tvec_colmap}")
print(f"\nC2W:")
print(c2w_colmap)
print(f"Camera position: {c2w_colmap[:3, 3]}")
print(f"Camera Z-axis (viewing direction): {c2w_colmap[:3, 2]}")

# Simulate colmap2nerf conversion (what DL3DV does to create transforms.json)
print("\n2. COLMAP2NERF CONVERSION (to OpenGL: Y up, Z back)")
print("-"*80)

# This mimics colmap2nerf.py lines 237-244
c2w_nerf = c2w_colmap.copy()
c2w_nerf[:3, 1] *= -1  # Flip Y column
c2w_nerf[:3, 2] *= -1  # Flip Z column

print(f"C2W (NeRF/OpenGL format in transforms.json):")
print(c2w_nerf)
print(f"Camera position: {c2w_nerf[:3, 3]}")
print(f"Camera Y-axis (up direction): {c2w_nerf[:3, 1]}")
print(f"Camera Z-axis (back direction): {c2w_nerf[:3, 2]}")

# Our conversion back to COLMAP
print("\n3. OUR CONVERSION BACK TO COLMAP")
print("-"*80)

c2w_converted = c2w_nerf.copy()
c2w_converted[:3, 1:3] *= -1  # Flip Y and Z columns back

print(f"C2W (converted back):")
print(c2w_converted)
print(f"Camera position: {c2w_converted[:3, 3]}")
print(f"Camera Z-axis (viewing direction): {c2w_converted[:3, 2]}")

w2c_converted = np.linalg.inv(c2w_converted)

print(f"\nW2C (final COLMAP format):")
print(w2c_converted[:3, :])

# Compare
print("\n4. VERIFICATION")
print("="*80)

pos_match = np.allclose(c2w_colmap[:3, 3], c2w_converted[:3, 3])
rot_match = np.allclose(c2w_colmap[:3, :3], c2w_converted[:3, :3])
w2c_match = np.allclose(w2c_colmap[:3, :], w2c_converted[:3, :])

print(f"Camera positions match: {pos_match}")
print(f"Camera rotations match: {rot_match}")  
print(f"W2C matrices match: {w2c_match}")

if pos_match and rot_match and w2c_match:
    print("\n✅ CONVERSION IS CORRECT!")
    print("   transforms.json (OpenGL) → Our conversion → COLMAP (OpenCV)")
else:
    print("\n❌ CONVERSION HAS ERRORS!")
    print("\nDifference in C2W:")
    print(c2w_colmap - c2w_converted)
    print("\nDifference in W2C:")
    print(w2c_colmap[:3, :] - w2c_converted[:3, :])

print("\n" + "="*80)

