#!/usr/bin/env python3
"""
Test the full COLMAP → NeRF → COLMAP roundtrip conversion.

This simulates what DL3DV does:
1. COLMAP original (qvec, tvec in W2C format)
2. colmap2nerf.py converts to transforms.json (C2W in OpenGL)
3. Our script converts back to COLMAP (qvec, tvec in W2C format)

Expected: Should get back the SAME qvec and tvec!
"""

import numpy as np

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

def rotmat2qvec(R):
    """Convert rotation matrix to quaternion."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

print("="*80)
print("COLMAP → NeRF → COLMAP Roundtrip Test")
print("="*80)

# Step 1: Original COLMAP camera
print("\n1. ORIGINAL COLMAP (W2C format, OpenCV convention)")
print("-"*80)

# Camera at (1, 2, 3) in OpenCV convention
qvec_original = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
tvec_original = np.array([-1.0, -2.0, -3.0])    # COLMAP stores -R^T @ camera_position

print(f"qvec: {qvec_original}")
print(f"tvec: {tvec_original}")

# Step 2: COLMAP → NeRF (what colmap2nerf.py does)
print("\n2. COLMAP → NeRF CONVERSION (colmap2nerf.py logic)")
print("-"*80)

# From colmap2nerf.py lines 237-245
R = qvec2rotmat(-qvec_original)  # ← NOTE: NEGATIVE qvec!
t = tvec_original.reshape([3,1])
bottom = np.array([0,0,0,1.]).reshape([1,4])
m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
c2w_nerf = np.linalg.inv(m)
c2w_nerf[:3, 2] *= -1  # flip Z
c2w_nerf[:3, 1] *= -1  # flip Y

print(f"C2W (NeRF/OpenGL in transforms.json):")
print(c2w_nerf)

# Step 3: NeRF → COLMAP (our conversion)
print("\n3. NeRF → COLMAP CONVERSION (our script)")
print("-"*80)

c2w_back = c2w_nerf.copy()
c2w_back[:3, 1:3] *= -1  # Flip Y and Z back

w2c_back = np.linalg.inv(c2w_back)
R_w2c_back = w2c_back[:3, :3]
t_w2c_back = w2c_back[:3, 3]

# Convert to quaternion
qvec_back = rotmat2qvec(R_w2c_back)
tvec_back = t_w2c_back

print(f"qvec: {qvec_back}")
print(f"tvec: {tvec_back}")

# Step 4: VERIFY
print("\n4. VERIFICATION")
print("="*80)

# Account for quaternion sign ambiguity (q and -q represent same rotation)
qvec_match = (np.allclose(qvec_original, qvec_back) or 
              np.allclose(qvec_original, -qvec_back))
tvec_match = np.allclose(tvec_original, tvec_back)

print(f"Quaternions match: {qvec_match}")
if not qvec_match:
    print(f"  Original: {qvec_original}")
    print(f"  Got back: {qvec_back}")
    print(f"  Negated:  {-qvec_back}")
    
print(f"Translations match: {tvec_match}")
if not tvec_match:
    print(f"  Original: {tvec_original}")
    print(f"  Got back: {tvec_back}")
    print(f"  Difference: {tvec_original - tvec_back}")

if qvec_match and tvec_match:
    print("\n✅ ROUNDTRIP SUCCESSFUL!")
    print("   COLMAP → NeRF (DL3DV) → Our Conversion → COLMAP ✓")
else:
    print("\n❌ ROUNDTRIP FAILED!")
    print("   There's a bug in our conversion logic!")

# Bonus: Check camera position
print("\n5. CAMERA POSITION CHECK")
print("="*80)

# Original camera position from COLMAP W2C
R_orig = qvec2rotmat(qvec_original)
cam_pos_orig = -R_orig.T @ tvec_original

# Converted camera position
cam_pos_back = -R_w2c_back.T @ tvec_back

print(f"Original camera position: {cam_pos_orig}")
print(f"Converted camera position: {cam_pos_back}")
print(f"Positions match: {np.allclose(cam_pos_orig, cam_pos_back)}")

