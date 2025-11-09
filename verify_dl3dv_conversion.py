#!/usr/bin/env python3
"""
Verify DL3DV coordinate conversion by comparing:
1. Original transforms.json (OpenGL/NeRF convention)
2. Converted COLMAP format
3. Expected OpenCV/COLMAP convention
"""

import numpy as np
import json
import os
import sys

def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (same as in conversion script)."""
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

def test_coordinate_conversion():
    """Test the coordinate system conversion."""
    print("="*80)
    print("Testing Coordinate System Conversion")
    print("="*80)
    
    # Create a test camera in OpenGL/Blender convention
    # Camera at (0, 0, 2), looking down -Z (into screen)
    print("\n1. Test Camera in OpenGL/Blender Convention:")
    print("   Position: (0, 0, 2)")
    print("   Looking: down -Z axis (into screen)")
    print("   Up: +Y axis")
    
    c2w_opengl = np.array([
        [1,  0,  0,  0],   # X-axis points right
        [0,  1,  0,  0],   # Y-axis points up
        [0,  0,  1,  2],   # Z-axis points back (out of screen), camera at z=2
        [0,  0,  0,  1]
    ], dtype=float)
    
    print(f"\n   C2W (OpenGL):")
    print(f"   {c2w_opengl}")
    
    # Apply the conversion: flip Y and Z columns
    c2w_opencv = c2w_opengl.copy()
    c2w_opencv[:3, 1:3] *= -1
    
    print(f"\n2. After flipping Y/Z columns -> OpenCV/COLMAP Convention:")
    print(f"   C2W (OpenCV):")
    print(f"   {c2w_opencv}")
    print(f"   Position: ({c2w_opencv[0,3]}, {c2w_opencv[1,3]}, {c2w_opencv[2,3]})")
    print(f"   X-axis: ({c2w_opencv[0,0]}, {c2w_opencv[1,0]}, {c2w_opencv[2,0]})")
    print(f"   Y-axis: ({c2w_opencv[0,1]}, {c2w_opencv[1,1]}, {c2w_opencv[2,1]})")
    print(f"   Z-axis: ({c2w_opencv[0,2]}, {c2w_opencv[1,2]}, {c2w_opencv[2,2]})")
    
    # Convert to W2C
    w2c = np.linalg.inv(c2w_opencv)
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    
    print(f"\n3. W2C (for COLMAP):")
    print(f"   Rotation:")
    print(f"   {R_w2c}")
    print(f"   Translation: {t_w2c}")
    
    # Convert to quaternion
    qvec = rotmat2qvec(R_w2c)
    print(f"\n4. Quaternion: [{qvec[0]:.6f}, {qvec[1]:.6f}, {qvec[2]:.6f}, {qvec[3]:.6f}]")
    
    # Verify: convert back to rotation matrix
    R_verify = qvec2rotmat(qvec)
    print(f"\n5. Verify - Rotation from quaternion:")
    print(f"   {R_verify}")
    print(f"   Match: {np.allclose(R_w2c, R_verify)}")
    
    # Check what the camera is looking at
    # In OpenCV convention, camera looks down +Z axis
    # Transform a point in front of the camera
    point_in_camera = np.array([0, 0, 1])  # 1 meter in front
    point_in_world = c2w_opencv[:3, :3] @ point_in_camera + c2w_opencv[:3, 3]
    
    print(f"\n6. Camera viewing direction:")
    print(f"   Point 1m in front of camera (camera coords): {point_in_camera}")
    print(f"   Same point in world coords: {point_in_world}")
    print(f"   Camera is looking towards: +Z in world (forward)")
    
    return True

def verify_real_scene(scene_path):
    """Verify conversion on a real DL3DV scene."""
    print("\n" + "="*80)
    print(f"Verifying Real Scene: {os.path.basename(scene_path)}")
    print("="*80)
    
    transforms_path = os.path.join(scene_path, 'gaussian_splat', 'transforms.json')
    if not os.path.exists(transforms_path):
        print(f"❌ transforms.json not found at {transforms_path}")
        return False
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    if len(transforms['frames']) == 0:
        print("❌ No frames in transforms.json")
        return False
    
    # Test first frame
    frame = transforms['frames'][0]
    c2w_original = np.array(frame['transform_matrix'])
    
    print(f"\nFirst frame: {frame['file_path']}")
    print(f"\nOriginal C2W (OpenGL/NeRF convention):")
    print(c2w_original)
    print(f"Camera position: {c2w_original[:3, 3]}")
    
    # Apply conversion
    c2w_converted = c2w_original.copy()
    c2w_converted[:3, 1:3] *= -1
    
    print(f"\nConverted C2W (OpenCV/COLMAP convention):")
    print(c2w_converted)
    print(f"Camera position: {c2w_converted[:3, 3]}")
    
    # Check if position makes sense (should be same magnitude)
    orig_dist = np.linalg.norm(c2w_original[:3, 3])
    conv_dist = np.linalg.norm(c2w_converted[:3, 3])
    print(f"\nCamera distance from origin:")
    print(f"  Original: {orig_dist:.3f}")
    print(f"  Converted: {conv_dist:.3f}")
    print(f"  Match: {np.isclose(orig_dist, conv_dist)}")
    
    # Convert to W2C
    w2c = np.linalg.inv(c2w_converted)
    print(f"\nW2C (COLMAP format):")
    print(f"Rotation:")
    print(w2c[:3, :3])
    print(f"Translation: {w2c[:3, 3]}")
    
    return True

if __name__ == "__main__":
    # Run coordinate system test
    test_coordinate_conversion()
    
    # If scene path provided, verify real scene
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
        verify_real_scene(scene_path)
    else:
        print("\n" + "="*80)
        print("To verify a real scene, run:")
        print(f"  python {sys.argv[0]} /path/to/dl3dv/scene")
        print("="*80)

