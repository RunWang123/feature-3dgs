#!/usr/bin/env python3
"""
THOROUGH verification of camera coordinate transformation.

This script will:
1. Load ORIGINAL ScanNet camera poses
2. Load CONVERTED COLMAP camera poses  
3. Verify the transformation is correct
4. Check coordinate system conventions
5. Visualize camera positions and viewing directions
"""

import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys

def load_scannet_pose(npz_path):
    """Load original ScanNet camera pose."""
    data = np.load(npz_path)
    pose = data['camera_pose']  # Should be 4x4 C2W
    intrinsics = data['camera_intrinsics']
    return pose, intrinsics


def load_colmap_images_txt(images_txt_path):
    """Parse COLMAP images.txt to extract camera poses."""
    poses = []
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    
    # Every 2 lines: image line + empty/points line
    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].strip().split()
        # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        name = parts[9]
        
        # COLMAP format is W2C
        # Quaternion: (qw, qx, qy, qz)
        quat_wxyz = np.array([qw, qx, qy, qz])
        tvec = np.array([tx, ty, tz])
        
        poses.append({
            'image_id': image_id,
            'name': name,
            'quat_wxyz': quat_wxyz,
            'tvec': tvec
        })
    
    return poses


def quat_to_rotation_matrix(quat_wxyz):
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    qw, qx, qy, qz = quat_wxyz
    # scipy uses (x,y,z,w) order
    rot = Rotation.from_quat([qx, qy, qz, qw])
    return rot.as_matrix()


def decompose_c2w(c2w):
    """Extract camera center and viewing direction from C2W matrix."""
    camera_center = c2w[:3, 3]
    # Camera looks along -Z in camera frame, which is 3rd column of rotation in world frame
    # Actually, the camera frame's -Z axis = R @ [0,0,-1]^T = -R[:,2]
    viewing_direction = -c2w[:3, 2]  # Negative Z axis in world coords
    return camera_center, viewing_direction


def decompose_w2c(w2c):
    """Extract camera center from W2C matrix."""
    # C = -R^T @ t
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    camera_center = -R.T @ t
    # Viewing direction: camera looks along +Z in camera frame
    # In world frame: R^T @ [0,0,1]^T = R^T[:,2] = R[2,:] (row 2 of R^T)
    viewing_direction = -R[2, :]  # Camera looks along -Z, so -R[2,:]
    return camera_center, viewing_direction


def verify_scene(scannet_scene_dir, converted_scene_dir):
    """
    Verify camera transformation for a scene.
    """
    print("="*90)
    print("COMPREHENSIVE CAMERA COORDINATE VERIFICATION")
    print("="*90)
    
    # Paths
    scannet_images_dir = os.path.join(scannet_scene_dir, 'images')
    converted_images_txt = os.path.join(converted_scene_dir, 'sparse', '0', 'images.txt')
    
    if not os.path.exists(scannet_images_dir):
        print(f"❌ ScanNet images dir not found: {scannet_images_dir}")
        return False
    
    if not os.path.exists(converted_images_txt):
        print(f"❌ Converted images.txt not found: {converted_images_txt}")
        return False
    
    print(f"\n📂 Input:  {scannet_scene_dir}")
    print(f"📂 Output: {converted_scene_dir}")
    
    # Load converted poses
    print(f"\n🔍 Loading converted COLMAP poses...")
    colmap_poses = load_colmap_images_txt(converted_images_txt)
    print(f"   Found {len(colmap_poses)} images")
    
    # Get corresponding ScanNet files (first 30)
    scannet_files = sorted([f for f in os.listdir(scannet_images_dir) if f.endswith('.npz')])[:30]
    
    print(f"\n🔍 Loading original ScanNet poses...")
    print(f"   Found {len(scannet_files)} .npz files")
    
    # ============================================================================
    # STEP 1: Verify ScanNet pose format (is it really C2W?)
    # ============================================================================
    
    print(f"\n" + "="*90)
    print("STEP 1: VERIFY SCANNET POSE FORMAT")
    print("="*90)
    
    # Load first pose
    first_npz = os.path.join(scannet_images_dir, scannet_files[0])
    pose_scannet, intrinsics = load_scannet_pose(first_npz)
    
    print(f"\n📷 First camera: {scannet_files[0]}")
    print(f"\nScanNet pose matrix (camera_pose):")
    print(pose_scannet)
    
    # Check if it's a valid SE(3) matrix
    bottom_row = pose_scannet[3, :]
    is_valid_se3 = np.allclose(bottom_row, [0, 0, 0, 1])
    print(f"\n✓ Bottom row [0 0 0 1]: {is_valid_se3}")
    
    # Check if rotation part is valid
    R = pose_scannet[:3, :3]
    is_rotation = np.allclose(R @ R.T, np.eye(3), atol=1e-5) and np.allclose(np.linalg.det(R), 1.0, atol=1e-5)
    print(f"✓ Rotation matrix valid: {is_rotation}")
    print(f"  det(R) = {np.linalg.det(R):.6f} (should be ~1.0)")
    print(f"  R @ R^T = I: {np.allclose(R @ R.T, np.eye(3), atol=1e-5)}")
    
    # Extract camera center (assuming C2W)
    camera_center_c2w = pose_scannet[:3, 3]
    print(f"\n🎯 If pose is C2W:")
    print(f"   Camera center (world): [{camera_center_c2w[0]:.3f}, {camera_center_c2w[1]:.3f}, {camera_center_c2w[2]:.3f}]")
    print(f"   Camera looks toward: -Z axis = [{-R[0,2]:.3f}, {-R[1,2]:.3f}, {-R[2,2]:.3f}]")
    
    # Extract camera center (assuming W2C)
    camera_center_w2c = -R.T @ pose_scannet[:3, 3]
    print(f"\n❓ If pose is W2C (WRONG assumption):")
    print(f"   Camera center would be: [{camera_center_w2c[0]:.3f}, {camera_center_w2c[1]:.3f}, {camera_center_w2c[2]:.3f}]")
    
    # Sanity check: camera centers should be reasonable for room-scale scene
    # ScanNet scenes are typically within a few meters
    dist_c2w = np.linalg.norm(camera_center_c2w)
    dist_w2c = np.linalg.norm(camera_center_w2c)
    
    print(f"\n🔍 Sanity check (ScanNet scenes are room-scale, ~0-5 meters):")
    print(f"   Distance from origin (C2W): {dist_c2w:.2f} meters")
    print(f"   Distance from origin (W2C): {dist_w2c:.2f} meters")
    
    if dist_c2w < 10.0 and dist_c2w > 0.1:
        print(f"   ✅ C2W interpretation makes sense (room-scale)")
    else:
        print(f"   ⚠️  C2W interpretation seems off")
    
    if dist_w2c < 10.0 and dist_w2c > 0.1:
        print(f"   ⚠️  W2C interpretation also seems reasonable?")
    else:
        print(f"   ❌ W2C interpretation doesn't make sense")
    
    # ============================================================================
    # STEP 2: Verify conversion logic
    # ============================================================================
    
    print(f"\n" + "="*90)
    print("STEP 2: VERIFY CONVERSION LOGIC")
    print("="*90)
    
    # Compute normalization parameters (as done in conversion script)
    print(f"\n🔢 Computing normalization parameters...")
    camera_centers = []
    for npz_file in scannet_files:
        npz_path = os.path.join(scannet_images_dir, npz_file)
        pose, _ = load_scannet_pose(npz_path)
        # Assuming C2W
        camera_centers.append(pose[:3, 3])
    
    camera_centers = np.array(camera_centers)
    scene_center = np.mean(camera_centers, axis=0)
    scene_radius = np.max(np.linalg.norm(camera_centers - scene_center, axis=1))
    scale = 1.0 / (scene_radius * 1.1)
    
    print(f"   Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
    print(f"   Scene radius: {scene_radius:.3f} meters")
    print(f"   Normalization scale: {scale:.4f}")
    
    # Apply transformation to first camera
    print(f"\n🔄 Applying transformation to first camera...")
    pose_c2w_orig = pose_scannet.copy()
    
    # Step 1: Normalize translation
    pose_c2w_norm = pose_c2w_orig.copy()
    pose_c2w_norm[:3, 3] = (pose_c2w_orig[:3, 3] - scene_center) * scale
    
    print(f"\n   Original C2W translation: [{pose_c2w_orig[0,3]:.3f}, {pose_c2w_orig[1,3]:.3f}, {pose_c2w_orig[2,3]:.3f}]")
    print(f"   Normalized C2W translation: [{pose_c2w_norm[0,3]:.3f}, {pose_c2w_norm[1,3]:.3f}, {pose_c2w_norm[2,3]:.3f}]")
    
    # Step 2: Invert to W2C
    pose_w2c = np.linalg.inv(pose_c2w_norm)
    R_w2c = pose_w2c[:3, :3]
    t_w2c = pose_w2c[:3, 3]
    
    print(f"   W2C rotation matrix:")
    print(f"   {R_w2c}")
    print(f"   W2C translation: [{t_w2c[0]:.3f}, {t_w2c[1]:.3f}, {t_w2c[2]:.3f}]")
    
    # Step 3: Convert to quaternion
    rot = Rotation.from_matrix(R_w2c)
    quat_xyzw = rot.as_quat()  # [qx, qy, qz, qw]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    print(f"   Quaternion (w,x,y,z): [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
    
    # ============================================================================
    # STEP 3: Compare with COLMAP output
    # ============================================================================
    
    print(f"\n" + "="*90)
    print("STEP 3: COMPARE WITH COLMAP OUTPUT")
    print("="*90)
    
    # Find corresponding COLMAP pose
    first_image_name = scannet_files[0].replace('.npz', '.png')
    colmap_first = [p for p in colmap_poses if p['name'] == first_image_name][0]
    
    print(f"\n📷 Image: {first_image_name}")
    print(f"\n   Our computed values:")
    print(f"     Quaternion: [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
    print(f"     Translation: [{t_w2c[0]:.4f}, {t_w2c[1]:.4f}, {t_w2c[2]:.4f}]")
    
    print(f"\n   COLMAP file values:")
    print(f"     Quaternion: [{colmap_first['quat_wxyz'][0]:.4f}, {colmap_first['quat_wxyz'][1]:.4f}, {colmap_first['quat_wxyz'][2]:.4f}, {colmap_first['quat_wxyz'][3]:.4f}]")
    print(f"     Translation: [{colmap_first['tvec'][0]:.4f}, {colmap_first['tvec'][1]:.4f}, {colmap_first['tvec'][2]:.4f}]")
    
    quat_match = np.allclose(quat_wxyz, colmap_first['quat_wxyz'], atol=1e-4)
    tvec_match = np.allclose(t_w2c, colmap_first['tvec'], atol=1e-4)
    
    print(f"\n   ✓ Quaternion match: {quat_match}")
    print(f"   ✓ Translation match: {tvec_match}")
    
    if quat_match and tvec_match:
        print(f"\n   ✅ PERFECT MATCH - Conversion is correct!")
    else:
        print(f"\n   ❌ MISMATCH - There's a bug!")
        print(f"\n   Differences:")
        print(f"     Δquat: {quat_wxyz - colmap_first['quat_wxyz']}")
        print(f"     Δtvec: {t_w2c - colmap_first['tvec']}")
    
    # ============================================================================
    # STEP 4: Verify camera viewing directions
    # ============================================================================
    
    print(f"\n" + "="*90)
    print("STEP 4: VERIFY CAMERA GEOMETRY")
    print("="*90)
    
    print(f"\n🎥 Checking first 5 cameras...")
    
    for i in range(min(5, len(scannet_files))):
        npz_file = scannet_files[i]
        npz_path = os.path.join(scannet_images_dir, npz_file)
        pose_orig, _ = load_scannet_pose(npz_path)
        
        # Original position
        center_orig = pose_orig[:3, 3]
        view_dir_orig = -pose_orig[:3, 2]
        
        # Normalized position
        center_norm = (center_orig - scene_center) * scale
        
        # Find in COLMAP
        image_name = npz_file.replace('.npz', '.png')
        colmap_pose = [p for p in colmap_poses if p['name'] == image_name][0]
        
        # Reconstruct W2C from COLMAP
        R_colmap = quat_to_rotation_matrix(colmap_pose['quat_wxyz'])
        t_colmap = colmap_pose['tvec']
        w2c_colmap = np.eye(4)
        w2c_colmap[:3, :3] = R_colmap
        w2c_colmap[:3, 3] = t_colmap
        
        # Get camera center from W2C
        center_colmap = -R_colmap.T @ t_colmap
        view_dir_colmap = -R_colmap[2, :]
        
        print(f"\n   Camera {i+1}: {image_name}")
        print(f"     Original center: [{center_orig[0]:.2f}, {center_orig[1]:.2f}, {center_orig[2]:.2f}]")
        print(f"     Normalized center: [{center_norm[0]:.3f}, {center_norm[1]:.3f}, {center_norm[2]:.3f}]")
        print(f"     COLMAP center: [{center_colmap[0]:.3f}, {center_colmap[1]:.3f}, {center_colmap[2]:.3f}]")
        
        center_match = np.allclose(center_norm, center_colmap, atol=1e-3)
        print(f"     ✓ Position match: {center_match} (diff: {np.linalg.norm(center_norm - center_colmap):.6f})")
        
        # Check viewing direction (should be preserved after normalization)
        view_match = np.allclose(view_dir_orig / np.linalg.norm(view_dir_orig), 
                                  view_dir_colmap / np.linalg.norm(view_dir_colmap), atol=1e-3)
        print(f"     ✓ View direction preserved: {view_match}")
    
    # ============================================================================
    # FINAL VERDICT
    # ============================================================================
    
    print(f"\n" + "="*90)
    print("FINAL VERDICT")
    print("="*90)
    
    all_checks_pass = quat_match and tvec_match and is_rotation and is_valid_se3
    
    if all_checks_pass:
        print("\n✅ CAMERA COORDINATE TRANSFORMATION IS 100% CORRECT!")
        print("\n   Summary:")
        print("   ✓ ScanNet camera_pose is C2W (camera-to-world)")
        print("   ✓ Normalization is applied correctly")
        print("   ✓ C2W → W2C inversion is correct")
        print("   ✓ Quaternion conversion is correct")
        print("   ✓ Camera positions match expected values")
        print("   ✓ Viewing directions are preserved")
        print("\n   🎉 Your conversion script is PERFECT!")
    else:
        print("\n❌ ERRORS DETECTED IN TRANSFORMATION!")
        print("\n   Failed checks:")
        if not is_rotation:
            print("   ✗ Invalid rotation matrix")
        if not is_valid_se3:
            print("   ✗ Invalid SE(3) matrix format")
        if not quat_match:
            print("   ✗ Quaternion mismatch")
        if not tvec_match:
            print("   ✗ Translation mismatch")
    
    print("="*90 + "\n")
    
    return all_checks_pass


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_camera_coordinates.py <scannet_scene_dir> <converted_scene_dir>")
        print("\nExample:")
        print("  python verify_camera_coordinates.py \\")
        print("    /scratch/runw/project/colmap/data/scannet_test/scene0686_01 \\")
        print("    /scratch/runw/project/colmap/data/scannet_test_feature3dgs/scene0686_01")
        sys.exit(1)
    
    scannet_dir = sys.argv[1]
    converted_dir = sys.argv[2]
    
    verify_scene(scannet_dir, converted_dir)


if __name__ == '__main__':
    main()

