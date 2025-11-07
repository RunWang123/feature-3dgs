#!/usr/bin/env python3
"""
Convert DL3DV dataset to feature-3dgs format with preprocessing.

This script:
1. Reads COLMAP sparse reconstruction from gaussian_splat/sparse/
2. Uses images from gaussian_splat/images_4/
3. Central crops and resizes images to 448x448
4. Adjusts camera intrinsics accordingly
5. Normalizes camera poses to [-1, 1] range
6. Outputs feature-3dgs compatible structure
"""

import os
import sys
import argparse
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import struct

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP cameras.bin file.
    Returns dict: camera_id -> (model, width, height, params)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            # Read intrinsic parameters
            if model == 0 or model == 1:  # SIMPLE_PINHOLE or PINHOLE
                num_params = 3 if model == 0 else 4
            elif model == 2:  # SIMPLE_RADIAL
                num_params = 4
            elif model == 3:  # RADIAL
                num_params = 5
            elif model == 4:  # OPENCV
                num_params = 8
            elif model == 5:  # OPENCV_FISHEYE
                num_params = 8
            else:
                raise ValueError(f"Unknown camera model: {model}")
            
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = (model, width, height, params)
    
    return cameras

def read_images_binary(path_to_model_file):
    """
    Read COLMAP images.bin file.
    Returns dict: image_id -> (qvec, tvec, camera_id, name)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Read 2D points (skip them)
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            
            images[image_id] = (qvec, tvec, camera_id, image_name)
    
    return images

def read_points3D_binary(path_to_model_file):
    """
    Read COLMAP points3D.bin file.
    Returns dict: point_id -> (xyz, rgb, error, track)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            
            # Read track length and track elements
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            
            points3D[point3D_id] = (xyz, rgb, error, track_elems)
    
    return points3D

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

def compute_camera_transform(orig_width, orig_height, target_size=448):
    """
    Compute the crop and scale transformation for central crop + resize.
    Returns: (crop_left, crop_top, crop_size), scale
    """
    crop_size = min(orig_width, orig_height)
    crop_left = (orig_width - crop_size) // 2
    crop_top = (orig_height - crop_size) // 2
    scale = target_size / crop_size
    return (crop_left, crop_top, crop_size), scale

def adjust_intrinsics(fx, fy, cx, cy, crop_left, crop_top, crop_size, scale):
    """
    Adjust camera intrinsics for central crop and resize.
    """
    # Adjust principal point for crop
    cx_crop = cx - crop_left
    cy_crop = cy - crop_top
    
    # Scale all parameters
    fx_final = fx * scale
    fy_final = fy * scale
    cx_final = cx_crop * scale
    cy_final = cy_crop * scale
    
    return fx_final, fy_final, cx_final, cy_final

def preprocess_image(input_path, output_path, target_size=448):
    """
    Central crop and resize image to target_size x target_size.
    """
    img = Image.open(input_path)
    orig_width, orig_height = img.size
    
    # Central crop to square
    crop_size = min(orig_width, orig_height)
    crop_left = (orig_width - crop_size) // 2
    crop_top = (orig_height - crop_size) // 2
    
    img_crop = img.crop((crop_left, crop_top, crop_left + crop_size, crop_top + crop_size))
    
    # Resize to target size
    img_resized = img_crop.resize((target_size, target_size), Image.LANCZOS)
    
    # Save
    img_resized.save(output_path, quality=95)
    
    return crop_left, crop_top, crop_size

def write_cameras_txt(output_dir, fx, fy, cx, cy, size):
    """Write COLMAP cameras.txt"""
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {size} {size} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

def write_images_txt(output_dir, image_data):
    """Write COLMAP images.txt"""
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_data)}, mean observations per image: 0\n")
        
        for img in image_data:
            qvec = img['qvec']
            tvec = img['tvec']
            f.write(f"{img['image_id']} {qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                   f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} {img['camera_id']} {img['image_name']}\n")
            f.write("\n")

def write_points3d_txt(output_dir):
    """Write empty COLMAP points3D.txt for random initialization"""
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")

def convert_dl3dv_scene(scene_path, output_path, target_size=448):
    """
    Convert a single DL3DV scene to feature-3dgs format.
    
    Args:
        scene_path: Path to DL3DV scene (hash folder containing gaussian_splat/)
        output_path: Output directory for feature-3dgs format
        target_size: Target square image size (default: 448)
    """
    scene_name = os.path.basename(scene_path)
    
    print(f"\n{'='*70}")
    print(f"Processing DL3DV scene: {scene_name}")
    print(f"{'='*70}\n")
    
    # DL3DV structure: scene_hash/gaussian_splat/
    gaussian_splat_dir = os.path.join(scene_path, 'gaussian_splat')
    if not os.path.exists(gaussian_splat_dir):
        print(f"❌ Error: gaussian_splat directory not found in {scene_path}")
        return False
    
    images_dir = os.path.join(gaussian_splat_dir, 'images_4')
    sparse_dir = os.path.join(gaussian_splat_dir, 'sparse')
    
    # Check for sparse/0 subdirectory (COLMAP convention)
    if os.path.exists(os.path.join(sparse_dir, '0')):
        sparse_dir = os.path.join(sparse_dir, '0')
    
    if not os.path.exists(images_dir):
        print(f"❌ Error: images_4 directory not found")
        return False
    
    if not os.path.exists(sparse_dir):
        print(f"❌ Error: sparse directory not found")
        return False
    
    # Read COLMAP data
    cameras_bin = os.path.join(sparse_dir, 'cameras.bin')
    images_bin = os.path.join(sparse_dir, 'images.bin')
    points3d_bin = os.path.join(sparse_dir, 'points3D.bin')
    
    if not all(os.path.exists(f) for f in [cameras_bin, images_bin]):
        print(f"❌ Error: COLMAP binary files not found")
        return False
    
    print("Reading COLMAP data...")
    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    
    if len(cameras) == 0 or len(images) == 0:
        print(f"❌ Error: No cameras or images in COLMAP reconstruction")
        return False
    
    print(f"  Cameras: {len(cameras)}")
    print(f"  Images: {len(images)}")
    
    # Create output directories
    output_images_dir = os.path.join(output_path, 'images')
    output_sparse_dir = os.path.join(output_path, 'sparse', '0')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)
    
    # Get first camera to determine original size
    first_camera_id = list(cameras.keys())[0]
    model, orig_width, orig_height, params = cameras[first_camera_id]
    
    print(f"\nOriginal image size: {orig_width}x{orig_height}")
    
    # Extract intrinsics based on camera model
    if model == 0:  # SIMPLE_PINHOLE
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model == 1:  # PINHOLE
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif model == 2:  # SIMPLE_RADIAL
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        print(f"⚠️  Warning: SIMPLE_RADIAL model detected, converting to PINHOLE")
    else:
        print(f"❌ Error: Unsupported camera model {model}")
        return False
    
    print(f"Original intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    # Compute transformation
    crop_params, scale = compute_camera_transform(orig_width, orig_height, target_size)
    crop_left, crop_top, crop_size = crop_params
    
    print(f"\nTransformation:")
    print(f"  1. Central crop: {crop_size}x{crop_size} (left={crop_left}, top={crop_top})")
    print(f"  2. Resize: {target_size}x{target_size} (scale={scale:.4f})")
    
    # Adjust intrinsics
    fx_new, fy_new, cx_new, cy_new = adjust_intrinsics(
        fx, fy, cx, cy,
        crop_left, crop_top, crop_size, scale
    )
    
    print(f"\nAdjusted intrinsics: fx={fx_new:.2f}, fy={fy_new:.2f}, cx={cx_new:.2f}, cy={cy_new:.2f}")
    
    # Preprocess images
    print(f"\nPreprocessing {len(images)} images...")
    available_images = {}
    for image_id, (qvec, tvec, camera_id, image_name) in tqdm(images.items()):
        input_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Warning: Image not found: {image_name}")
            continue
        
        # Keep original extension or convert to PNG
        output_name = os.path.splitext(image_name)[0] + '.png'
        output_path = os.path.join(output_images_dir, output_name)
        
        try:
            preprocess_image(input_path, output_path, target_size)
            available_images[image_id] = (qvec, tvec, camera_id, output_name)
        except Exception as e:
            print(f"⚠️  Error processing {image_name}: {e}")
    
    if len(available_images) == 0:
        print(f"❌ Error: No images could be processed")
        return False
    
    print(f"\n✓ Processed {len(available_images)} images")
    
    # Compute scene normalization
    print("\nComputing scene normalization...")
    camera_centers = []
    for image_id in available_images:
        qvec, tvec, _, _ = available_images[image_id]
        R = qvec2rotmat(qvec)
        t = tvec
        # Camera center in world coordinates: -R^T * t
        center = -R.T @ t
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    scene_center = np.mean(camera_centers, axis=0)
    scene_radius = np.max(np.linalg.norm(camera_centers - scene_center, axis=1))
    scale_factor = 1.0 / (scene_radius * 1.1)  # Scale to fit in [-1, 1] range
    
    print(f"  Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
    print(f"  Scene radius: {scene_radius:.3f}")
    print(f"  Scale factor: {scale_factor:.6f}")
    
    # Normalize poses and write COLMAP files
    print("\nNormalizing camera poses...")
    image_data = []
    for idx, image_id in enumerate(sorted(available_images.keys())):
        qvec, tvec, camera_id, image_name = available_images[image_id]
        
        # Convert to world-to-camera with normalization
        R_w2c = qvec2rotmat(qvec)
        t_w2c = tvec
        
        # Convert to camera-to-world for normalization
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        
        # Normalize
        t_c2w_normalized = (t_c2w - scene_center) * scale_factor
        
        # Convert back to world-to-camera
        t_w2c_normalized = -R_c2w.T @ t_c2w_normalized
        
        # Convert rotation back to quaternion
        qvec_new = rotmat2qvec(R_w2c)
        
        image_data.append({
            'image_id': idx + 1,
            'qvec': qvec_new,
            'tvec': t_w2c_normalized,
            'camera_id': 1,
            'image_name': image_name
        })
    
    # Write COLMAP files
    write_cameras_txt(output_sparse_dir, fx_new, fy_new, cx_new, cy_new, target_size)
    write_images_txt(output_sparse_dir, image_data)
    write_points3d_txt(output_sparse_dir)
    
    print(f"\n✅ Scene processed successfully!")
    print(f"   Output: {output_path}")
    print(f"   Images: {len(image_data)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert DL3DV scene to feature-3dgs format")
    parser.add_argument('--scene', '-s', type=str, required=True,
                       help='Path to DL3DV scene directory (hash folder)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for feature-3dgs format')
    parser.add_argument('--size', type=int, default=448,
                       help='Target square size (default: 448)')
    
    args = parser.parse_args()
    
    success = convert_dl3dv_scene(args.scene, args.output, args.size)
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()

