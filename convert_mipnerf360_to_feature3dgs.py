#!/usr/bin/env python3
"""
Convert MipNeRF360 dataset to feature-3dgs format with preprocessing.

This script:
1. Reads COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
2. Central crops and resizes images to 448x448
3. Adjusts camera intrinsics accordingly
4. Normalizes camera poses to [-1, 1] range
5. Outputs feature-3dgs compatible structure
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
            binary_point_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            
            # Read track
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            
            points3D[point_id] = (xyz, rgb, error, track)
    
    return points3D

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

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

def write_cameras_text(cameras_data, output_path):
    """Write cameras.txt in COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_data)}\n")
        for cam_id, (model_name, width, height, params) in cameras_data.items():
            params_str = ' '.join(map(str, params))
            f.write(f"{cam_id} {model_name} {width} {height} {params_str}\n")

def write_images_text(images_data, output_path):
    """Write images.txt in COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_data)}\n")
        for img_id, (qvec, tvec, cam_id, name) in images_data.items():
            qw, qx, qy, qz = qvec
            tx, ty, tz = tvec
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n")
            f.write("\n")  # Empty line for 2D points (we don't have them)

def write_points3D_text(points_data, output_path):
    """Write points3D.txt in COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points_data)}\n")
        for pt_id, (xyz, rgb) in points_data.items():
            x, y, z = xyz
            r, g, b = rgb
            f.write(f"{pt_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")

def convert_mipnerf360_scene(scene_path, output_path, target_size=448, use_downscale=None):
    """
    Convert a single MipNeRF360 scene to feature-3dgs format.
    
    Args:
        scene_path: Path to the scene folder (e.g., .../bicycle)
        output_path: Output path for converted scene
        target_size: Target image size (default 448)
        use_downscale: If specified, use images_2, images_4, or images_8 (e.g., 2, 4, 8)
    """
    
    print(f"\n{'='*80}")
    print(f"Converting scene: {os.path.basename(scene_path)}")
    print(f"{'='*80}\n")
    
    # Determine input image folder
    if use_downscale:
        images_dir = os.path.join(scene_path, f"images_{use_downscale}")
        if not os.path.exists(images_dir):
            print(f"Warning: images_{use_downscale} not found, using full resolution images")
            images_dir = os.path.join(scene_path, "images")
    else:
        images_dir = os.path.join(scene_path, "images")
    
    sparse_dir = os.path.join(scene_path, "sparse", "0")
    
    # Check if required files exist
    if not os.path.exists(sparse_dir):
        raise ValueError(f"Sparse reconstruction not found at {sparse_dir}")
    
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points3D_bin = os.path.join(sparse_dir, "points3D.bin")
    
    if not all(os.path.exists(f) for f in [cameras_bin, images_bin, points3D_bin]):
        raise ValueError(f"Missing COLMAP files in {sparse_dir}")
    
    # Read COLMAP data
    print("📖 Reading COLMAP sparse reconstruction...")
    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    points3D = read_points3D_binary(points3D_bin)
    
    print(f"  Cameras: {len(cameras)}")
    print(f"  Images: {len(images)}")
    print(f"  3D Points: {len(points3D)}")
    
    # Create output directories
    output_images_dir = os.path.join(output_path, "images")
    output_sparse_dir = os.path.join(output_path, "sparse", "0")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)
    
    # Process images (no normalization - keep original scale)
    print(f"\n🖼️  Processing images to {target_size}x{target_size}...")
    
    image_list = sorted(images.items(), key=lambda x: x[1][3])  # Sort by name
    
    # Process images and adjust cameras
    new_cameras = {}
    new_images = {}
    
    for img_id, (qvec, tvec, cam_id, name) in tqdm(image_list, desc="Processing images"):
        # Get camera parameters
        model, width, height, params = cameras[cam_id]
        
        # Parse intrinsics based on camera model
        if model == 0:  # SIMPLE_PINHOLE
            f, cx, cy = params
            fx = fy = f
        elif model == 1:  # PINHOLE
            fx, fy, cx, cy = params
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        
        # Process image
        input_image_path = os.path.join(images_dir, name)
        output_image_path = os.path.join(output_images_dir, name)
        
        if not os.path.exists(input_image_path):
            print(f"  Warning: Image not found: {input_image_path}")
            continue
        
        crop_left, crop_top, crop_size = preprocess_image(
            input_image_path, output_image_path, target_size
        )
        
        # Adjust intrinsics
        (crop_left, crop_top, crop_size), scale = compute_camera_transform(width, height, target_size)
        fx_new, fy_new, cx_new, cy_new = adjust_intrinsics(fx, fy, cx, cy, crop_left, crop_top, crop_size, scale)
        
        # Store adjusted camera (use PINHOLE model)
        if cam_id not in new_cameras:
            new_cameras[cam_id] = ("PINHOLE", target_size, target_size, [fx_new, fy_new, cx_new, cy_new])
        
        # Keep original pose (no normalization)
        new_images[img_id] = (qvec, tvec, cam_id, name)
    
    # Keep 3D points at original scale (no normalization)
    print(f"\n📍 Processing {len(points3D)} 3D points...")
    new_points3D = {}
    for pt_id, (xyz, rgb, error, track) in points3D.items():
        new_points3D[pt_id] = (xyz, rgb)  # Keep original coordinates
    
    # Write COLMAP text files
    print(f"\n💾 Writing COLMAP files...")
    write_cameras_text(new_cameras, os.path.join(output_sparse_dir, "cameras.txt"))
    write_images_text(new_images, os.path.join(output_sparse_dir, "images.txt"))
    write_points3D_text(new_points3D, os.path.join(output_sparse_dir, "points3D.txt"))
    
    print(f"\n✅ Conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Images: {len(new_images)}")
    print(f"  3D Points: {len(new_points3D)} (original scale)")


def main():
    parser = argparse.ArgumentParser(description="Convert MipNeRF360 dataset to feature-3dgs format")
    parser.add_argument("--scene_path", type=str, required=True,
                      help="Path to MipNeRF360 scene (e.g., /path/to/bicycle)")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Output path for converted scene")
    parser.add_argument("--target_size", type=int, default=448,
                      help="Target image size (default: 448)")
    parser.add_argument("--use_downscale", type=int, choices=[2, 4, 8], default=None,
                      help="Use downscaled images (2, 4, or 8). If not set, uses full resolution.")
    
    args = parser.parse_args()
    
    convert_mipnerf360_scene(
        scene_path=args.scene_path,
        output_path=args.output_path,
        target_size=args.target_size,
        use_downscale=args.use_downscale
    )


if __name__ == "__main__":
    main()

