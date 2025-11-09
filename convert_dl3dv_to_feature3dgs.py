#!/usr/bin/env python3
"""
Convert DL3DV dataset to feature-3dgs format with preprocessing.

This script:
1. Reads camera intrinsics from COLMAP sparse/0/cameras.bin (matches 3D points!)
2. Reads camera poses from transforms.json (C2W format, OpenGL/NeRF convention)
3. Uses images from gaussian_splat/images_4/
4. Central crops and resizes images to 448x448
5. Adjusts camera intrinsics accordingly
6. Flips Y/Z axes to convert from OpenGL (Y up, Z back) to COLMAP (Y down, Z forward)
7. Converts poses from C2W to W2C (no normalization - keeps original scale)
8. Outputs feature-3dgs compatible structure with aligned 3D points
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

def undistort_image(img_array, K, dist_coeffs, new_K, new_size):
    """
    Undistort image using OpenCV camera model.
    
    Args:
        img_array: Input image as numpy array
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        new_K: New camera matrix after undistortion
        new_size: Output image size (width, height)
    
    Returns:
        Undistorted image as numpy array
    """
    import cv2
    
    # Undistort
    img_undistorted = cv2.undistort(img_array, K, dist_coeffs, None, new_K)
    
    # Crop to new size if needed
    h, w = img_undistorted.shape[:2]
    new_w, new_h = new_size
    if w != new_w or h != new_h:
        # Center crop
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        img_undistorted = img_undistorted[start_y:start_y+new_h, start_x:start_x+new_w]
    
    return img_undistorted

def preprocess_image(input_path, output_path, target_size=448, undistort_params=None):
    """
    Preprocess image: optionally undistort, then central crop and resize to target_size x target_size.
    
    Args:
        input_path: Input image path
        output_path: Output image path
        target_size: Target square size
        undistort_params: Dict with 'K', 'dist_coeffs', 'new_K', 'new_size' for undistortion, or None
    """
    import cv2
    
    # Read image
    img = Image.open(input_path)
    
    # Apply undistortion if parameters provided
    if undistort_params is not None:
        img_array = np.array(img)
        img_array = undistort_image(
            img_array,
            undistort_params['K'],
            undistort_params['dist_coeffs'],
            undistort_params['new_K'],
            undistort_params['new_size']
        )
        img = Image.fromarray(img_array)
    
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

def write_points3d_txt(output_dir, points3d_data=None):
    """
    Write COLMAP points3D.txt
    
    Args:
        output_dir: Output directory
        points3d_data: List of dicts with keys: point_id, xyz, rgb
    """
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        if points3d_data is None or len(points3d_data) == 0:
            f.write("# Number of points: 0, mean track length: 0\n")
        else:
            f.write(f"# Number of points: {len(points3d_data)}, mean track length: 0\n")
            for pt in points3d_data:
                xyz = pt['xyz']
                rgb = pt['rgb']
                f.write(f"{pt['point_id']} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} "
                       f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.0\n")

def write_points3d_ply(output_dir, points3d_data=None):
    """
    Write points3D.ply file (required by feature-3dgs)
    
    Args:
        output_dir: Output directory
        points3d_data: List of dicts with keys: point_id, xyz, rgb
    """
    from plyfile import PlyData, PlyElement
    
    ply_path = os.path.join(output_dir, 'points3D.ply')
    
    if points3d_data is None or len(points3d_data) == 0:
        # Write empty PLY
        xyz = np.zeros((0, 3))
        rgb = np.zeros((0, 3))
    else:
        xyz = np.array([pt['xyz'] for pt in points3d_data])
        rgb = np.array([pt['rgb'] for pt in points3d_data])
    
    # Create PLY format
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # Create vertex element
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

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
    transforms_json = os.path.join(gaussian_splat_dir, 'transforms.json')
    sparse_dir = os.path.join(gaussian_splat_dir, 'sparse', '0')
    
    if not os.path.exists(images_dir):
        print(f"❌ Error: images_4 directory not found")
        return False
    
    if not os.path.exists(transforms_json):
        print(f"❌ Error: transforms.json not found")
        return False
    
    # Read COLMAP cameras.bin for intrinsics (these match the 3D points!)
    print("Reading COLMAP cameras...")
    cameras_bin = os.path.join(sparse_dir, 'cameras.bin')
    if not os.path.exists(cameras_bin):
        print(f"❌ Error: cameras.bin not found")
        return False
    
    cameras = read_cameras_binary(cameras_bin)
    if len(cameras) == 0:
        print(f"❌ Error: No cameras in cameras.bin")
        return False
    
    # Read transforms.json first (need it to find first image)
    print("Reading transforms.json for poses...")
    with open(transforms_json, 'r') as f:
        transforms = json.load(f)
    
    frames = transforms['frames']
    if len(frames) == 0:
        print(f"❌ Error: No frames in transforms.json")
        return False
    
    # Get camera intrinsics from COLMAP (this is what was used for 3D reconstruction)
    cam_id = list(cameras.keys())[0]
    model, colmap_width, colmap_height, params = cameras[cam_id]
    
    if model == 0:  # SIMPLE_PINHOLE
        fx_colmap = fy_colmap = params[0]
        cx_colmap, cy_colmap = params[1], params[2]
    elif model == 1:  # PINHOLE
        fx_colmap, fy_colmap = params[0], params[1]
        cx_colmap, cy_colmap = params[2], params[3]
    elif model == 2:  # SIMPLE_RADIAL
        fx_colmap = fy_colmap = params[0]
        cx_colmap, cy_colmap = params[1], params[2]
        print(f"⚠️  Warning: SIMPLE_RADIAL model, converting to PINHOLE (ignoring distortion)")
    else:
        print(f"❌ Error: Unsupported camera model {model}")
        return False
    
    # Check if images_4/ are distorted or undistorted
    # Get actual image size from images_4/
    first_frame = frames[0]
    first_image_path = os.path.join(images_dir, os.path.basename(first_frame['file_path']))
    if not os.path.exists(first_image_path):
        print(f"❌ Error: First image not found: {first_image_path}")
        return False
    
    from PIL import Image as PILImage
    with PILImage.open(first_image_path) as img:
        distorted_width, distorted_height = img.size
    
    print(f"  COLMAP reconstructed size: {colmap_width}x{colmap_height}")
    print(f"  images_4/ size: {distorted_width}x{distorted_height}")
    
    # Determine if we need undistortion
    # If images_4/ size matches transforms.json (3840x2160 / 4 = 960x540), they are DISTORTED
    # If they match COLMAP size (~3819x2147 / 4 = 954x536), they are UNDISTORTED
    transforms_width = transforms['w']
    transforms_height = transforms['h']
    expected_distorted_width = transforms_width // 4
    expected_distorted_height = transforms_height // 4
    
    need_undistortion = (abs(distorted_width - expected_distorted_width) < 5 and 
                         abs(distorted_height - expected_distorted_height) < 5)
    
    # Check if COLMAP was run on undistorted images
    # If COLMAP size != transforms.json size, COLMAP used undistorted images
    colmap_matches_transforms = (abs(colmap_width - transforms_width) < 50 and
                                  abs(colmap_height - transforms_height) < 50)
    
    if not colmap_matches_transforms:
        print(f"  ⚠️  COLMAP size ({colmap_width}x{colmap_height}) != transforms.json size ({transforms_width}x{transforms_height})")
        print(f"  → COLMAP used undistorted images - SKIPPING undistortion!")
        need_undistortion = False
    
    if need_undistortion:
        print(f"  ⚠️  images_4/ are DISTORTED ({distorted_width}x{distorted_height})")
        print(f"  Will undistort using OPENCV model from transforms.json")
        
        # Prepare undistortion parameters scaled to images_4/ resolution
        scale_factor = distorted_width / transforms_width
        
        # Distorted camera matrix for images_4/
        K_dist = np.array([
            [transforms['fl_x'] * scale_factor, 0, transforms['cx'] * scale_factor],
            [0, transforms['fl_y'] * scale_factor, transforms['cy'] * scale_factor],
            [0, 0, 1]
        ])
        
        # Distortion coefficients
        dist_coeffs = np.array([
            transforms.get('k1', 0),
            transforms.get('k2', 0),
            transforms.get('p1', 0),
            transforms.get('p2', 0),
            transforms.get('k3', 0) if 'k3' in transforms else 0
        ])
        
        # New camera matrix after undistortion (COLMAP PINHOLE intrinsics scaled to images_4/)
        downscale_factor = colmap_width / distorted_width
        K_undist = np.array([
            [fx_colmap / downscale_factor, 0, cx_colmap / downscale_factor],
            [0, fy_colmap / downscale_factor, cy_colmap / downscale_factor],
            [0, 0, 1]
        ])
        
        # Undistorted size (should match COLMAP)
        undistorted_width = int(colmap_width / downscale_factor)
        undistorted_height = int(colmap_height / downscale_factor)
        
        undistort_params = {
            'K': K_dist,
            'dist_coeffs': dist_coeffs,
            'new_K': K_undist,
            'new_size': (undistorted_width, undistorted_height)
        }
        
        # After undistortion, images will be undistorted_width x undistorted_height
        fx = fx_colmap / downscale_factor
        fy = fy_colmap / downscale_factor
        cx = cx_colmap / downscale_factor
        cy = cy_colmap / downscale_factor
        orig_width = undistorted_width
        orig_height = undistorted_height
        
        print(f"  Undistorted size: {undistorted_width}x{undistorted_height}")
        print(f"  Undistorted intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    else:
        print(f"  ✓ Skipping undistortion - using images_4/ as-is")
        undistort_params = None
        
        # Use transforms.json intrinsics scaled to images_4/ size
        # (images are in distorted coordinate system, not COLMAP's undistorted system)
        scale_factor = distorted_width / transforms_width
        fx = transforms['fl_x'] * scale_factor
        fy = transforms['fl_y'] * scale_factor
        cx = transforms['cx'] * scale_factor
        cy = transforms['cy'] * scale_factor
        orig_width = distorted_width
        orig_height = distorted_height
        
        print(f"  Using transforms.json intrinsics (distorted camera model)")
    
    print(f"  Frames: {len(frames)}")
    print(f"  Scaled intrinsics (for {orig_width}x{orig_height}): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    # Create output directories
    output_images_dir = os.path.join(output_path, 'images')
    output_sparse_dir = os.path.join(output_path, 'sparse', '0')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)
    
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
    
    # Preprocess images and extract poses
    print(f"\nPreprocessing {len(frames)} images...")
    available_frames = []
    for frame_idx, frame in enumerate(tqdm(frames)):
        # Fix file path: JSON says "images/" but actual folder is "images_4/"
        file_path = frame['file_path']
        image_name = os.path.basename(file_path)
        input_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Warning: Image not found: {image_name}")
            continue
        
        # Output as PNG
        output_name = os.path.splitext(image_name)[0] + '.png'
        output_path = os.path.join(output_images_dir, output_name)
        
        try:
            preprocess_image(input_path, output_path, target_size, undistort_params)
            
            # Extract transform matrix (C2W format in NeRF convention)
            transform_matrix = np.array(frame['transform_matrix'])
            
            available_frames.append({
                'output_name': output_name,
                'transform_matrix': transform_matrix
            })
        except Exception as e:
            print(f"⚠️  Error processing {image_name}: {e}")
    
    if len(available_frames) == 0:
        print(f"❌ Error: No images could be processed")
        return False
    
    print(f"\n✓ Processed {len(available_frames)} images")
    
    # Convert poses from C2W to W2C format (no normalization - keep original scale)
    print("\nConverting camera poses to COLMAP format...")
    image_data = []
    for idx, frame in enumerate(available_frames):
        # IMPORTANT: Make a copy to avoid modifying the original array
        c2w = np.array(frame['transform_matrix'], copy=True)
        output_name = frame['output_name']
        
        # ⚠️ CRITICAL: Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # DL3DV transforms.json uses NeRF/OpenGL convention (generated by colmap2nerf.py)
        # We need to flip Y and Z axes to convert back to COLMAP/OpenCV convention
        c2w[:3, 1:3] *= -1
        
        # Convert C2W to W2C (COLMAP format) - keep original scale
        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        
        # Convert rotation to quaternion
        qvec = rotmat2qvec(R_w2c)
        
        image_data.append({
            'image_id': idx + 1,
            'qvec': qvec,
            'tvec': t_w2c,
            'camera_id': 1,
            'image_name': output_name
        })
    
    # Read 3D points (keep original scale - no normalization)
    print("\nProcessing 3D points...")
    points3d_data = None
    points3d_bin = os.path.join(sparse_dir, 'points3D.bin')
    
    if os.path.exists(points3d_bin):
        try:
            points3d = read_points3D_binary(points3d_bin)
            print(f"  Found {len(points3d)} 3D points")
            
            # Keep 3D points at original scale (no normalization)
            points3d_data = []
            for point_id, (xyz, rgb, error, track) in points3d.items():
                points3d_data.append({
                    'point_id': point_id,
                    'xyz': xyz,  # Keep original coordinates
                    'rgb': rgb
                })
            
            print(f"  Loaded {len(points3d_data)} 3D points (original scale)")
        except Exception as e:
            print(f"⚠️  Warning: Could not read 3D points: {e}")
            print("  Will use random initialization instead")
    else:
        print("  No 3D points found, will use random initialization")
    
    # Write COLMAP files
    write_cameras_txt(output_sparse_dir, fx_new, fy_new, cx_new, cy_new, target_size)
    write_images_txt(output_sparse_dir, image_data)
    write_points3d_txt(output_sparse_dir, points3d_data)
    write_points3d_ply(output_sparse_dir, points3d_data)  # feature-3dgs expects PLY format
    
    print(f"\n✅ Scene processed successfully!")
    print(f"   Output: {output_path}")
    print(f"   Images: {len(image_data)}")
    if points3d_data:
        print(f"   3D Points: {len(points3d_data)}")
    else:
        print(f"   3D Points: 0 (will use random initialization)")
    
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

