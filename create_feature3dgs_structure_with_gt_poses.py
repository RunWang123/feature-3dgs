#!/usr/bin/env python3
"""
Create feature-3dgs compatible dataset with ScanNet ground truth poses.
Replicates the exact structure of scannet_test_feature3dgs:
- Only first 30 images (000000, 000010, 000020, ..., 000290)
- 448×448 PNG format
- All 131 labels
- Ground truth camera poses from ScanNet (not COLMAP reconstruction)
"""

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


def compute_camera_transform(orig_width, orig_height, target_size=448):
    """Compute crop and resize parameters."""
    crop_size = min(orig_width, orig_height)
    crop_left = (orig_width - crop_size) // 2
    crop_top = (orig_height - crop_size) // 2
    scale = target_size / crop_size
    return (crop_left, crop_top, crop_size), scale


def adjust_intrinsics(fx, fy, cx, cy, crop_left, crop_top, crop_size, scale):
    """Adjust camera intrinsics after crop and resize."""
    cx_crop = cx - crop_left
    cy_crop = cy - crop_top
    fx_crop = fx
    fy_crop = fy
    
    fx_final = fx_crop * scale
    fy_final = fy_crop * scale
    cx_final = cx_crop * scale
    cy_final = cy_crop * scale
    
    return fx_final, fy_final, cx_final, cy_final


def preprocess_image(image_path, output_path, crop_params, target_size=448):
    """Central crop and resize image to PNG."""
    crop_left, crop_top, crop_size = crop_params
    
    img = Image.open(image_path)
    img_cropped = img.crop((crop_left, crop_top, crop_left + crop_size, crop_top + crop_size))
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)
    img_resized.save(output_path, 'PNG')
    
    return img_resized


def process_scene(scene_path, output_path, target_size=448, num_images=30):
    """
    Process one ScanNet scene to match scannet_test_feature3dgs structure.
    """
    scene_name = os.path.basename(scene_path)
    images_dir = os.path.join(scene_path, 'images')
    labels_dir = os.path.join(scene_path, 'labels')
    
    print(f"\n{'='*70}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*70}\n")
    
    # Create output directories matching feature3dgs structure
    output_images_dir = os.path.join(output_path, 'images')
    output_labels_dir = os.path.join(output_path, 'labels')
    output_sparse_dir = os.path.join(output_path, 'sparse', '0')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)
    
    # Get all NPZ files and sort
    all_npz_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npz')])
    
    if len(all_npz_files) == 0:
        print(f"❌ No NPZ files found")
        return False
    
    # Select first 30 frames only
    selected_npz_files = all_npz_files[:num_images]
    
    print(f"Total frames available: {len(all_npz_files)}")
    print(f"Processing first {len(selected_npz_files)} frames")
    print(f"Frame range: {selected_npz_files[0]} to {selected_npz_files[-1]}")
    
    # Read first NPZ to get camera parameters
    first_npz = os.path.join(images_dir, selected_npz_files[0])
    data = np.load(first_npz)
    intrinsics = data['camera_intrinsics']
    
    fx_orig = intrinsics[0, 0]
    fy_orig = intrinsics[1, 1]
    cx_orig = intrinsics[0, 2]
    cy_orig = intrinsics[1, 2]
    
    # Get original image size
    first_jpg = os.path.join(images_dir, selected_npz_files[0].replace('.npz', '.jpg'))
    img = Image.open(first_jpg)
    orig_width, orig_height = img.size
    
    print(f"\nOriginal image size: {orig_width}×{orig_height}")
    print(f"Original intrinsics: fx={fx_orig:.2f}, fy={fy_orig:.2f}, cx={cx_orig:.2f}, cy={cy_orig:.2f}")
    
    # Compute transformation
    crop_params, scale = compute_camera_transform(orig_width, orig_height, target_size)
    crop_left, crop_top, crop_size = crop_params
    
    print(f"\nTransformation:")
    print(f"  1. Central crop: {crop_size}×{crop_size} (left={crop_left}, top={crop_top})")
    print(f"  2. Resize: {target_size}×{target_size} (scale={scale:.4f})")
    
    # Adjust intrinsics
    fx_new, fy_new, cx_new, cy_new = adjust_intrinsics(
        fx_orig, fy_orig, cx_orig, cy_orig,
        crop_left, crop_top, crop_size, scale
    )
    
    print(f"\nAdjusted intrinsics: fx={fx_new:.2f}, fy={fy_new:.2f}, cx={cx_new:.2f}, cy={cy_new:.2f}")
    
    # Process first 30 images only
    print(f"\nProcessing {len(selected_npz_files)} images to PNG format...")
    for npz_file in tqdm(selected_npz_files):
        jpg_file = npz_file.replace('.npz', '.jpg')
        png_file = npz_file.replace('.npz', '.png')
        
        input_jpg = os.path.join(images_dir, jpg_file)
        output_png = os.path.join(output_images_dir, png_file)
        
        if os.path.exists(input_jpg):
            preprocess_image(input_jpg, output_png, crop_params, target_size)
    
    # Copy ALL labels (131 labels, not just 30)
    print(f"\nCopying labels folder...")
    if os.path.exists(labels_dir):
        # Copy all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
        for label_file in tqdm(label_files):
            src = os.path.join(labels_dir, label_file)
            dst = os.path.join(output_labels_dir, label_file)
            shutil.copy2(src, dst)
        print(f"  Copied {len(label_files)} label files")
    else:
        print(f"  ⚠️  Warning: Labels folder not found at {labels_dir}")
    
    # Convert camera poses (only for the 30 selected images)
    print(f"\nConverting camera poses to COLMAP format...")
    image_data = []
    camera_id = 1
    
    for idx, npz_file in enumerate(tqdm(selected_npz_files)):
        npz_path = os.path.join(images_dir, npz_file)
        image_name = npz_file.replace('.npz', '.png')
        
        # Read camera data
        data = np.load(npz_path)
        pose_c2w = data['camera_pose']
        
        # Convert C2W to W2C
        pose_w2c = np.linalg.inv(pose_c2w)
        R_w2c = pose_w2c[:3, :3]
        t_w2c = pose_w2c[:3, 3]
        
        # Convert rotation to quaternion (COLMAP format: qw, qx, qy, qz)
        rotation = Rotation.from_matrix(R_w2c)
        qvec_scipy = rotation.as_quat()  # [qx, qy, qz, qw]
        qvec = [qvec_scipy[3], qvec_scipy[0], qvec_scipy[1], qvec_scipy[2]]
        
        image_data.append({
            'image_id': idx + 1,
            'qvec': qvec,
            'tvec': t_w2c,
            'camera_id': camera_id,
            'image_name': image_name
        })
    
    # Write COLMAP files
    write_cameras_txt(output_sparse_dir, fx_new, fy_new, cx_new, cy_new, target_size)
    write_images_txt(output_sparse_dir, image_data)
    write_points3d_txt(output_sparse_dir)
    
    # Generate binary files (optional but matches feature3dgs structure)
    print(f"\nGenerating COLMAP binary files...")
    generate_colmap_binary(output_sparse_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Scene processed successfully!")
    print(f"{'='*70}")
    print(f"Output structure:")
    print(f"  {output_path}/")
    print(f"    ├── images/         ({len(selected_npz_files)} PNG files, 448×448)")
    if os.path.exists(output_labels_dir):
        label_count = len([f for f in os.listdir(output_labels_dir) if f.endswith('.png')])
        print(f"    ├── labels/         ({label_count} PNG files)")
    print(f"    └── sparse/0/       (cameras.txt, images.txt, points3D.txt + .bin)")
    print(f"{'='*70}\n")
    
    return True


def write_cameras_txt(output_dir, fx, fy, cx, cy, size):
    """Write COLMAP cameras.txt"""
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {size} {size} {fx} {fy} {cx} {cy}\n")


def write_images_txt(output_dir, image_data):
    """Write COLMAP images.txt"""
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_data)}, mean observations per image: 0\n")
        
        for img in image_data:
            f.write(f"{img['image_id']} {img['qvec'][0]} {img['qvec'][1]} {img['qvec'][2]} {img['qvec'][3]} "
                   f"{img['tvec'][0]} {img['tvec'][1]} {img['tvec'][2]} {img['camera_id']} {img['image_name']}\n")
            f.write("\n")


def write_points3d_txt(output_dir):
    """Write empty COLMAP points3D.txt"""
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


def generate_colmap_binary(sparse_dir):
    """Generate COLMAP binary files from text files using colmap command."""
    try:
        import subprocess
        cmd = [
            'colmap', 'model_converter',
            '--input_path', sparse_dir,
            '--output_path', sparse_dir,
            '--output_type', 'BIN'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Binary files generated successfully")
        else:
            print("  ⚠️  Could not generate binary files (colmap not available)")
            print("     Text files are sufficient for feature-3dgs")
    except Exception as e:
        print(f"  ⚠️  Could not generate binary files: {e}")
        print("     Text files are sufficient for feature-3dgs")


def main():
    parser = argparse.ArgumentParser(
        description="Create feature-3dgs structure with ScanNet GT poses"
    )
    parser.add_argument('--scene', '-s', type=str, required=True,
                       help='Path to ScanNet scene directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--size', type=int, default=448,
                       help='Target square size (default: 448)')
    parser.add_argument('--num_images', type=int, default=30,
                       help='Number of images to process (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.scene):
        print(f"❌ Error: Scene directory not found: {args.scene}")
        return
    
    process_scene(args.scene, args.output, args.size, args.num_images)


if __name__ == '__main__':
    main()

