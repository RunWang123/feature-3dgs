#!/usr/bin/env python3
"""
Create a JSON split file for DL3DV based on the ACTUALLY CONVERTED frames.

This ensures the JSON split matches what's in the converted COLMAP data,
avoiding the mismatch that causes test views to use wrong camera poses.
"""

import os
import sys
import json
import argparse

def read_images_txt(images_txt_path):
    """Read COLMAP images.txt and return list of image names (without extension)"""
    images = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = line.split()
            if len(parts) >= 10:  # Valid image line
                try:
                    image_name = parts[9]
                    # Strip extension for comparison
                    name_no_ext = os.path.splitext(image_name)[0]
                    images.append(name_no_ext)
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
    
    return sorted(images)

def create_split(images, train_ratio=0.9, stride=None):
    """
    Create train/test split from list of images.
    
    Args:
        images: List of image names
        train_ratio: Ratio of training images (default: 0.9 for 90% train)
        stride: If set, use stride-based split (e.g., stride=8 means every 8th is test)
    
    Returns:
        train_images, test_images
    """
    if stride is not None:
        # Stride-based split (like LLFF hold)
        train_images = [img for i, img in enumerate(images) if i % stride != 0]
        test_images = [img for i, img in enumerate(images) if i % stride == 0]
    else:
        # Ratio-based split
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        test_images = images[n_train:]
    
    return train_images, test_images

def main():
    parser = argparse.ArgumentParser(
        description="Create DL3DV JSON split from converted COLMAP data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create split with 90% train, 10% test
  python create_dl3dv_split_from_converted.py \\
      --scene /path/to/converted/scene \\
      --scene_name SCENE_HASH \\
      --output split.json
  
  # Use stride-based split (every 8th frame is test, like LLFF)
  python create_dl3dv_split_from_converted.py \\
      --scene /path/to/converted/scene \\
      --scene_name SCENE_HASH \\
      --output split.json \\
      --stride 8
        """
    )
    
    parser.add_argument('--scene', type=str, required=True,
                       help='Path to converted scene directory')
    parser.add_argument('--scene_name', type=str, required=True,
                       help='Scene name (hash) for JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON split file path')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Training ratio (default: 0.9)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Use stride-based split (e.g., 8 for LLFF-style)')
    parser.add_argument('--num_cases', type=int, default=1,
                       help='Number of cases/splits to generate (default: 1)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Creating DL3DV JSON Split from Converted Data")
    print("="*80)
    print(f"Scene: {args.scene}")
    print(f"Scene Name: {args.scene_name}")
    print(f"Output: {args.output}")
    if args.stride:
        print(f"Split method: Stride-based (stride={args.stride})")
    else:
        print(f"Split method: Ratio-based (train={args.train_ratio*100:.0f}%)")
    print()
    
    # Read converted images
    images_txt = os.path.join(args.scene, 'sparse', '0', 'images.txt')
    if not os.path.exists(images_txt):
        print(f"❌ Error: images.txt not found: {images_txt}")
        return 1
    
    images = read_images_txt(images_txt)
    print(f"Found {len(images)} converted images")
    print(f"Sample (first 10): {images[:10]}")
    print()
    
    # Create split(s)
    cases = []
    for case_id in range(args.num_cases):
        train_images, test_images = create_split(images, args.train_ratio, args.stride)
        
        case = {
            "views": [],  # Empty for compatibility
            "ref_views": train_images,
            "target_views": test_images
        }
        cases.append(case)
        
        print(f"Case {case_id}:")
        print(f"  Train views: {len(train_images)}")
        print(f"  Test views:  {len(test_images)}")
        if case_id == 0:
            print(f"  Sample train: {train_images[:5]}")
            print(f"  Sample test: {test_images[:5]}")
    
    # Create JSON structure
    split_data = {
        "config": {
            "case_size": args.num_cases,
            "split_method": "stride" if args.stride else "ratio",
            "stride": args.stride if args.stride else None,
            "train_ratio": args.train_ratio if not args.stride else None,
            "total_images": len(images)
        },
        "scenes": {
            args.scene_name: cases
        }
    }
    
    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print()
    print("="*80)
    print("✅ JSON split created successfully!")
    print(f"   Output: {args.output}")
    print(f"   Total images: {len(images)}")
    print(f"   Train: {len(train_images)}, Test: {len(test_images)}")
    print()
    print("Usage:")
    print(f"  python train.py -s {args.scene} \\")
    print(f"      --json_split_path {args.output} \\")
    print(f"      --case_id 0")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

