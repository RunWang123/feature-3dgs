#!/usr/bin/env python3
"""
Diagnose DL3DV train/test split to see if cameras are being loaded correctly.

This checks:
1. What image names are in the converted COLMAP data
2. What image names are in the JSON split (train vs test)
3. If there's a mismatch or ordering issue
"""

import os
import sys
import json
import argparse

def read_images_txt(images_txt_path):
    """Read COLMAP images.txt and return dict of image_name -> image_id"""
    images = {}
    with open(images_txt_path, 'r') as f:
        reading_image = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if not reading_image:
                # Image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                parts = line.split()
                image_id = int(parts[0])
                image_name = parts[9]
                # Strip extension for comparison
                name_no_ext = os.path.splitext(image_name)[0]
                images[name_no_ext] = image_id
                reading_image = True
            else:
                # Points line (skip)
                reading_image = False
    
    return images

def main():
    parser = argparse.ArgumentParser(description="Diagnose DL3DV train/test split")
    parser.add_argument('--scene', type=str, required=True,
                       help='Path to converted scene directory')
    parser.add_argument('--json_split', type=str, required=True,
                       help='Path to JSON split file')
    parser.add_argument('--scene_name', type=str, required=True,
                       help='Scene name in JSON (e.g., hash)')
    parser.add_argument('--case_id', type=int, default=0,
                       help='Case ID to check (default: 0)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DL3DV Train/Test Split Diagnosis")
    print("="*80)
    print(f"Scene: {args.scene}")
    print(f"JSON Split: {args.json_split}")
    print(f"Scene Name: {args.scene_name}")
    print(f"Case ID: {args.case_id}")
    print()
    
    # 1. Read converted COLMAP data
    images_txt = os.path.join(args.scene, 'sparse', '0', 'images.txt')
    if not os.path.exists(images_txt):
        print(f"❌ images.txt not found: {images_txt}")
        return 1
    
    colmap_images = read_images_txt(images_txt)
    print(f"1. Converted COLMAP data:")
    print(f"   Found {len(colmap_images)} images")
    print(f"   Sample (first 5):")
    for name in sorted(colmap_images.keys())[:5]:
        print(f"     {name} -> ID {colmap_images[name]}")
    print()
    
    # 2. Read JSON split
    with open(args.json_split, 'r') as f:
        split_data = json.load(f)
    
    if args.scene_name not in split_data['scenes']:
        print(f"❌ Scene '{args.scene_name}' not found in JSON")
        print(f"   Available scenes: {list(split_data['scenes'].keys())[:5]}...")
        return 1
    
    scene_data = split_data['scenes'][args.scene_name]
    if args.case_id >= len(scene_data):
        print(f"❌ Case ID {args.case_id} out of range (max: {len(scene_data)-1})")
        return 1
    
    case = scene_data[args.case_id]
    train_names = case.get('ref_views', [])
    test_names = case.get('target_views', [])
    
    print(f"2. JSON Split (case {args.case_id}):")
    print(f"   Train views: {len(train_names)}")
    print(f"   Test views: {len(test_names)}")
    print(f"   Sample train (first 5): {train_names[:5]}")
    print(f"   Sample test (first 5): {test_names[:5]}")
    print()
    
    # 3. Check for mismatches
    print(f"3. Checking for mismatches:")
    print(f"   {'='*76}")
    
    # Check train views
    missing_train = [name for name in train_names if name not in colmap_images]
    if missing_train:
        print(f"   ❌ {len(missing_train)} TRAIN views NOT in COLMAP data:")
        for name in missing_train[:10]:
            print(f"      - {name}")
        if len(missing_train) > 10:
            print(f"      ... and {len(missing_train)-10} more")
    else:
        print(f"   ✅ All {len(train_names)} train views found in COLMAP data")
    
    # Check test views
    missing_test = [name for name in test_names if name not in colmap_images]
    if missing_test:
        print(f"   ❌ {len(missing_test)} TEST views NOT in COLMAP data:")
        for name in missing_test[:10]:
            print(f"      - {name}")
        if len(missing_test) > 10:
            print(f"      ... and {len(missing_test)-10} more")
    else:
        print(f"   ✅ All {len(test_names)} test views found in COLMAP data")
    
    print()
    
    # 4. Check for duplicate IDs
    train_ids = [colmap_images[name] for name in train_names if name in colmap_images]
    test_ids = [colmap_images[name] for name in test_names if name in colmap_images]
    
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        print(f"   ⚠️  WARNING: {len(overlap)} cameras appear in BOTH train and test!")
        print(f"      IDs: {sorted(overlap)[:10]}")
    else:
        print(f"   ✅ No overlap between train and test cameras")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if missing_train or missing_test:
        print("❌ MISMATCH DETECTED!")
        print("   Some views in JSON split are not in converted COLMAP data.")
        print("   This will cause cameras to be skipped or mismatched.")
    else:
        print("✅ All views in JSON split exist in COLMAP data")
        print("   The train/test split should work correctly.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

