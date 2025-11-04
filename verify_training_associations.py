#!/usr/bin/env python3
"""
Verify that camera poses, images, and LSeg features are correctly associated
during training with JSON split.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add feature-3dgs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "feature-3dgs"))

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


def verify_training_associations(scene_path, json_split_path, scene_name, case_id=0):
    """
    Verify the complete association chain:
    COLMAP pose → image filename → LSeg feature
    """
    print("=" * 80)
    print("TRAINING DATA ASSOCIATION VERIFICATION")
    print("=" * 80)
    print(f"Scene: {scene_name}")
    print(f"Case ID: {case_id}")
    print()
    
    # 1. Load COLMAP
    print("Step 1: Loading COLMAP data...")
    colmap_folder = os.path.join(scene_path, "sparse/0")
    cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_folder, "images.bin"))
    cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_folder, "cameras.bin"))
    print(f"  Loaded {len(cam_extrinsics)} cameras from COLMAP")
    print()
    
    # 2. Simulate what readColmapCameras does
    print("Step 2: Simulating camera loading (as code does)...")
    images_folder = os.path.join(scene_path, "images")
    features_folder = os.path.join(scene_path, "rgb_feature_langseg")
    
    cam_info_dict = {}  # image_name -> (pose, image_path, feature_path)
    
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        
        # Extract pose (as code does)
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        # Extract image name (as code does - LINE 156-157)
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]  # ⚠️ POTENTIAL BUG HERE!
        
        # Construct feature path (as code does - LINE 161)
        semantic_feature_path = os.path.join(features_folder, image_name) + '_fmap_CxHxW.pt'
        
        cam_info_dict[image_name] = {
            'colmap_filename': extr.name,
            'image_name': image_name,
            'pose': (R, T),
            'image_path': image_path,
            'feature_path': semantic_feature_path,
            'image_exists': os.path.exists(image_path),
            'feature_exists': os.path.exists(semantic_feature_path)
        }
    
    print(f"  Processed {len(cam_info_dict)} camera infos")
    print()
    
    # 3. Load JSON split
    print("Step 3: Loading JSON split...")
    with open(json_split_path, 'r') as f:
        split_data = json.load(f)
    
    scene_data = split_data['scenes'][scene_name]
    case = scene_data[case_id]
    train_names = case.get('ref_views', [])
    test_names = case.get('target_views', [])
    
    print(f"  Training images from JSON: {train_names}")
    print(f"  Test images from JSON: {test_names}")
    print()
    
    # 4. Simulate JSON split filtering (as code does - LINE 243-250)
    print("Step 4: Simulating JSON split filtering...")
    
    train_names_set = set(train_names)
    test_names_set = set(test_names)
    
    matched_train = []
    matched_test = []
    unmatched_json = []
    
    # Check what code does: LINE 245
    print("\n  Checking image_name matching logic:")
    for img_name in cam_info_dict.keys():
        # This is what LINE 245 does:
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        if img_name_no_ext != img_name:
            print(f"    ⚠️  WARNING: image_name has extension! '{img_name}' vs '{img_name_no_ext}'")
        
        if img_name_no_ext in train_names_set:
            matched_train.append(img_name)
        elif img_name_no_ext in test_names_set:
            matched_test.append(img_name)
    
    # Check for unmatched JSON entries
    all_matched = set(matched_train + matched_test)
    for json_name in train_names + test_names:
        if json_name not in all_matched:
            unmatched_json.append(json_name)
    
    print(f"\n  Matched {len(matched_train)} training images")
    print(f"  Matched {len(matched_test)} test images")
    
    if unmatched_json:
        print(f"  ⚠️  WARNING: {len(unmatched_json)} JSON entries not matched!")
        print(f"     Unmatched: {unmatched_json}")
    print()
    
    # 5. Verify associations for training images
    print("Step 5: Verifying TRAINING image associations in detail...")
    print()
    
    issues = []
    
    for img_name in matched_train:
        info = cam_info_dict[img_name]
        
        print(f"  Training image: {img_name}")
        print(f"    COLMAP filename: {info['colmap_filename']}")
        print(f"    Image path: {info['image_path']}")
        print(f"    Feature path: {info['feature_path']}")
        print(f"    Pose T: [{info['pose'][1][0]:.3f}, {info['pose'][1][1]:.3f}, {info['pose'][1][2]:.3f}]")
        
        # Check file existence
        if not info['image_exists']:
            print(f"    ❌ ERROR: Image file NOT FOUND!")
            issues.append(f"Image {img_name}: file not found")
        else:
            # Verify image can be loaded
            try:
                img = Image.open(info['image_path'])
                print(f"    ✅ Image: {img.size[0]}x{img.size[1]} OK")
            except Exception as e:
                print(f"    ❌ ERROR: Cannot load image: {e}")
                issues.append(f"Image {img_name}: cannot load")
        
        if not info['feature_exists']:
            print(f"    ❌ ERROR: Feature file NOT FOUND!")
            issues.append(f"Feature {img_name}: file not found")
        else:
            # Verify feature can be loaded
            try:
                feature = torch.load(info['feature_path'])
                print(f"    ✅ Feature: {feature.shape} OK")
                
                # Check if feature dimensions make sense
                if len(feature.shape) != 3:
                    print(f"    ⚠️  WARNING: Feature should be 3D (C,H,W), got {feature.shape}")
                    issues.append(f"Feature {img_name}: wrong dimensions")
            except Exception as e:
                print(f"    ❌ ERROR: Cannot load feature: {e}")
                issues.append(f"Feature {img_name}: cannot load")
        
        print()
    
    # 6. Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Associations are correct:")
        print(f"  - {len(matched_train)} training images properly mapped")
        print(f"  - Camera poses, images, and features are correctly associated")
        print()
        print("If training PSNR is still low (~10), the issue is likely:")
        print("  1. Image resolution mismatch between training and features")
        print("  2. Images were preprocessed differently for COLMAP vs LSeg")
        print("  3. Training hyperparameters need adjustment")
        print("  4. Insufficient training iterations for 2-view setup")
    else:
        print(f"❌ FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("These issues will cause low PSNR!")
    
    print("=" * 80)
    
    return issues


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify training data associations")
    parser.add_argument("--scene_path", type=str, required=True,
                        help="Path to scene directory")
    parser.add_argument("--json_split", type=str, required=True,
                        help="Path to JSON split file")
    parser.add_argument("--scene_name", type=str, required=True,
                        help="Scene name in JSON file")
    parser.add_argument("--case_id", type=int, default=0,
                        help="Case ID to check (default: 0)")
    
    args = parser.parse_args()
    
    issues = verify_training_associations(
        args.scene_path,
        args.json_split,
        args.scene_name,
        args.case_id
    )
    
    sys.exit(0 if len(issues) == 0 else 1)

