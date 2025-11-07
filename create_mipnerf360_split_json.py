#!/usr/bin/env python3
"""
Create split JSON files for MipNeRF360 dataset.

Test/train split: if index % 8 == 1, it's test, otherwise training
(following NeRF/MipNeRF360 convention)
"""

import os
import json
from pathlib import Path

def create_mipnerf360_split(base_dir, output_path, test_every=8, test_offset=1):
    """
    Create split JSON for MipNeRF360 dataset.
    
    Args:
        base_dir: Base directory containing scene folders (e.g., bonsai, counter, etc.)
        output_path: Output path for the JSON file
        test_every: Test every N-th image (default 8 for MipNeRF360)
        test_offset: Offset for test images (default 1, so index % 8 == 1 is test)
    """
    
    # Get all scene folders
    scene_folders = sorted([d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Found {len(scene_folders)} scenes: {scene_folders}")
    
    scenes_data = {}
    
    for scene_name in scene_folders:
        scene_path = os.path.join(base_dir, scene_name)
        
        # Get all image files (sorted)
        image_files = sorted([f for f in os.listdir(scene_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"  ⚠️  No images found in {scene_name}, skipping")
            continue
        
        print(f"\nProcessing {scene_name}:")
        print(f"  Total images: {len(image_files)}")
        
        # Create image names without extension (for compatibility)
        image_names = [os.path.splitext(f)[0] for f in image_files]
        
        # Split into train and test based on index % test_every == test_offset
        ref_views = []      # training views
        target_views = []   # test views
        
        for idx, img_name in enumerate(image_names):
            if idx % test_every == test_offset:
                target_views.append(img_name)
            else:
                ref_views.append(img_name)
        
        print(f"  Train views: {len(ref_views)}")
        print(f"  Test views:  {len(target_views)}")
        print(f"  Test indices: {[i for i in range(len(image_names)) if i % test_every == test_offset]}")
        
        # Create single case with all views
        case_data = {
            "views": image_names,           # All images
            "ref_views": ref_views,         # Training images
            "target_views": target_views    # Test images
        }
        
        scenes_data[scene_name] = [case_data]  # Single case per scene
    
    # Create config
    config = {
        "test_every": test_every,
        "test_offset": test_offset,
        "split_rule": f"index % {test_every} == {test_offset} -> test, otherwise train",
        "data_root": base_dir
    }
    
    # Create final JSON structure
    split_data = {
        "config": config,
        "scenes": scenes_data
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n✅ Split JSON saved to: {output_path}")
    print(f"\n📊 Summary:")
    print(f"  Total scenes: {len(scenes_data)}")
    for scene_name, cases in scenes_data.items():
        case = cases[0]
        print(f"  {scene_name}: {len(case['ref_views'])} train + {len(case['target_views'])} test = {len(case['views'])} total")


def main():
    # Base directory for MipNeRF360 data
    mipnerf_base = "/home/runw/Project/data/colmap/data/mipnerf360/mipnerf360"
    
    # Process 32view
    print("="*80)
    print("Creating split for mipnerf_32view")
    print("="*80)
    create_mipnerf360_split(
        base_dir=os.path.join(mipnerf_base, "mipnerf_32view"),
        output_path="/home/runw/Project/feature-3dgs/mipnerf360_32view_split.json",
        test_every=8,
        test_offset=1
    )
    
    print("\n\n")
    
    # Process 64view
    print("="*80)
    print("Creating split for mipnerf_64view")
    print("="*80)
    create_mipnerf360_split(
        base_dir=os.path.join(mipnerf_base, "mipnerf_64view"),
        output_path="/home/runw/Project/feature-3dgs/mipnerf360_64view_split.json",
        test_every=8,
        test_offset=1
    )


if __name__ == "__main__":
    main()

