#!/usr/bin/env python3
"""
Generate split.json for pre-processed DL3DV dataset.

Structure:
  DL3DV-10K-processed/
    32/[scene_hash]/frame_*.png
    48/[scene_hash]/frame_*.png  
    64/[scene_hash]/frame_*.png

Uses test split: index % 8 == 1
"""

import os
import json
import argparse


def generate_split_for_view_count(base_dir, view_count, output_json):
    """
    Generate split.json for a specific view count folder.
    
    Args:
        base_dir: Base directory (e.g., DL3DV-10K-processed)
        view_count: View count (32, 48, or 64)
        output_json: Output JSON file path
    """
    
    view_dir = os.path.join(base_dir, str(view_count))
    
    if not os.path.exists(view_dir):
        print(f"❌ Error: Directory not found: {view_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Generating DL3DV {view_count}-view split")
    print(f"  Source: {view_dir}")
    print(f"{'='*80}\n")
    
    # Find all scene directories
    scenes = {}
    scene_folders = []
    
    for item in sorted(os.listdir(view_dir)):
        item_path = os.path.join(view_dir, item)
        
        if not os.path.isdir(item_path):
            continue
        
        scene_folders.append(item)
    
    print(f"Found {len(scene_folders)} scenes")
    
    # Generate split for each scene
    for scene_name in scene_folders:
        scene_path = os.path.join(view_dir, scene_name)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(scene_path) 
                            if f.endswith(('.png', '.jpg', '.PNG', '.JPG'))])
        
        if len(image_files) == 0:
            print(f"  ⚠️  Warning: No images found for {scene_name}, skipping")
            continue
        
        # Extract frame IDs (remove extension)
        frame_ids = [os.path.splitext(f)[0] for f in image_files]
        
        # Split into train/test based on index % 8 == 1
        # Same logic as MipNeRF360
        ref_views = []      # Training views (index % 8 != 1)
        target_views = []   # Test views (index % 8 == 1)
        
        for idx, frame_id in enumerate(frame_ids):
            if idx % 8 == 1:
                target_views.append(frame_id)
            else:
                ref_views.append(frame_id)
        
        if len(ref_views) == 0 or len(target_views) == 0:
            print(f"  ⚠️  Warning: Invalid split for {scene_name}, skipping")
            continue
        
        # Create single case
        # "views" key is ignored, only ref_views and target_views matter
        scenes[scene_name] = [
            {
                "views": [],                    # Ignored by feature-3dgs
                "ref_views": ref_views,         # ALL training views
                "target_views": target_views    # ALL test views
            }
        ]
        
        print(f"  ✓ {scene_name}: {len(ref_views)} train, {len(target_views)} test")
    
    if len(scenes) == 0:
        print("❌ Error: No valid scenes found!")
        return False
    
    # Create split JSON
    split_data = {
        "config": {
            "case_size": 1,
            "ref_indices": [0, -1],
            "stride": 8,
            "llff_hold": 8,
            "data_root": view_dir,
            "view_count": view_count
        },
        "scenes": scenes
    }
    
    # Write to file
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n✅ Split JSON created: {output_json}")
    print(f"   Total scenes: {len(scenes)}")
    
    # Calculate totals
    total_train = sum(len(cases[0]['ref_views']) for cases in scenes.values())
    total_test = sum(len(cases[0]['target_views']) for cases in scenes.values())
    
    print(f"   Total train views: {total_train}")
    print(f"   Total test views: {total_test}")
    print(f"   Avg per scene: {total_train/len(scenes):.1f} train, {total_test/len(scenes):.1f} test")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate split JSONs for pre-processed DL3DV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate splits for 32, 48, and 64 views
  python create_dl3dv_processed_split_json.py \\
      --base_dir /home/runw/Project/data/colmap/data/DL3DV-10K-processed \\
      --output_dir /home/runw/Project/data/colmap/data/dl3dv_splits \\
      --view_counts 32 48 64
        """
    )
    
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory (DL3DV-10K-processed)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for split JSON files')
    parser.add_argument('--view_counts', nargs='+', type=int, required=True,
                       help='View counts to generate (e.g., 32 48 64)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate split for each view count
    all_success = True
    for view_count in args.view_counts:
        output_json = os.path.join(args.output_dir, f'dl3dv_{view_count}view_split.json')
        success = generate_split_for_view_count(args.base_dir, view_count, output_json)
        all_success = all_success and success
    
    if all_success:
        print(f"\n{'='*80}")
        print(f"✅ All split JSONs generated successfully!")
        print(f"   Output directory: {args.output_dir}")
        print(f"{'='*80}")
    else:
        exit(1)


if __name__ == '__main__':
    main()

