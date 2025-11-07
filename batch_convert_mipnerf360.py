#!/usr/bin/env python3
"""
Batch convert MipNeRF360 scenes to feature-3dgs format.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def batch_convert_mipnerf360(input_base, output_base, target_size=448, use_downscale=None):
    """
    Batch convert all MipNeRF360 scenes.
    
    Args:
        input_base: Base directory containing scene folders (e.g., bicycle, bonsai, etc.)
        output_base: Base output directory
        target_size: Target image size
        use_downscale: Downscale factor (2, 4, or 8), or None for full resolution
    """
    
    # Get all scene folders
    scene_folders = sorted([d for d in os.listdir(input_base) 
                          if os.path.isdir(os.path.join(input_base, d))])
    
    print(f"{'='*80}")
    print(f"Batch Converting MipNeRF360 Scenes")
    print(f"{'='*80}")
    print(f"Input base: {input_base}")
    print(f"Output base: {output_base}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Use downscale: {use_downscale if use_downscale else 'Full resolution'}")
    print(f"Found {len(scene_folders)} scenes: {scene_folders}")
    print(f"{'='*80}\n")
    
    # Create output base directory
    os.makedirs(output_base, exist_ok=True)
    
    # Convert each scene
    success_count = 0
    failed_scenes = []
    
    for scene_name in scene_folders:
        scene_path = os.path.join(input_base, scene_name)
        output_path = os.path.join(output_base, scene_name)
        
        print(f"\n{'='*80}")
        print(f"Processing scene {success_count + 1}/{len(scene_folders)}: {scene_name}")
        print(f"{'='*80}")
        
        # Check if sparse reconstruction exists
        sparse_dir = os.path.join(scene_path, "sparse", "0")
        if not os.path.exists(sparse_dir):
            print(f"⚠️  Warning: No sparse reconstruction found at {sparse_dir}, skipping")
            failed_scenes.append((scene_name, "No sparse reconstruction"))
            continue
        
        # Build command
        script_path = os.path.join(os.path.dirname(__file__), "convert_mipnerf360_to_feature3dgs.py")
        cmd = [
            sys.executable,
            script_path,
            "--scene_path", scene_path,
            "--output_path", output_path,
            "--target_size", str(target_size)
        ]
        
        if use_downscale:
            cmd.extend(["--use_downscale", str(use_downscale)])
        
        # Run conversion
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            success_count += 1
            print(f"\n✅ Successfully converted {scene_name}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to convert {scene_name}")
            print(f"Error: {e}")
            failed_scenes.append((scene_name, str(e)))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Batch Conversion Summary")
    print(f"{'='*80}")
    print(f"Total scenes: {len(scene_folders)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {len(failed_scenes)}")
    
    if failed_scenes:
        print(f"\n❌ Failed scenes:")
        for scene_name, reason in failed_scenes:
            print(f"  - {scene_name}: {reason}")
    
    print(f"\n✅ All conversions complete!")
    print(f"Output directory: {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Batch convert MipNeRF360 scenes")
    parser.add_argument("--input_base", type=str, required=True,
                      help="Base directory with MipNeRF360 scenes")
    parser.add_argument("--output_base", type=str, required=True,
                      help="Output base directory")
    parser.add_argument("--target_size", type=int, default=448,
                      help="Target image size (default: 448)")
    parser.add_argument("--use_downscale", type=int, choices=[2, 4, 8], default=None,
                      help="Use downscaled images (2, 4, or 8)")
    
    args = parser.parse_args()
    
    batch_convert_mipnerf360(
        input_base=args.input_base,
        output_base=args.output_base,
        target_size=args.target_size,
        use_downscale=args.use_downscale
    )


if __name__ == "__main__":
    main()

