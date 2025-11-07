#!/usr/bin/env python3
"""
Batch convert DL3DV scenes to feature-3dgs format.

This script processes multiple DL3DV scenes (hash-named folders) and converts them
to the feature-3dgs format with image preprocessing and camera parameter adjustment.
"""

import os
import sys
import argparse
from pathlib import Path
from convert_dl3dv_to_feature3dgs import convert_dl3dv_scene

def find_dl3dv_scenes(base_dir):
    """
    Find all DL3DV scene directories (hash folders containing gaussian_splat/).
    
    Args:
        base_dir: Base directory containing DL3DV scenes
        
    Returns:
        List of (scene_path, scene_name) tuples
    """
    scenes = []
    
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory not found: {base_dir}")
        return scenes
    
    # Iterate through all subdirectories
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        # Skip known non-scene directories
        if item in ['cache', 'mipnerf360', 'benchmark-meta.csv']:
            continue
        
        # Check if it contains gaussian_splat/
        gaussian_splat_dir = os.path.join(item_path, 'gaussian_splat')
        if os.path.exists(gaussian_splat_dir):
            scenes.append((item_path, item))
    
    return scenes

def batch_convert_dl3dv(input_dir, output_dir, target_size=448, skip_existing=False, max_scenes=None):
    """
    Batch convert DL3DV scenes.
    
    Args:
        input_dir: Input directory containing DL3DV scenes
        output_dir: Output directory for converted scenes
        target_size: Target square image size
        skip_existing: Skip scenes that already exist in output directory
        max_scenes: Maximum number of scenes to process (None for all)
    """
    print("="*80)
    print("DL3DV Dataset Batch Conversion")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Skip existing: {skip_existing}")
    
    # Find all scenes
    print("\nScanning for DL3DV scenes...")
    scenes = find_dl3dv_scenes(input_dir)
    
    if len(scenes) == 0:
        print("❌ No DL3DV scenes found!")
        return
    
    print(f"Found {len(scenes)} DL3DV scenes")
    
    # Limit number of scenes if specified
    if max_scenes is not None and max_scenes < len(scenes):
        scenes = scenes[:max_scenes]
        print(f"Processing first {max_scenes} scenes")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each scene
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, (scene_path, scene_name) in enumerate(scenes, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(scenes)}] Scene: {scene_name}")
        print(f"{'='*80}")
        
        output_scene_path = os.path.join(output_dir, scene_name)
        
        # Check if already exists
        if skip_existing and os.path.exists(output_scene_path):
            sparse_dir = os.path.join(output_scene_path, 'sparse', '0')
            if os.path.exists(os.path.join(sparse_dir, 'cameras.txt')):
                print(f"⏭️  Skipping (already exists): {scene_name}")
                skipped += 1
                continue
        
        # Convert scene
        try:
            success = convert_dl3dv_scene(scene_path, output_scene_path, target_size)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing scene: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("BATCH CONVERSION SUMMARY")
    print("="*80)
    print(f"Total scenes found: {len(scenes)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    if skipped > 0:
        print(f"⏭️  Skipped: {skipped}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert DL3DV scenes to feature-3dgs format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all scenes
  python batch_convert_dl3dv.py \\
      --input_dir /path/to/DL3DV-10K-Benchmark \\
      --output_dir /path/to/dl3dv_feature3dgs
  
  # Convert first 10 scenes only
  python batch_convert_dl3dv.py \\
      --input_dir /path/to/DL3DV-10K-Benchmark \\
      --output_dir /path/to/dl3dv_feature3dgs \\
      --max_scenes 10
  
  # Skip already converted scenes
  python batch_convert_dl3dv.py \\
      --input_dir /path/to/DL3DV-10K-Benchmark \\
      --output_dir /path/to/dl3dv_feature3dgs \\
      --skip_existing
        """
    )
    
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Input directory containing DL3DV scenes')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for converted scenes')
    parser.add_argument('--size', type=int, default=448,
                       help='Target square size (default: 448)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip scenes that already exist in output directory')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process (default: all)')
    
    args = parser.parse_args()
    
    batch_convert_dl3dv(
        args.input_dir,
        args.output_dir,
        args.size,
        args.skip_existing,
        args.max_scenes
    )

if __name__ == '__main__':
    main()

