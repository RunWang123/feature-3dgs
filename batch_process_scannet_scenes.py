#!/usr/bin/env python3
"""
Batch process all ScanNet scenes to feature-3dgs format with ground truth poses.
Uses the create_feature3dgs_structure_with_gt_poses.py conversion logic.
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
import subprocess


def process_single_scene(args):
    """
    Wrapper function for processing a single scene.
    
    Args:
        args: tuple of (scene_path, output_path, target_size, num_images)
    
    Returns:
        tuple of (scene_name, success)
    """
    scene_path, output_path, target_size, num_images = args
    scene_name = os.path.basename(scene_path)
    
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        conversion_script = os.path.join(script_dir, 'create_feature3dgs_structure_with_gt_poses.py')
        
        # Run the conversion script as subprocess
        cmd = [
            sys.executable,  # Use same Python interpreter
            conversion_script,
            '--scene', scene_path,
            '--output', output_path,
            '--size', str(target_size),
            '--num_images', str(num_images)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per scene
        )
        
        if result.returncode == 0:
            return (scene_name, True)
        else:
            print(f"\n❌ Error processing {scene_name}:")
            print(result.stderr)
            return (scene_name, False)
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ Timeout processing {scene_name} (exceeded 10 minutes)")
        return (scene_name, False)
    except Exception as e:
        print(f"\n❌ Exception processing {scene_name}: {e}")
        return (scene_name, False)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all ScanNet scenes to feature-3dgs format"
    )
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Input directory containing ScanNet scenes')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for processed scenes')
    parser.add_argument('--scenes', nargs='+', type=str, default=None,
                       help='Specific scenes to process (e.g., scene0686_01 scene0687_00). If not specified, processes all scenes.')
    parser.add_argument('--size', type=int, default=448,
                       help='Target square image size (default: 448)')
    parser.add_argument('--num_images', type=int, default=30,
                       help='Number of images per scene (default: 30)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1, use 4-8 for fast processing)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip scenes that already have output directories')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.is_dir():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of scenes
    if args.scenes:
        # Use specified scenes
        scenes = [input_dir / scene for scene in args.scenes]
        # Verify scenes exist
        scenes = [s for s in scenes if s.is_dir()]
        if len(scenes) == 0:
            print(f"❌ Error: None of the specified scenes found in {input_dir}")
            sys.exit(1)
    else:
        # Get all scenes starting with 'scene'
        scenes = sorted([d for d in input_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('scene')])
    
    if len(scenes) == 0:
        print(f"❌ Error: No scenes found in {input_dir}")
        sys.exit(1)
    
    # Filter out existing scenes if requested
    if args.skip_existing:
        original_count = len(scenes)
        scenes = [s for s in scenes if not (output_dir / s.name).exists()]
        skipped = original_count - len(scenes)
        if skipped > 0:
            print(f"⏭️  Skipping {skipped} existing scenes")
    
    if len(scenes) == 0:
        print("✅ All scenes already processed!")
        sys.exit(0)
    
    print(f"\n{'='*70}")
    print(f"Batch Processing ScanNet Scenes")
    print(f"{'='*70}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of scenes: {len(scenes)}")
    print(f"Images per scene: {args.num_images}")
    print(f"Target size:      {args.size}×{args.size}")
    print(f"Workers:          {args.workers}")
    print(f"{'='*70}\n")
    
    # List some scenes
    print("Sample scenes to process:")
    for scene in scenes[:5]:
        print(f"  - {scene.name}")
    if len(scenes) > 5:
        print(f"  ... and {len(scenes) - 5} more")
    print()
    
    # Prepare arguments for each scene
    scene_args = [
        (str(scene), str(output_dir / scene.name), args.size, args.num_images)
        for scene in scenes
    ]
    
    # Process scenes
    if args.workers > 1:
        # Parallel processing
        print(f"🚀 Processing scenes in parallel with {args.workers} workers...\n")
        with mp.Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_scene, scene_args),
                total=len(scene_args),
                desc="Overall progress"
            ))
    else:
        # Sequential processing
        print("🔄 Processing scenes sequentially...\n")
        results = []
        for scene_arg in tqdm(scene_args, desc="Processing scenes"):
            result = process_single_scene(scene_arg)
            results.append(result)
    
    # Summary
    success_count = sum(1 for _, success in results if success)
    fail_count = len(results) - success_count
    
    print(f"\n{'='*70}")
    print(f"Batch Processing Summary")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {success_count} scenes")
    print(f"❌ Failed:                 {fail_count} scenes")
    print(f"📁 Output directory:       {output_dir}")
    print(f"{'='*70}\n")
    
    if fail_count > 0:
        print("❌ Failed scenes:")
        for scene_name, success in results:
            if not success:
                print(f"  - {scene_name}")
        print()
    
    print("✅ Next steps:")
    print(f"1. Your scenes are ready for feature-3dgs training in: {output_dir}")
    print(f"2. Each scene has {args.num_images} images (448×448 PNG)")
    print(f"3. All labels and depths are copied")
    print(f"4. Camera poses are normalized to [-1, 1] range")
    print()
    print("Example training command:")
    print(f"  python train.py \\")
    print(f"    -s {output_dir}/scene0686_01 \\")
    print(f"    -m /path/to/output/scene0686_01 \\")
    print(f"    -f lseg --speedup --iterations 30000 \\")
    print(f"    --eval --json_split_path /path/to/split.json --case_id 0")
    print()


if __name__ == '__main__':
    main()

