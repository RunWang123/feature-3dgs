#!/usr/bin/env python3
"""
Diagnose DL3DV conversion issues by comparing all scenes.
Helps identify why some scenes produce blurry images.
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def analyze_image_quality(image_path):
    """Compute image quality metrics."""
    img = Image.open(image_path)
    arr = np.array(img)
    
    # Convert to grayscale for variance calculation
    if len(arr.shape) == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    
    # Compute Laplacian variance (blur detection)
    from scipy import ndimage
    laplacian = ndimage.laplace(gray)
    laplacian_var = laplacian.var()
    
    return {
        'mean': arr.mean(),
        'std': arr.std(),
        'min': arr.min(),
        'max': arr.max(),
        'laplacian_var': laplacian_var,  # Higher = sharper
    }

def check_source_scene(scene_path):
    """Check source scene properties."""
    info = {
        'exists': os.path.exists(scene_path),
        'has_gaussian_splat': False,
        'has_images_4': False,
        'has_transforms': False,
        'has_sparse': False,
        'transforms_data': None,
        'colmap_size': None,
        'images_4_size': None,
        'first_image_quality': None,
    }
    
    if not info['exists']:
        return info
    
    gaussian_splat_path = os.path.join(scene_path, 'gaussian_splat')
    info['has_gaussian_splat'] = os.path.exists(gaussian_splat_path)
    
    if info['has_gaussian_splat']:
        images_4_path = os.path.join(gaussian_splat_path, 'images_4')
        transforms_path = os.path.join(gaussian_splat_path, 'transforms.json')
        sparse_path = os.path.join(gaussian_splat_path, 'sparse', '0')
        
        info['has_images_4'] = os.path.exists(images_4_path)
        info['has_transforms'] = os.path.exists(transforms_path)
        info['has_sparse'] = os.path.exists(sparse_path)
        
        # Read transforms.json
        if info['has_transforms']:
            with open(transforms_path) as f:
                transforms = json.load(f)
                info['transforms_data'] = {
                    'w': transforms.get('w'),
                    'h': transforms.get('h'),
                    'k1': transforms.get('k1', 0),
                    'k2': transforms.get('k2', 0),
                    'p1': transforms.get('p1', 0),
                    'p2': transforms.get('p2', 0),
                }
        
        # Check COLMAP camera size
        if info['has_sparse']:
            cameras_bin = os.path.join(sparse_path, 'cameras.bin')
            cameras_txt = os.path.join(sparse_path, 'cameras.txt')
            
            if os.path.exists(cameras_txt):
                # Parse cameras.txt
                with open(cameras_txt) as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            width = int(parts[2])
                            height = int(parts[3])
                            info['colmap_size'] = (width, height)
                            break
        
        # Check images_4/ size and quality
        if info['has_images_4']:
            images = sorted([f for f in os.listdir(images_4_path) if f.endswith('.png')])
            if images:
                first_image_path = os.path.join(images_4_path, images[0])
                img = Image.open(first_image_path)
                info['images_4_size'] = img.size
                
                # Analyze image quality
                try:
                    info['first_image_quality'] = analyze_image_quality(first_image_path)
                except:
                    pass
    
    return info

def check_converted_scene(output_path):
    """Check converted scene output."""
    info = {
        'exists': os.path.exists(output_path),
        'has_images': False,
        'has_sparse': False,
        'num_images': 0,
        'image_size': None,
        'first_image_quality': None,
        'last_image_quality': None,
    }
    
    if not info['exists']:
        return info
    
    images_path = os.path.join(output_path, 'images')
    sparse_path = os.path.join(output_path, 'sparse', '0')
    
    info['has_images'] = os.path.exists(images_path)
    info['has_sparse'] = os.path.exists(sparse_path)
    
    if info['has_images']:
        images = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
        info['num_images'] = len(images)
        
        if images:
            first_path = os.path.join(images_path, images[0])
            last_path = os.path.join(images_path, images[-1])
            
            img = Image.open(first_path)
            info['image_size'] = img.size
            
            try:
                info['first_image_quality'] = analyze_image_quality(first_path)
                info['last_image_quality'] = analyze_image_quality(last_path)
            except:
                pass
    
    return info

def diagnose_scene(scene_hash, input_base, output_base):
    """Diagnose a single scene."""
    source_path = os.path.join(input_base, scene_hash)
    output_path = os.path.join(output_base, scene_hash)
    
    source_info = check_source_scene(source_path)
    output_info = check_converted_scene(output_path)
    
    return {
        'scene_hash': scene_hash,
        'source': source_info,
        'output': output_info,
    }

def print_diagnosis(diagnosis):
    """Print diagnosis for a scene."""
    scene = diagnosis['scene_hash']
    src = diagnosis['source']
    out = diagnosis['output']
    
    print(f"\n{'='*80}")
    print(f"Scene: {scene[:16]}...")
    print(f"{'='*80}")
    
    # Source info
    print(f"\n📁 SOURCE:")
    print(f"  Exists: {src['exists']}")
    if src['exists']:
        print(f"  Structure: gaussian_splat={src['has_gaussian_splat']}, "
              f"images_4={src['has_images_4']}, transforms={src['has_transforms']}")
        
        if src['transforms_data']:
            t = src['transforms_data']
            print(f"  transforms.json: {t['w']}x{t['h']}")
            print(f"  Distortion: k1={t['k1']:.6f}, k2={t['k2']:.6f}, "
                  f"p1={t['p1']:.6f}, p2={t['p2']:.6f}")
            
            # Check if distortion is significant
            has_dist = (abs(t['k1']) > 1e-5 or abs(t['k2']) > 1e-5 or 
                       abs(t['p1']) > 1e-5 or abs(t['p2']) > 1e-5)
            print(f"  Significant distortion: {has_dist}")
        
        if src['colmap_size']:
            print(f"  COLMAP reconstructed: {src['colmap_size'][0]}x{src['colmap_size'][1]}")
        
        if src['images_4_size']:
            print(f"  images_4/ size: {src['images_4_size'][0]}x{src['images_4_size'][1]}")
        
        if src['first_image_quality']:
            q = src['first_image_quality']
            print(f"  Source image quality:")
            print(f"    Mean={q['mean']:.1f}, Std={q['std']:.1f}, Laplacian={q['laplacian_var']:.1f}")
    
    # Output info
    print(f"\n📤 OUTPUT:")
    print(f"  Exists: {out['exists']}")
    if out['exists']:
        print(f"  Images: {out['num_images']} files")
        if out['image_size']:
            print(f"  Output size: {out['image_size'][0]}x{out['image_size'][1]}")
        
        if out['first_image_quality']:
            q = out['first_image_quality']
            print(f"  Output image quality (first):")
            print(f"    Mean={q['mean']:.1f}, Std={q['std']:.1f}, Laplacian={q['laplacian_var']:.1f}")
            
            # Detect if blurry
            if q['laplacian_var'] < 10:
                print(f"    ⚠️  VERY BLURRY (Laplacian < 10)")
            elif q['laplacian_var'] < 50:
                print(f"    ⚠️  Possibly blurry (Laplacian < 50)")
            else:
                print(f"    ✓ Sharp enough")
        
        if out['last_image_quality']:
            q = out['last_image_quality']
            print(f"  Output image quality (last):")
            print(f"    Mean={q['mean']:.1f}, Std={q['std']:.1f}, Laplacian={q['laplacian_var']:.1f}")
    
    # Comparison
    print(f"\n🔍 ANALYSIS:")
    if src['exists'] and out['exists']:
        # Compare source vs output quality
        if src['first_image_quality'] and out['first_image_quality']:
            src_lap = src['first_image_quality']['laplacian_var']
            out_lap = out['first_image_quality']['laplacian_var']
            quality_ratio = out_lap / src_lap if src_lap > 0 else 0
            
            print(f"  Quality ratio (output/source): {quality_ratio:.2f}x")
            if quality_ratio < 0.5:
                print(f"  ❌ OUTPUT IS MUCH BLURRIER THAN SOURCE!")
                print(f"     → Likely over-undistorted or double-undistorted")
            elif quality_ratio < 0.8:
                print(f"  ⚠️  Output is somewhat blurrier than source")
            else:
                print(f"  ✓ Output quality preserved")
        
        # Check size consistency
        if src['transforms_data'] and src['colmap_size']:
            t = src['transforms_data']
            colmap_w, colmap_h = src['colmap_size']
            
            colmap_matches_transforms = (abs(colmap_w - t['w']) < 50 and 
                                          abs(colmap_h - t['h']) < 50)
            
            print(f"  COLMAP size matches transforms.json: {colmap_matches_transforms}")
            if colmap_matches_transforms:
                print(f"     → COLMAP likely used undistorted images")
                print(f"     → images_4/ should NOT be undistorted again!")
            else:
                print(f"     → COLMAP used distorted images")
                print(f"     → images_4/ needs undistortion")

def main():
    parser = argparse.ArgumentParser(description='Diagnose DL3DV conversion issues')
    parser.add_argument('--input_dir', required=True, help='Input DL3DV directory')
    parser.add_argument('--output_dir', required=True, help='Output converted directory')
    parser.add_argument('--scenes', nargs='*', help='Specific scene hashes to check (all if not specified)')
    args = parser.parse_args()
    
    # Find all scenes
    if args.scenes:
        scene_hashes = args.scenes
    else:
        # Auto-detect from output directory
        if os.path.exists(args.output_dir):
            scene_hashes = sorted([d for d in os.listdir(args.output_dir) 
                                  if os.path.isdir(os.path.join(args.output_dir, d))])
        else:
            print(f"Output directory not found: {args.output_dir}")
            return
    
    if not scene_hashes:
        print("No scenes found to diagnose")
        return
    
    print(f"{'='*80}")
    print(f"DL3DV Conversion Diagnosis")
    print(f"{'='*80}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Scenes: {len(scene_hashes)}")
    
    # Diagnose each scene
    all_diagnoses = []
    for scene_hash in scene_hashes:
        diagnosis = diagnose_scene(scene_hash, args.input_dir, args.output_dir)
        all_diagnoses.append(diagnosis)
        print_diagnosis(diagnosis)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    blurry_scenes = []
    good_scenes = []
    
    for diag in all_diagnoses:
        out = diag['output']
        if out['exists'] and out['first_image_quality']:
            lap = out['first_image_quality']['laplacian_var']
            scene = diag['scene_hash'][:16]
            
            if lap < 10:
                blurry_scenes.append((scene, lap))
                print(f"❌ BLURRY: {scene}... (Laplacian={lap:.1f})")
            else:
                good_scenes.append((scene, lap))
    
    print(f"\nTotal scenes: {len(all_diagnoses)}")
    print(f"Blurry scenes: {len(blurry_scenes)}")
    print(f"Good scenes: {len(good_scenes)}")
    
    if blurry_scenes:
        print(f"\n⚠️  PROBLEMATIC SCENES:")
        for scene, lap in blurry_scenes:
            print(f"  - {scene}... (Laplacian={lap:.1f})")

if __name__ == '__main__':
    main()

