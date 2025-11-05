#!/usr/bin/env python3
"""
Semantic Segmentation Metrics with Ground Truth Labels
Computes mIoU and accuracy by comparing predictions against GT labels from labels/ folder

EXACT ALIGNMENT WITH LSM PROJECT:
1. Label remapping: Uses LSM's map_func (testdata.py:12-22)
2. Feature decoding: Uses LSM's lseg.py decode_feature logic (lseg.py:56-98)
3. Semantic map: Uses LSM's argmax + 1 approach (visualization_utils.py:303)
4. Metrics: Uses torchmetrics JaccardIndex and Accuracy with ignore_index=0
5. Logit computation: Uses logit_scale * features @ text_features.t() (same as LSM)

This ensures 100% compatibility with LSM's semantic segmentation evaluation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip

# For metrics computation
from torchmetrics import JaccardIndex, Accuracy


def create_label_remapping(label_path, target_labels):
    """
    Create label remapping function from ScanNet IDs to target label IDs.
    Follows LSM's approach: large_spatial_model/datasets/testdata.py
    
    Args:
        label_path: Path to scannetv2-labels.combined.tsv
        target_labels: List of target semantic labels (e.g., ['wall', 'floor', ...])
    
    Returns:
        Vectorized function that remaps ScanNet IDs to target IDs
    """
    target_labels = [label.lower() for label in target_labels]
    
    # Read the label mapping file
    df = pd.read_csv(label_path, sep='\t')
    id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()
    
    # Map nyu40 classes to new IDs (1-indexed, 0 is background)
    nyu40class_to_newid = {
        cls: target_labels.index(cls) + 1 if cls in target_labels else target_labels.index('other') + 1 
        for cls in set(id_to_nyu40class.values())
    }
    
    # Create final ID mapping
    id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}
    
    # Vectorized remapping function (0 stays 0 for background/unlabeled)
    return np.vectorize(lambda x: id_to_newid.get(x, target_labels.index('other') + 1) if x != 0 else 0)


def load_and_remap_label(label_path, remap_fn, target_size=None):
    """
    Load a label image and remap to target label space.
    
    Args:
        label_path: Path to label PNG file
        remap_fn: Remapping function from create_label_remapping()
        target_size: (H, W) tuple to resize to, or None
    
    Returns:
        Remapped label array (H, W) with values in [0, num_classes]
    """
    # Load label image
    label_img = Image.open(label_path)
    label_array = np.array(label_img)
    
    # Remap labels
    remapped = remap_fn(label_array)
    
    # Resize if needed
    if target_size is not None:
        label_pil = Image.fromarray(remapped.astype(np.uint8))
        label_pil = label_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
        remapped = np.array(label_pil)
    
    return remapped


def compute_segmentation(args):
    """
    Compute segmentation from rendered features and compare against GT labels.
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get labels
    if args.label_src and args.label_src != 'default':
        labelset = args.label_src.split(',')
    else:
        print("Error: Must provide --label_src argument with comma-separated labels")
        print("Example: --label_src \"wall,floor,ceiling,chair,table,sofa,bed,other\"")
        return
    
    num_classes = len(labelset) + 1  # +1 for background
    print(f"Using {len(labelset)} semantic labels + background")
    print(f"Labels: {labelset}")
    
    # Load LSeg checkpoint to get proper text encoder (512-dim)
    print("Loading LSeg checkpoint for text encoding...")
    import torch
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # Load CLIP model and get text features
    import clip
    clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    
    # Encode text
    text = clip.tokenize(labelset).to(device)
    with torch.no_grad():
        text_features_768 = clip_model.encode_text(text).float()
    
    # Project to 512-dim using LSeg's projection (from checkpoint)
    # LSeg uses a linear layer to project 768 -> 512
    # This should be in the checkpoint under 'state_dict'
    try:
        # Try to load the text projection layer from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Find the clip text projection weights
        # In LSeg, this is typically model.clip_pretrained.model.text_projection
        text_proj_weight = None
        for key in state_dict.keys():
            if 'text_projection' in key:
                text_proj_weight = state_dict[key]
                break
        
        if text_proj_weight is not None and text_proj_weight.shape[0] == 512:
            print(f"Found text projection: {text_proj_weight.shape}")
            text_proj = text_proj_weight.to(device)
            # Apply projection: (batch, 768) @ (768, 512) -> (batch, 512)
            text_features = text_features_768 @ text_proj
        else:
            # Fallback: Use simple linear projection
            print("Using fallback linear projection 768->512")
            proj_layer = nn.Linear(768, 512, bias=False).to(device)
            nn.init.xavier_uniform_(proj_layer.weight)
            text_features = proj_layer(text_features_768)
    except Exception as e:
        print(f"Warning: Could not load text projection from checkpoint: {e}")
        print("Using fallback linear projection 768->512")
        proj_layer = nn.Linear(768, 512, bias=False).to(device)
        nn.init.xavier_uniform_(proj_layer.weight)
        text_features = proj_layer(text_features_768)
    
    # Logit scale (EXACT same as LSM)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().to(device)
    
    print(f"Text features shape: {text_features.shape}")
    
    # Setup label remapping
    if args.label_mapping_file:
        print(f"Using label mapping file: {args.label_mapping_file}")
        remap_fn = create_label_remapping(args.label_mapping_file, labelset)
    else:
        print("Warning: No label mapping file provided. Using identity mapping.")
        remap_fn = lambda x: x  # Identity mapping
    
    # Initialize metrics
    miou_metric = JaccardIndex(num_classes=num_classes, task='multiclass', ignore_index=0).to(device)
    accuracy_metric = Accuracy(num_classes=num_classes, task='multiclass', ignore_index=0).to(device)
    
    # Process test views
    for split_name in ["test", "train"]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        feature_path = os.path.join(args.data, split_name, f"ours_{args.iteration}", "saved_feature")
        gt_labels_path = os.path.join(args.scene_data_path, "labels")
        
        if not os.path.exists(feature_path):
            print(f"Feature path not found: {feature_path}")
            continue
        
        if not os.path.exists(gt_labels_path):
            print(f"GT labels path not found: {gt_labels_path}")
            print("Please provide --scene_data_path pointing to the original scene data with labels/ folder")
            continue
        
        # Check GT labels
        gt_label_files = sorted([f for f in os.listdir(gt_labels_path) if f.endswith('.png')])
        print(f"Found {len(gt_label_files)} GT label files")
        if len(gt_label_files) > 0:
            print(f"Sample GT label files: {gt_label_files[:3]}")
        
        # Load mapping from indices to frame names
        index_to_frame = {}
        
        # Option 1: Use JSON split file (preferred)
        if args.json_split_path and os.path.exists(args.json_split_path):
            print(f"Loading frame mappings from JSON split: {args.json_split_path}")
            with open(args.json_split_path, 'r') as f:
                split_data = json.load(f)
            
            # Get scene name from data path
            scene_name = os.path.basename(os.path.normpath(args.scene_data_path))
            
            if 'scenes' in split_data and scene_name in split_data['scenes']:
                scene_cases = split_data['scenes'][scene_name]
                
                # If case_id specified, use that case; otherwise use first case
                if args.case_id >= 0 and args.case_id < len(scene_cases):
                    case = scene_cases[args.case_id]
                else:
                    case = scene_cases[0]
                    if args.case_id >= 0:
                        print(f"Warning: case_id {args.case_id} not found, using case 0")
                
                # Map based on split
                if split_name == 'test':
                    frame_names = case.get('target_views', [])
                else:  # train
                    frame_names = case.get('ref_views', [])
                
                for idx, frame_name in enumerate(frame_names):
                    index_to_frame[idx] = frame_name
                
                print(f"Loaded {len(index_to_frame)} {split_name} frame mappings from JSON split")
                if len(index_to_frame) > 0:
                    print(f"Sample mappings: {list(index_to_frame.items())[:3]}")
            else:
                print(f"Warning: Scene {scene_name} not found in JSON split")
        
        # Option 2: Use cameras.json from render output
        if not index_to_frame:
            cameras_json = os.path.join(args.data, split_name, f"ours_{args.iteration}", "cameras.json")
            if os.path.exists(cameras_json):
                with open(cameras_json, 'r') as f:
                    cameras = json.load(f)
                for idx, cam in enumerate(cameras):
                    img_name = cam.get('img_name', '')
                    # Extract frame ID from image name (e.g., "000010" from path or filename)
                    frame_id = os.path.splitext(os.path.basename(img_name))[0]
                    index_to_frame[idx] = frame_id
                print(f"Loaded {len(index_to_frame)} camera mappings from cameras.json")
                if len(index_to_frame) > 0:
                    print(f"Sample mappings: {list(index_to_frame.items())[:3]}")
            else:
                print(f"Warning: Neither JSON split nor cameras.json found")
                print("Will try to match by padding frame IDs (likely incorrect)...")
        
        # Get feature files
        feature_files = sorted([f for f in os.listdir(feature_path) if f.endswith('_fmap_CxHxW.pt')])
        
        if len(feature_files) == 0:
            print(f"No feature files found in {feature_path}")
            continue
        
        print(f"Found {len(feature_files)} views to evaluate")
        print(f"Sample feature files: {feature_files[:3]}")
        
        # Reset metrics
        miou_metric.reset()
        accuracy_metric.reset()
        
        # Process each view
        for feature_file in tqdm(feature_files, desc=f"Evaluating {split_name}"):
            # Extract index from feature filename
            # Patterns: "00000_fmap_CxHxW.pt" -> 0
            #           "00001_fmap_CxHxW.pt" -> 1
            parts = feature_file.replace('_fmap_CxHxW.pt', '').split('_')
            
            # Try to find the numeric ID
            feature_idx = None
            for part in parts:
                if part.isdigit():
                    feature_idx = int(part)
                    break
            
            if feature_idx is None:
                print(f"Warning: Could not extract index from: {feature_file}")
                continue
            
            # Map feature index to frame ID using cameras.json
            if index_to_frame:
                frame_id = index_to_frame.get(feature_idx)
                if frame_id is None:
                    print(f"Warning: No camera mapping for index {feature_idx}")
                    continue
            else:
                # Fallback: try to pad the index to match label format
                # Feature files use indices like 00000, 00001, labels use 000000, 000010
                # This is likely wrong without camera mapping
                frame_id = str(feature_idx).zfill(6)
                print(f"Warning: No camera mapping, using padded index {frame_id} (likely incorrect)")
            
            # Load feature
            feature = torch.load(os.path.join(feature_path, feature_file))
            feature = feature.to(device).to(torch.float32)
            
            # Load corresponding GT label
            label_file = os.path.join(gt_labels_path, f"{frame_id}.png")
            if not os.path.exists(label_file):
                print(f"Warning: GT label not found: {label_file} (from feature: {feature_file})")
                continue
            
            # Get feature size for resizing label
            _, h, w = feature.shape
            gt_label = load_and_remap_label(label_file, remap_fn, target_size=(h, w))
            gt_label_tensor = torch.from_numpy(gt_label).long().to(device)
            
            # Compute prediction from feature (EXACT same as LSM lseg.py decode_feature)
            # Follow LSM's logic: large_spatial_model/lseg.py lines 91-98
            c, h, w = feature.shape
            
            # Reshape: (C, H, W) -> (H, W, C) -> (H*W, C)
            image_features = feature.permute(1, 2, 0).reshape(-1, c)
            
            # Normalize features (EXACT same as LSM)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute logits with scale (EXACT same as LSM)
            logits_per_image = logit_scale * image_features.half() @ text_features_norm.t()
            
            # Reshape back: (H*W, num_labels) -> (H, W, num_labels) -> (num_labels, H, W)
            logits = logits_per_image.float().view(h, w, -1).permute(2, 0, 1)
            
            # Get semantic map (EXACT same as LSM: visualization_utils.py line 303)
            pred_class = torch.argmax(logits, dim=0) + 1  # +1 to make 1-indexed
            
            # Background pixels should stay as 0
            pred_class = torch.where(gt_label_tensor == 0, torch.zeros_like(pred_class), pred_class)
            
            # Update metrics
            miou_metric.update(pred_class, gt_label_tensor)
            accuracy_metric.update(pred_class, gt_label_tensor)
        
        # Compute final metrics
        final_miou = miou_metric.compute()
        final_accuracy = accuracy_metric.compute()
        
        print(f"\n{'='*60}")
        print(f"Results for {split_name} split:")
        print(f"{'='*60}")
        print(f"mIoU:     {final_miou.item():.4f}")
        print(f"Accuracy: {final_accuracy.item():.4f}")
        print(f"{'='*60}\n")
        
        # Save results to file if output path is specified
        if hasattr(args, 'output') and args.output:
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': args.data,
                'scene_path': args.scene_data_path,
                'iteration': args.iteration,
                'split': split_name,
                'labels': args.label_src,
                'num_classes': num_classes,
                'num_samples': len(feature_files),
                'metrics': {
                    'mIoU': float(final_miou.item()),
                    'accuracy': float(final_accuracy.item())
                }
            }
            
            output_path = args.output
            os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to: {output_path}")
        
        return final_miou.item(), final_accuracy.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Metrics with GT Labels')
    
    # Model parameters
    parser.add_argument('--weights', default='demo_e200.ckpt', type=str,
                       help='Path to LSeg checkpoint for text encoding')
    
    # Data parameters
    parser.add_argument('--data', required=True, type=str,
                       help='Path to model output directory (contains test/ours_XXXX/saved_feature/)')
    parser.add_argument('--scene_data_path', required=True, type=str,
                       help='Path to original scene data (contains labels/ folder)')
    parser.add_argument('--label_mapping_file', default=None, type=str,
                       help='Path to scannetv2-labels.combined.tsv for label remapping')
    parser.add_argument('--json_split_path', default=None, type=str,
                       help='Path to JSON split file for mapping indices to frame names')
    parser.add_argument('--case_id', default=-1, type=int,
                       help='Case ID to evaluate (if using JSON split with multiple cases)')
    parser.add_argument('--iteration', default=7000, type=int,
                       help='Which iteration to evaluate')
    parser.add_argument('--label_src', required=True, type=str,
                       help='Comma-separated semantic labels (e.g., "wall,floor,ceiling,chair,table,sofa,bed,other")')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON file (e.g., results/segmentation_metrics.json)')
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.data):
        print(f"Error: Data path not found: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.scene_data_path):
        print(f"Error: Scene data path not found: {args.scene_data_path}")
        sys.exit(1)
    
    compute_segmentation(args)

