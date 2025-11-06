#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def load_scannet_depth(depth_path, target_size=448):
    """
    Load ScanNet depth image with the SAME crop+resize as RGB images.
    
    IMPORTANT: This applies the exact same preprocessing as create_feature3dgs_structure_with_gt_poses.py:
    1. Central crop from 512×386 to 386×386 (square)
    2. Resize to 448×448
    
    ScanNet depth is stored as uint16 PNG in millimeters.
    
    Args:
        depth_path: Path to depth PNG file
        target_size: Target size after crop+resize (default 448)
        
    Returns:
        depth: Depth tensor in meters (float32), shape (448, 448)
    """
    depth_img = Image.open(depth_path)
    
    # Get original dimensions (ScanNet: 512×386)
    orig_width, orig_height = depth_img.size
    
    # Apply same crop as RGB: central crop to square
    crop_size = min(orig_width, orig_height)
    crop_left = (orig_width - crop_size) // 2
    crop_top = (orig_height - crop_size) // 2
    
    # Crop to square (386×386 for ScanNet)
    depth_img_cropped = depth_img.crop((crop_left, crop_top, 
                                        crop_left + crop_size, 
                                        crop_top + crop_size))
    
    # Resize to target size (448×448)
    # Use NEAREST to preserve depth values (same as F.interpolate mode='nearest')
    depth_img_resized = depth_img_cropped.resize((target_size, target_size), 
                                                   Image.Resampling.NEAREST)
    
    # Convert to numpy array
    depth_array = np.array(depth_img_resized, dtype=np.float32)
    
    # Convert from millimeters to meters
    depth_array = depth_array / 1000.0
    
    # Set invalid depth to 0
    depth_array[~np.isfinite(depth_array)] = 0
    
    return torch.from_numpy(depth_array).float()

def readDepthMaps(pred_depth_dir, gt_depth_dir=None, json_split_path=None, case_id=None, use_train_views=False):
    """
    Read predicted depth maps and optionally ground truth depth maps.
    Supports .npy, .pt, and PNG (ScanNet) formats.
    
    Args:
        pred_depth_dir: Directory containing predicted depth maps
        gt_depth_dir: Optional directory containing ground truth depth maps
        json_split_path: Optional JSON split file to map indices to frame IDs
        case_id: Optional case ID to get correct views
        use_train_views: If True, use ref_views (training); if False, use target_views (test)
        
    Returns:
        pred_depths: List of predicted depth tensors
        gt_depths: List of GT depth tensors (None if gt_depth_dir is None)
        depth_names: List of file names
    """
    pred_depths = []
    gt_depths = [] if gt_depth_dir else None
    depth_names = []
    
    # Get list of depth files (predicted)
    # NOTE: Each depth is saved in both .npy and .pt formats during rendering
    # We only load .npy files to avoid duplicates
    depth_files = sorted([f for f in os.listdir(pred_depth_dir) 
                         if f.endswith('.npy')])
    
    # If we have a JSON split, load it to map indices to frame IDs
    frame_id_mapping = None
    if json_split_path and case_id is not None and gt_depth_dir:
        try:
            with open(json_split_path, 'r') as f:
                split_data = json.load(f)
            
            # Extract scene name from gt_depth_dir
            scene_name = None
            for part in str(gt_depth_dir).split('/'):
                if part.startswith('scene'):
                    scene_name = part
                    break
            
            # Follow exact same logic as dataset_readers.py
            if scene_name and 'scenes' in split_data and scene_name in split_data['scenes']:
                cases = split_data['scenes'][scene_name]
                if case_id < len(cases):
                    # Use 'ref_views' for training depth or 'target_views' for test
                    if use_train_views:
                        views = cases[case_id].get('ref_views', [])
                        view_type = "ref_views (training)"
                    else:
                        views = cases[case_id].get('target_views', [])
                        view_type = "target_views (test)"
                    
                    # Create mapping: sequential index -> frame name (without extension)
                    frame_id_mapping = {i: views[i] for i in range(len(views))}
                    print(f"  Using frame ID mapping from JSON split (case {case_id})")
                    print(f"  {view_type}: {views}")
        except Exception as e:
            print(f"  Warning: Could not load JSON split: {e}")
    
    for fname in depth_files:
        # Load predicted depth
        pred_path = pred_depth_dir / fname
        if fname.endswith('.npy'):
            pred_depth = torch.from_numpy(np.load(pred_path)).float().cuda()
        else:  # .pt file
            pred_depth = torch.load(pred_path).float().cuda()
        
        # Ensure 2D (H, W)
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.squeeze(0)
        
        pred_depths.append(pred_depth)
        depth_names.append(fname)
        
        # Load GT depth if directory provided
        if gt_depth_dir:
            # Determine GT filename
            if frame_id_mapping is not None:
                # Extract index from filename (e.g., "00000.npy" -> 0)
                base_name = fname.split('.')[0]
                try:
                    idx = int(base_name)
                    if idx in frame_id_mapping:
                        frame_name = frame_id_mapping[idx]  # e.g., "000020"
                        gt_fname = f"{frame_name}.png"  # frame_name already has leading zeros
                    else:
                        print(f"  Warning: Index {idx} not in frame mapping")
                        gt_depths.append(None)
                        continue
                except ValueError:
                    gt_fname = fname.replace('.npy', '.png').replace('.pt', '.png')
            else:
                # Try direct mapping
                gt_fname = fname.replace('.npy', '.png').replace('.pt', '.png')
            
            gt_path = gt_depth_dir / gt_fname
            
            if os.path.exists(gt_path):
                # Check if it's a PNG (ScanNet format) or numpy/torch
                if gt_fname.endswith('.png'):
                    # Load with same crop+resize as RGB (already at 448×448)
                    gt_depth = load_scannet_depth(gt_path).cuda()
                elif gt_fname.endswith('.npy'):
                    gt_depth = torch.from_numpy(np.load(gt_path)).float().cuda()
                else:
                    gt_depth = torch.load(gt_path).float().cuda()
                
                # Ensure 2D (H, W)
                if gt_depth.dim() == 3:
                    gt_depth = gt_depth.squeeze(0)
                
                # Verify dimensions match (should be 448×448 after preprocessing)
                if gt_depth.shape != pred_depth.shape:
                    print(f"  Warning: GT depth shape {gt_depth.shape} != pred shape {pred_depth.shape}")
                    print(f"           Resizing GT depth (this shouldn't happen if preprocessing is correct)")
                    import torch.nn.functional as F
                    gt_depth_4d = gt_depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    gt_depth_resized = F.interpolate(
                        gt_depth_4d,
                        size=pred_depth.shape,
                        mode='nearest'
                    )
                    gt_depth = gt_depth_resized.squeeze(0).squeeze(0)
                    
                gt_depths.append(gt_depth)
            else:
                print(f"  Warning: GT depth not found: {gt_path}")
                gt_depths.append(None)
    
    return pred_depths, gt_depths, depth_names

def compute_depth_metrics(pred, gt, min_depth=0.1, max_depth=100.0, use_scale_alignment=True):
    """
    Compute depth evaluation metrics following VGGT/DUSt3R standard.
    
    IMPORTANT: This implementation exactly matches the VGGT losses.py calculate_depth_metrics()
    for consistent comparison across methods.
    
    Metrics computed:
    - abs_rel (rel): Absolute Relative Error (main metric in LSM)
    - tau_103 (inlier_ratio): Inlier Ratio with threshold 1.03 (main metric in LSM)
    - sq_rel: Squared relative error  
    - rmse: Root mean squared error
    - rmse_log: RMSE of log depth
    - a1, a2, a3: Accuracy under thresholds 1.25, 1.25^2, 1.25^3 (Eigen et al.)
    
    Args:
        pred: Predicted depth map (H, W)
        gt: Ground truth depth map (H, W)
        min_depth: Minimum valid depth (not used, kept for API compatibility)
        max_depth: Maximum valid depth (not used, kept for API compatibility)
        use_scale_alignment: If True, apply median normalization (default True)
        
    Returns:
        dict: Dictionary of computed metrics, or None if no valid pixels
    """
    # Create mask - EXACT MATCH to VGGT losses.py line 50
    mask = (gt > 0) & (pred > 0)
    
    # Apply mask
    gt_depth_masked = gt[mask]
    pred_depth_masked = pred[mask]
    
    # Avoid division by zero and handle empty masks
    if gt_depth_masked.numel() == 0 or pred_depth_masked.numel() == 0:
        return None
    
    # Calculate medians
    median_gt = torch.median(gt_depth_masked)
    median_pred = torch.median(pred_depth_masked)
    
    if torch.isclose(median_pred, torch.tensor(0.0)):
        return None
    
    # Scale alignment using median normalization - EXACT MATCH to VGGT
    # Scale pred to match GT (lines 67-70 in VGGT losses.py)
    scale = median_gt / median_pred
    pred_depth_masked = pred_depth_masked * scale
    
    # === Primary Metrics (matching VGGT exactly) ===
    
    # Absolute relative error - EXACT MATCH to VGGT line 73
    rel_err = torch.abs(gt_depth_masked - pred_depth_masked) / gt_depth_masked
    
    # Avoid NaN in relative error - EXACT MATCH to VGGT line 76
    rel_err[torch.isnan(rel_err)] = 0
    
    # Tau metric - EXACT MATCH to VGGT lines 78-81
    ratio = torch.max(pred_depth_masked / gt_depth_masked, 
                      gt_depth_masked / pred_depth_masked)
    tau_103 = (ratio < 1.03).float().mean()
    
    # === Additional Standard Metrics ===
    
    # Squared relative error
    sq_rel = torch.mean(((gt_depth_masked - pred_depth_masked) ** 2) / gt_depth_masked)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((gt_depth_masked - pred_depth_masked) ** 2))
    
    # RMSE log (add small epsilon to avoid log(0))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt_depth_masked + 1e-8) - 
                                      torch.log(pred_depth_masked + 1e-8)) ** 2))
    
    # Threshold accuracies (Eigen et al. 2014)
    thresh = torch.max((gt_depth_masked / (pred_depth_masked + 1e-8)), 
                       (pred_depth_masked / (gt_depth_masked + 1e-8)))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    # Return metrics (multiply by 100 to match VGGT format)
    return {
        'abs_rel': (rel_err.mean() * 100).item(),  # EXACT MATCH to VGGT return format
        'tau_103': (tau_103 * 100).item(),         # EXACT MATCH to VGGT return format
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item() * 100.0,
        'a2': a2.item() * 100.0,
        'a3': a3.item() * 100.0
    }

def evaluate(model_paths, eval_depth=False, gt_depth_dir=None, min_depth=0.1, max_depth=100.0, 
             json_split_path=None, case_id=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                # Evaluate depth if requested (on TRAINING views)
                if eval_depth:
                    # Look for depth in train directory, not test directory
                    train_dir = Path(scene_dir) / "train"
                    if train_dir.exists():
                        train_method_dir = train_dir / method
                        depth_raw_dir = train_method_dir / "depth_raw"
                        
                        if depth_raw_dir.exists():
                            print("\n  Evaluating depth metrics on TRAINING views...")
                            
                            # Determine GT depth directory
                            gt_depth_path = None
                            if gt_depth_dir:
                                gt_depth_path = Path(gt_depth_dir)
                            
                            pred_depths, gt_depths, depth_names = readDepthMaps(
                                depth_raw_dir, gt_depth_path, 
                                json_split_path=json_split_path, 
                                case_id=case_id,
                                use_train_views=True  # Use training views for depth
                            )
                            
                            if gt_depths is not None:
                                # Compute depth metrics
                                abs_rels = []
                                tau_103s = []
                                sq_rels = []
                                rmses = []
                                rmse_logs = []
                                a1s = []
                                a2s = []
                                a3s = []
                                depth_per_view = {}
                                
                                for pred_depth, gt_depth, depth_name in tqdm(zip(pred_depths, gt_depths, depth_names), 
                                                                             total=len(pred_depths),
                                                                             desc="Depth metrics"):
                                    if gt_depth is None:
                                        continue
                                        
                                    metrics = compute_depth_metrics(pred_depth, gt_depth, 
                                                                   min_depth=min_depth, max_depth=max_depth,
                                                                   use_scale_alignment=True)  # LSM uses scale alignment
                                    if metrics is not None:
                                        abs_rels.append(metrics['abs_rel'])
                                        tau_103s.append(metrics['tau_103'])
                                        sq_rels.append(metrics['sq_rel'])
                                        rmses.append(metrics['rmse'])
                                        rmse_logs.append(metrics['rmse_log'])
                                        a1s.append(metrics['a1'])
                                        a2s.append(metrics['a2'])
                                        a3s.append(metrics['a3'])
                                        depth_per_view[depth_name] = metrics
                                
                                if abs_rels:
                                    # Compute mean metrics
                                    mean_abs_rel = np.mean(abs_rels)
                                    mean_tau_103 = np.mean(tau_103s)
                                    mean_sq_rel = np.mean(sq_rels)
                                    mean_rmse = np.mean(rmses)
                                    mean_rmse_log = np.mean(rmse_logs)
                                    mean_a1 = np.mean(a1s)
                                    mean_a2 = np.mean(a2s)
                                    mean_a3 = np.mean(a3s)
                                    
                                    print(f"\n  === LSM/DUSt3R Metrics (with scale alignment) ===")
                                    print(f"  Abs Rel (rel) : {mean_abs_rel:>12.4f}%")
                                    print(f"  Inlier τ<1.03 : {mean_tau_103:>12.4f}%")
                                    print(f"\n  === Additional Metrics ===")
                                    print(f"  Sq Rel        : {mean_sq_rel:>12.7f}")
                                    print(f"  RMSE          : {mean_rmse:>12.4f} m")
                                    print(f"  RMSE log      : {mean_rmse_log:>12.7f}")
                                    print(f"  δ < 1.25      : {mean_a1:>12.4f}%")
                                    print(f"  δ < 1.25²     : {mean_a2:>12.4f}%")
                                    print(f"  δ < 1.25³     : {mean_a3:>12.4f}%")
                                    
                                    # Add to full dict (prioritize LSM metrics)
                                    full_dict[scene_dir][method].update({
                                        "depth_abs_rel": mean_abs_rel,
                                        "depth_tau_103": mean_tau_103,
                                        "depth_sq_rel": mean_sq_rel,
                                        "depth_rmse": mean_rmse,
                                        "depth_rmse_log": mean_rmse_log,
                                        "depth_a1": mean_a1,
                                        "depth_a2": mean_a2,
                                        "depth_a3": mean_a3
                                    })
                                    
                                    # Add to per-view dict
                                    per_view_dict[scene_dir][method]["depth_metrics"] = depth_per_view
                                else:
                                    print("  No valid depth metrics computed")
                            else:
                                print("  GT depth directory not provided, skipping depth evaluation")
                        else:
                            print(f"  Depth directory not found: {depth_raw_dir}")
                    else:
                        print(f"  Train directory not found: {train_dir}")
                
                print("")

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script for rendering and depth metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[],
                       help="Paths to model output directories to evaluate")
    parser.add_argument('--eval_depth', action='store_true',
                       help="Enable depth metrics evaluation")
    parser.add_argument('--gt_depth_dir', type=str, default=None,
                       help="Path to ground truth depth directory (required for depth evaluation)")
    parser.add_argument('--min_depth', type=float, default=0.1,
                       help="Minimum valid depth value (default: 0.1m per LSM paper)")
    parser.add_argument('--max_depth', type=float, default=100.0,
                       help="Maximum valid depth value (default: 100m per LSM paper)")
    parser.add_argument('--json_split_path', type=str, default=None,
                       help="Path to JSON split file for mapping test view indices to frame IDs")
    parser.add_argument('--case_id', type=int, default=None,
                       help="Case ID for looking up test views in JSON split file")
    args = parser.parse_args()
    
    evaluate(args.model_paths, 
            eval_depth=args.eval_depth,
            gt_depth_dir=args.gt_depth_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            json_split_path=args.json_split_path,
            case_id=args.case_id)
