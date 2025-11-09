#!/usr/bin/env python3
"""
Aggregate test results from all scene_case directories.

This script processes results from feature-3dgs evaluation across multiple scenes and test cases.
It computes:
- Per-scene metrics (averaged across cases)
- Overall metrics (averaged across all scenes and cases)
- Statistics (mean, std, min, max)

Expected directory structure:
  output_dir/
    scene{id}_case{n}/
      results.json
      per_view.json

Usage:
  python aggregate_test_results.py --results_dir /path/to/output_dir
"""

import json
import os
import sys
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_results_json(json_path):
    """Load results.json file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


def load_segmentation_metrics_all_iterations(scene_case_dir):
    """
    Load segmentation metrics for ALL iterations (2000, 3000, 4000, 5000, 6000, 7000).
    Filters out metadata fields to keep only actual metrics.
    
    Returns:
        dict: {
            'iter_2000': {'train': {...}, 'test': {...}},
            'iter_3000': {'train': {...}, 'test': {...}},
            ...
        } or None
    """
    results_dir = scene_case_dir / 'results'
    if not results_dir.exists():
        return None
    
    # Extract scene name and case from directory name
    dir_name = scene_case_dir.name
    parts = dir_name.split('_case')
    if len(parts) != 2:
        return None
    
    scene_name = parts[0]
    case_id = parts[1]
    
    # All iterations to check
    iterations = ['2000', '3000', '4000', '5000', '6000', '7000']
    
    # Metadata fields to exclude (not actual metrics)
    metadata_fields = {
        'iteration', 'num_classes', 'num_samples', 'labels', 
        'split', 'scene_name', 'case_id', 'timestamp', 'model_path', 'scene_path'
    }
    
    def filter_metrics(data):
        """Extract only actual metric values, exclude metadata."""
        if not isinstance(data, dict):
            return {}
        
        filtered = {}
        
        # Check if metrics are nested under a "metrics" key (segmentation JSON format)
        if 'metrics' in data and isinstance(data['metrics'], dict):
            # Extract from nested metrics dict
            for metric_key, metric_val in data['metrics'].items():
                if isinstance(metric_val, (int, float)):
                    filtered[metric_key] = metric_val
        else:
            # Extract from flat structure (results.json format)
            for key, value in data.items():
                # Skip metadata fields
                if key in metadata_fields:
                    continue
                # Keep numeric metric values
                if isinstance(value, (int, float)):
                    filtered[key] = value
        
        return filtered
    
    all_seg_metrics = {}
    
    for iteration in iterations:
        # Try gt_metrics first (ScanNet), fall back to teacher_student_metrics (DL3DV)
        # GT metrics compare to ground truth labels (only available for ScanNet)
        # Teacher-student metrics compare to teacher model predictions (available for all datasets)
        
        gt_base = f"{scene_name}_case{case_id}_iter{iteration}_gt_metrics"
        ts_base = f"{scene_name}_case{case_id}_iter{iteration}_teacher_student_metrics"
        
        iter_metrics = {}
        
        # Load train split (try GT first, then teacher-student)
        train_file_gt = results_dir / f"{gt_base}_train.json"
        train_file_ts = results_dir / f"{ts_base}_train.json"
        
        train_loaded = False
        for train_file in [train_file_gt, train_file_ts]:
            if train_file.exists() and not train_loaded:
                try:
                    with open(train_file, 'r') as f:
                        data = json.load(f)
                        filtered = filter_metrics(data)
                        if filtered:
                            iter_metrics['train'] = filtered
                            train_loaded = True
                except Exception as e:
                    pass  # Try next file
        
        # Load test split (try GT first, then teacher-student)
        test_file_gt = results_dir / f"{gt_base}_test.json"
        test_file_ts = results_dir / f"{ts_base}_test.json"
        
        test_loaded = False
        for test_file in [test_file_gt, test_file_ts]:
            if test_file.exists() and not test_loaded:
                try:
                    with open(test_file, 'r') as f:
                        data = json.load(f)
                        filtered = filter_metrics(data)
                        if filtered:
                            iter_metrics['test'] = filtered
                            test_loaded = True
                except Exception as e:
                    pass  # Try next file
        
        if iter_metrics:
            all_seg_metrics[f'iter_{iteration}'] = iter_metrics
    
    return all_seg_metrics if all_seg_metrics else None


def extract_scene_and_case(dirname):
    """
    Extract scene name and case number from directory name.
    Example: 'scene0686_01_case5' -> ('scene0686_01', 5)
    """
    parts = dirname.split('_case')
    if len(parts) == 2:
        scene_name = parts[0]
        try:
            case_id = int(parts[1])
            return scene_name, case_id
        except ValueError:
            return None, None
    return None, None


def collect_all_results(results_dir):
    """
    Collect all results from scene_case directories.
    
    Loads:
    - results.json: RGB metrics (TEST split) and depth metrics (TRAIN split) for all iterations
    - segmentation metrics: Both TRAIN and TEST splits for all iterations
    
    Returns:
        dict: {scene_name: {case_id: {
            'rgb_and_depth': {...},  # From results.json
            'segmentation': {
                'iter_2000': {'train': {...}, 'test': {...}},
                'iter_3000': {'train': {...}, 'test': {...}},
                ...
            }
        }}}
    """
    results_dir = Path(results_dir)
    all_results = defaultdict(dict)
    
    # Find all scene_case directories
    scene_case_dirs = sorted([d for d in results_dir.iterdir() 
                             if d.is_dir() and '_case' in d.name])
    
    print(f"Found {len(scene_case_dirs)} scene_case directories")
    print(f"Collecting results for all iterations...\n")
    
    missing_rgb = 0
    missing_seg = 0
    
    for scene_case_dir in tqdm(scene_case_dirs, desc="Loading results"):
        scene_name, case_id = extract_scene_and_case(scene_case_dir.name)
        
        if scene_name is None:
            continue
        
        case_results = {}
        
        # Load RGB and depth metrics from results.json
        results_json_path = scene_case_dir / "results.json"
        if results_json_path.exists():
            rgb_depth_results = load_results_json(results_json_path)
            if rgb_depth_results is not None:
                case_results['rgb_and_depth'] = rgb_depth_results
        else:
            missing_rgb += 1
        
        # Load segmentation metrics for ALL iterations
        seg_metrics = load_segmentation_metrics_all_iterations(scene_case_dir)
        if seg_metrics:
            case_results['segmentation'] = seg_metrics
        else:
            missing_seg += 1
        
        if case_results:
            all_results[scene_name][case_id] = case_results
    
    if missing_rgb > 0:
        print(f"\n⚠️  Warning: {missing_rgb} cases missing results.json")
    if missing_seg > 0:
        print(f"⚠️  Warning: {missing_seg} cases missing segmentation metrics")
    
    return dict(all_results)


def compute_scene_statistics(scene_results):
    """
    Compute statistics for a single scene across all cases.
    
    Args:
        scene_results: dict {case_id: {
            'rgb_and_depth': {method: metrics},
            'segmentation': {
                'iter_2000': {'train': {...}, 'test': {...}},
                'iter_3000': {'train': {...}, 'test': {...}},
                ...
            }
        }}
    
    Returns:
        dict: {
            'rgb_test': {method: {metric: stats}},  # PSNR, SSIM, LPIPS on TEST views
            'depth_train': {method: {metric: stats}},  # Depth metrics on TRAIN views
            'seg_test': {iteration: {metric: stats}},  # Segmentation on TEST views per iteration
            'seg_train': {iteration: {metric: stats}}  # Segmentation on TRAIN views per iteration
        }
    """
    # Organize metrics by type and split
    rgb_metrics = defaultdict(lambda: defaultdict(list))
    depth_metrics = defaultdict(lambda: defaultdict(list))
    seg_test_metrics = defaultdict(lambda: defaultdict(list))  # iteration -> metric -> values
    seg_train_metrics = defaultdict(lambda: defaultdict(list))  # iteration -> metric -> values
    
    for case_id, case_data in scene_results.items():
        # Process RGB and depth from results.json
        if 'rgb_and_depth' in case_data:
            for method_name, metrics in case_data['rgb_and_depth'].items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        # Separate RGB metrics (test) from depth metrics (train)
                        if metric_name in ['PSNR', 'SSIM', 'LPIPS']:
                            rgb_metrics[method_name][metric_name].append(value)
                        elif 'depth' in metric_name.lower() or metric_name in ['abs_rel', 'tau_103', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
                            depth_metrics[method_name][metric_name].append(value)
        
        # Process segmentation metrics (per iteration)
        if 'segmentation' in case_data:
            for iter_key, iter_data in case_data['segmentation'].items():
                # Test split segmentation
                if 'test' in iter_data:
                    test_seg = iter_data['test']
                    for metric_name, value in test_seg.items():
                        if isinstance(value, (int, float)):
                            seg_test_metrics[iter_key][metric_name].append(value)
                
                # Train split segmentation
                if 'train' in iter_data:
                    train_seg = iter_data['train']
                    for metric_name, value in train_seg.items():
                        if isinstance(value, (int, float)):
                            seg_train_metrics[iter_key][metric_name].append(value)
    
    # Compute statistics
    def compute_stats(metrics_dict):
        stats = {}
        for key, metrics in metrics_dict.items():
            stats[key] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                if len(values) > 0:
                    stats[key][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values),
                        'values': values.tolist()
                    }
        return stats
    
    def compute_iter_stats(seg_metrics_dict):
        """Compute stats for per-iteration segmentation metrics."""
        stats = {}
        for iter_key, metrics in seg_metrics_dict.items():
            stats[iter_key] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                if len(values) > 0:
                    stats[iter_key][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values),
                        'values': values.tolist()
                    }
        return stats
    
    return {
        'rgb_test': compute_stats(rgb_metrics),
        'depth_train': compute_stats(depth_metrics),
        'seg_test': compute_iter_stats(seg_test_metrics),
        'seg_train': compute_iter_stats(seg_train_metrics)
    }


def compute_overall_statistics(all_scene_stats):
    """
    Compute overall statistics across all scenes.
    
    Args:
        all_scene_stats: dict {scene_name: {
            'rgb_test': {...},
            'depth_train': {...},
            'seg_test': {iteration: {metric: stats}},
            'seg_train': {iteration: {metric: stats}}
        }}
    
    Returns:
        dict: {
            'rgb_test': {method: {metric: stats}},
            'depth_train': {method: {metric: stats}},
            'seg_test': {iteration: {metric: stats}},
            'seg_train': {iteration: {metric: stats}}
        }
    """
    # Organize by split type
    rgb_overall = defaultdict(lambda: defaultdict(list))
    depth_overall = defaultdict(lambda: defaultdict(list))
    seg_test_overall = defaultdict(lambda: defaultdict(list))  # iteration -> metric -> values
    seg_train_overall = defaultdict(lambda: defaultdict(list))  # iteration -> metric -> values
    
    for scene_name, scene_stats in all_scene_stats.items():
        # RGB metrics (test split)
        if 'rgb_test' in scene_stats:
            for method_name, metrics in scene_stats['rgb_test'].items():
                for metric_name, stats in metrics.items():
                    rgb_overall[method_name][metric_name].append(stats['mean'])
        
        # Depth metrics (train split)
        if 'depth_train' in scene_stats:
            for method_name, metrics in scene_stats['depth_train'].items():
                for metric_name, stats in metrics.items():
                    depth_overall[method_name][metric_name].append(stats['mean'])
        
        # Segmentation test (per iteration)
        if 'seg_test' in scene_stats:
            for iter_key, metrics in scene_stats['seg_test'].items():
                for metric_name, stats in metrics.items():
                    seg_test_overall[iter_key][metric_name].append(stats['mean'])
        
        # Segmentation train (per iteration)
        if 'seg_train' in scene_stats:
            for iter_key, metrics in scene_stats['seg_train'].items():
                for metric_name, stats in metrics.items():
                    seg_train_overall[iter_key][metric_name].append(stats['mean'])
    
    # Compute overall statistics
    def compute_method_stats(metrics_dict):
        stats = {}
        for method_name, metrics in metrics_dict.items():
            stats[method_name] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                if len(values) > 0:
                    stats[method_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
        return stats
    
    def compute_iter_stats(seg_metrics_dict):
        """Compute stats for per-iteration segmentation metrics."""
        stats = {}
        for iter_key, metrics in seg_metrics_dict.items():
            stats[iter_key] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                if len(values) > 0:
                    stats[iter_key][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
        return stats
    
    return {
        'rgb_test': compute_method_stats(rgb_overall),
        'depth_train': compute_method_stats(depth_overall),
        'seg_test': compute_iter_stats(seg_test_overall),
        'seg_train': compute_iter_stats(seg_train_overall)
    }


def print_results_table(overall_stats, title="Overall Results"):
    """Print results in a formatted table with clear train/test split labels."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    # Print RGB metrics (TEST split)
    if 'rgb_test' in overall_stats and overall_stats['rgb_test']:
        print(f"\n{'RGB METRICS (TEST SPLIT - Novel View Synthesis)':^80}")
        print(f"{'='*80}")
        for method_name, metrics in overall_stats['rgb_test'].items():
            print(f"\n{method_name}:")
            print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {'-'*63}")
            
            for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
                if metric_name in metrics:
                    stat = metrics[metric_name]
                    print(f"  {metric_name:<15} {stat['mean']:>12.4f} {stat['std']:>12.4f} "
                          f"{stat['min']:>12.4f} {stat['max']:>12.4f}")
    
    # Print Depth metrics (TRAIN split)
    if 'depth_train' in overall_stats and overall_stats['depth_train']:
        print(f"\n{'DEPTH METRICS (TRAIN SPLIT - Training Views)':^80}")
        print(f"{'='*80}")
        for method_name, metrics in overall_stats['depth_train'].items():
            print(f"\n{method_name}:")
            print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {'-'*63}")
            
            depth_order = ['abs_rel', 'tau_103', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
            for metric_name in depth_order:
                if metric_name in metrics:
                    stat = metrics[metric_name]
                    print(f"  {metric_name:<15} {stat['mean']:>12.4f} {stat['std']:>12.4f} "
                          f"{stat['min']:>12.4f} {stat['max']:>12.4f}")
    
    # Print Segmentation metrics (TEST split) - per iteration
    if 'seg_test' in overall_stats and overall_stats['seg_test']:
        print(f"\n{'SEGMENTATION METRICS (TEST SPLIT - Novel Views)':^80}")
        print(f"{'='*80}")
        
        # Sort iterations
        iterations = sorted(overall_stats['seg_test'].keys())
        for iter_key in iterations:
            metrics = overall_stats['seg_test'][iter_key]
            iter_num = iter_key.replace('iter_', '')
            print(f"\n{iter_key} (Iteration {iter_num}):")
            print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {'-'*63}")
            
            for metric_name in sorted(metrics.keys()):
                stat = metrics[metric_name]
                print(f"  {metric_name:<15} {stat['mean']:>12.4f} {stat['std']:>12.4f} "
                      f"{stat['min']:>12.4f} {stat['max']:>12.4f}")
    
    # Print Segmentation metrics (TRAIN split) - per iteration
    if 'seg_train' in overall_stats and overall_stats['seg_train']:
        print(f"\n{'SEGMENTATION METRICS (TRAIN SPLIT - Reference Views)':^80}")
        print(f"{'='*80}")
        
        # Sort iterations
        iterations = sorted(overall_stats['seg_train'].keys())
        for iter_key in iterations:
            metrics = overall_stats['seg_train'][iter_key]
            iter_num = iter_key.replace('iter_', '')
            print(f"\n{iter_key} (Iteration {iter_num}):")
            print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {'-'*63}")
            
            for metric_name in sorted(metrics.keys()):
                stat = metrics[metric_name]
                print(f"  {metric_name:<15} {stat['mean']:>12.4f} {stat['std']:>12.4f} "
                      f"{stat['min']:>12.4f} {stat['max']:>12.4f}")


def save_summary_json(output_path, scene_stats, overall_stats, metadata):
    """Save complete summary to JSON file."""
    summary = {
        'metadata': metadata,
        'overall_statistics': overall_stats,
        'per_scene_statistics': scene_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Full summary saved to: {output_path}")


def save_latex_table(output_path, overall_stats):
    """Save results in LaTeX table format with clear train/test split labels."""
    with open(output_path, 'w') as f:
        f.write("% Feature-3DGS Test Results\n")
        f.write("% NOTE: RGB metrics are on TEST split, depth metrics on TRAIN split\n\n")
        
        # RGB metrics table (TEST split)
        if 'rgb_test' in overall_stats and overall_stats['rgb_test']:
            f.write("% RGB Metrics (Novel View Synthesis - TEST Split)\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\n")
            f.write("Metric & Mean & Std & Min & Max \\\\\n")
            f.write("\\hline\n")
            
            for method_name, metrics in overall_stats['rgb_test'].items():
                f.write(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{method_name}}}}} \\\\\n")
                f.write("\\hline\n")
                
                for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
                    if metric_name in metrics:
                        stat = metrics[metric_name]
                        f.write(f"{metric_name} & {stat['mean']:.4f} & {stat['std']:.4f} & "
                               f"{stat['min']:.4f} & {stat['max']:.4f} \\\\\n")
                
                f.write("\\hline\n")
            
            f.write("\\end{tabular}\n")
            f.write("\\caption{RGB metrics on TEST split (novel view synthesis)}\n")
            f.write("\\end{table}\n\n")
        
        # Segmentation table (TEST split) - per iteration
        if 'seg_test' in overall_stats and overall_stats['seg_test']:
            f.write("% Segmentation Metrics (TEST Split - per iteration)\n")
            for iter_key in sorted(overall_stats['seg_test'].keys()):
                metrics = overall_stats['seg_test'][iter_key]
                iteration = iter_key.replace('iter_', '')
                
                f.write(f"\\begin{{table}}[h]\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Metric & Mean & Std & Min & Max \\\\\n")
                f.write("\\hline\n")
                
                for metric_name in sorted(metrics.keys()):
                    stat = metrics[metric_name]
                    f.write(f"{metric_name} & {stat['mean']:.4f} & {stat['std']:.4f} & "
                           f"{stat['min']:.4f} & {stat['max']:.4f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write(f"\\caption{{Segmentation metrics on TEST split (iteration {iteration})}}\n")
                f.write("\\end{table}\n\n")
    
    print(f"✅ LaTeX table saved to: {output_path}")


def save_csv(output_path, overall_stats):
    """Save results in CSV format with split labels."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Split', 'Iteration', 'Method', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'])
        
        # RGB metrics (TEST split)
        if 'rgb_test' in overall_stats:
            for method_name, metrics in overall_stats['rgb_test'].items():
                iteration = method_name.replace('ours_', '')
                for metric_name, stat in sorted(metrics.items()):
                    writer.writerow([
                        'RGB_TEST',
                        iteration,
                        method_name,
                        metric_name,
                        f"{stat['mean']:.6f}",
                        f"{stat['std']:.6f}",
                        f"{stat['min']:.6f}",
                        f"{stat['max']:.6f}",
                        stat['count']
                    ])
        
        # Depth metrics (TRAIN split)
        if 'depth_train' in overall_stats:
            for method_name, metrics in overall_stats['depth_train'].items():
                iteration = method_name.replace('ours_', '')
                for metric_name, stat in sorted(metrics.items()):
                    writer.writerow([
                        'DEPTH_TRAIN',
                        iteration,
                        method_name,
                        metric_name,
                        f"{stat['mean']:.6f}",
                        f"{stat['std']:.6f}",
                        f"{stat['min']:.6f}",
                        f"{stat['max']:.6f}",
                        stat['count']
                    ])
        
        # Segmentation TEST (per iteration)
        if 'seg_test' in overall_stats:
            for iter_key, metrics in sorted(overall_stats['seg_test'].items()):
                iteration = iter_key.replace('iter_', '')
                for metric_name, stat in sorted(metrics.items()):
                    writer.writerow([
                        'SEG_TEST',
                        iteration,
                        '-',
                        metric_name,
                        f"{stat['mean']:.6f}",
                        f"{stat['std']:.6f}",
                        f"{stat['min']:.6f}",
                        f"{stat['max']:.6f}",
                        stat['count']
                    ])
        
        # Segmentation TRAIN (per iteration)
        if 'seg_train' in overall_stats:
            for iter_key, metrics in sorted(overall_stats['seg_train'].items()):
                iteration = iter_key.replace('iter_', '')
                for metric_name, stat in sorted(metrics.items()):
                    writer.writerow([
                        'SEG_TRAIN',
                        iteration,
                        '-',
                        metric_name,
                        f"{stat['mean']:.6f}",
                        f"{stat['std']:.6f}",
                        f"{stat['min']:.6f}",
                        f"{stat['max']:.6f}",
                        stat['count']
                    ])
    
    print(f"✅ CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate test results from feature-3dgs evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: Metrics are separated by train/test split:
  - RGB metrics (PSNR, SSIM, LPIPS): TEST split (novel view synthesis)
  - Depth metrics: TRAIN split (training/reference views)  
  - Segmentation metrics: Both TRAIN and TEST splits

This separation is important for fair comparison with other methods.
        """
    )
    parser.add_argument('--results_dir', '-r', type=str, required=True,
                       help='Directory containing scene_case subdirectories with results')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for summary files (default: results_dir)')
    parser.add_argument('--formats', nargs='+', default=['json', 'csv', 'latex'],
                       choices=['json', 'csv', 'latex'],
                       help='Output formats (default: all)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.is_dir():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Feature-3DGS Test Results Aggregation")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Iterations:        2000, 3000, 4000, 5000, 6000, 7000 (all)")
    print(f"{'='*80}\n")
    
    # Collect all results
    all_results = collect_all_results(results_dir)
    
    if not all_results:
        print("❌ No results found!")
        sys.exit(1)
    
    num_scenes = len(all_results)
    num_cases = sum(len(cases) for cases in all_results.values())
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total scenes: {num_scenes}")
    print(f"  Total cases:  {num_cases}")
    print(f"  Average cases per scene: {num_cases / num_scenes:.1f}")
    
    # Compute per-scene statistics
    print(f"\n📈 Computing per-scene statistics...")
    scene_stats = {}
    for scene_name, scene_results in tqdm(all_results.items(), desc="Processing scenes"):
        scene_stats[scene_name] = compute_scene_statistics(scene_results)
    
    # Compute overall statistics
    print(f"\n📈 Computing overall statistics...")
    overall_stats = compute_overall_statistics(scene_stats)
    
    # Print results
    print_results_table(overall_stats, "Overall Results (Across All Scenes)")
    
    # Save outputs
    metadata = {
        'results_directory': str(results_dir),
        'num_scenes': num_scenes,
        'num_cases': num_cases,
        'scene_list': sorted(all_results.keys())
    }
    
    if 'json' in args.formats:
        save_summary_json(
            output_dir / 'summary_statistics.json',
            scene_stats,
            overall_stats,
            metadata
        )
    
    if 'csv' in args.formats:
        save_csv(output_dir / 'summary_statistics.csv', overall_stats)
    
    if 'latex' in args.formats:
        save_latex_table(output_dir / 'summary_table.tex', overall_stats)
    
    print(f"\n{'='*80}")
    print(f"✅ Aggregation Complete!")
    print(f"{'='*80}\n")
    
    # Print quick reference
    print("📁 Output files:")
    if 'json' in args.formats:
        print(f"  - summary_statistics.json   (detailed per-scene + overall stats)")
    if 'csv' in args.formats:
        print(f"  - summary_statistics.csv    (overall stats in CSV format)")
    if 'latex' in args.formats:
        print(f"  - summary_table.tex         (LaTeX table for papers)")
    print()
    print("⚠️  IMPORTANT - Train/Test Split Information:")
    print("  • RGB metrics (PSNR, SSIM, LPIPS): TEST split (novel views)")
    print("  • Depth metrics: TRAIN split (reference/training views)")
    print("  • Segmentation: Both TRAIN and TEST splits reported")
    print("  → Use TEST split metrics for paper comparisons!")
    print()


if __name__ == '__main__':
    main()

