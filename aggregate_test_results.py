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
    
    Returns:
        dict: {scene_name: {case_id: {method: metrics}}}
    """
    results_dir = Path(results_dir)
    all_results = defaultdict(dict)
    
    # Find all scene_case directories
    scene_case_dirs = sorted([d for d in results_dir.iterdir() 
                             if d.is_dir() and '_case' in d.name])
    
    print(f"Found {len(scene_case_dirs)} scene_case directories")
    print(f"Collecting results...\n")
    
    for scene_case_dir in tqdm(scene_case_dirs, desc="Loading results"):
        scene_name, case_id = extract_scene_and_case(scene_case_dir.name)
        
        if scene_name is None:
            continue
        
        results_json_path = scene_case_dir / "results.json"
        if not results_json_path.exists():
            print(f"Warning: {results_json_path} not found")
            continue
        
        results = load_results_json(results_json_path)
        if results is not None:
            all_results[scene_name][case_id] = results
    
    return dict(all_results)


def compute_scene_statistics(scene_results):
    """
    Compute statistics for a single scene across all cases.
    
    Args:
        scene_results: dict {case_id: {method: metrics}}
    
    Returns:
        dict: {method: {metric: {mean, std, min, max, values}}}
    """
    # Organize by method and metric
    method_metrics = defaultdict(lambda: defaultdict(list))
    
    for case_id, methods in scene_results.items():
        for method_name, metrics in methods.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    method_metrics[method_name][metric_name].append(value)
    
    # Compute statistics
    stats = {}
    for method_name, metrics in method_metrics.items():
        stats[method_name] = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            stats[method_name][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
                'values': values.tolist()
            }
    
    return stats


def compute_overall_statistics(all_scene_stats):
    """
    Compute overall statistics across all scenes.
    
    Args:
        all_scene_stats: dict {scene_name: {method: {metric: stats}}}
    
    Returns:
        dict: {method: {metric: {mean, std, min, max}}}
    """
    method_metrics = defaultdict(lambda: defaultdict(list))
    
    for scene_name, methods in all_scene_stats.items():
        for method_name, metrics in methods.items():
            for metric_name, stats in metrics.items():
                # Use the mean value from each scene
                method_metrics[method_name][metric_name].append(stats['mean'])
    
    # Compute overall statistics
    overall_stats = {}
    for method_name, metrics in method_metrics.items():
        overall_stats[method_name] = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            overall_stats[method_name][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    return overall_stats


def print_results_table(stats, title="Results"):
    """Print results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    for method_name, metrics in stats.items():
        print(f"\n{method_name}:")
        print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print(f"  {'-'*63}")
        
        # Sort metrics for consistent ordering
        metric_order = ['PSNR', 'SSIM', 'LPIPS', 'abs_rel', 'tau_103', 'sq_rel', 
                       'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'depth_abs_rel', 
                       'depth_tau_103', 'depth_sq_rel', 'depth_rmse', 
                       'depth_rmse_log', 'depth_a1', 'depth_a2', 'depth_a3']
        
        sorted_metrics = sorted(metrics.keys(), 
                               key=lambda x: metric_order.index(x) if x in metric_order else 999)
        
        for metric_name in sorted_metrics:
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
    """Save results in LaTeX table format."""
    with open(output_path, 'w') as f:
        f.write("% Feature-3DGS Test Results\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Metric & Mean & Std & Min & Max \\\\\n")
        f.write("\\hline\n")
        
        for method_name, metrics in overall_stats.items():
            f.write(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{method_name}}}}} \\\\\n")
            f.write("\\hline\n")
            
            metric_order = ['PSNR', 'SSIM', 'LPIPS', 'abs_rel', 'tau_103', 
                           'rmse', 'rmse_log', 'a1', 'a2', 'a3']
            
            for metric_name in metric_order:
                if metric_name in metrics:
                    stat = metrics[metric_name]
                    f.write(f"{metric_name} & {stat['mean']:.4f} & {stat['std']:.4f} & "
                           f"{stat['min']:.4f} & {stat['max']:.4f} \\\\\n")
            
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\caption{Feature-3DGS evaluation results on ScanNet test set}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved to: {output_path}")


def save_csv(output_path, overall_stats):
    """Save results in CSV format."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'])
        
        for method_name, metrics in overall_stats.items():
            for metric_name, stat in sorted(metrics.items()):
                writer.writerow([
                    method_name,
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
        formatter_class=argparse.RawDescriptionHelpFormatter
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


if __name__ == '__main__':
    main()

