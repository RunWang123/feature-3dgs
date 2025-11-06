#!/usr/bin/env python3
"""
Detailed per-scene analysis of test results.

This script provides detailed breakdowns for each scene, showing:
- Per-scene metrics across all cases
- Best/worst performing scenes
- Case-by-case comparison
- Optional visualization plots

Usage:
  python analyze_per_scene_results.py --summary summary_statistics.json
"""

import json
import argparse
from pathlib import Path
import sys


def load_summary(json_path):
    """Load summary statistics JSON."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        sys.exit(1)


def print_per_scene_table(scene_stats, metric='PSNR', method='ours_7000'):
    """Print table showing metric values for all scenes."""
    print(f"\n{'='*80}")
    print(f"Per-Scene {metric} ({method})")
    print(f"{'='*80}")
    print(f"{'Scene':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Cases':>8}")
    print(f"{'-'*80}")
    
    # Collect scene data
    scene_data = []
    for scene_name, methods in scene_stats.items():
        if method in methods and metric in methods[method]:
            stats = methods[method][metric]
            scene_data.append((scene_name, stats))
    
    # Sort by mean value (descending for PSNR/SSIM, ascending for LPIPS/errors)
    reverse = metric in ['PSNR', 'SSIM', 'a1', 'a2', 'a3']
    scene_data.sort(key=lambda x: x[1]['mean'], reverse=reverse)
    
    # Print rows
    for scene_name, stats in scene_data:
        print(f"{scene_name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['count']:>8}")
    
    return scene_data


def print_top_bottom_scenes(scene_data, metric='PSNR', n=5):
    """Print top and bottom N scenes."""
    print(f"\n{'='*80}")
    print(f"Top {n} Scenes by {metric}:")
    print(f"{'='*80}")
    for i, (scene_name, stats) in enumerate(scene_data[:n], 1):
        print(f"{i}. {scene_name:<20} {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Bottom {n} Scenes by {metric}:")
    print(f"{'='*80}")
    for i, (scene_name, stats) in enumerate(scene_data[-n:], 1):
        print(f"{i}. {scene_name:<20} {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def print_metric_comparison(scene_stats, scene_name, method='ours_7000'):
    """Print all metrics for a specific scene."""
    if scene_name not in scene_stats:
        print(f"❌ Scene {scene_name} not found in results")
        return
    
    if method not in scene_stats[scene_name]:
        print(f"❌ Method {method} not found for scene {scene_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"Detailed Metrics for {scene_name} ({method})")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    
    metrics = scene_stats[scene_name][method]
    for metric_name in sorted(metrics.keys()):
        stats = metrics[metric_name]
        print(f"{metric_name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")


def export_per_scene_csv(scene_stats, output_path, method='ours_7000'):
    """Export per-scene results to CSV."""
    import csv
    
    # Collect all metrics
    all_metrics = set()
    for methods in scene_stats.values():
        if method in methods:
            all_metrics.update(methods[method].keys())
    
    all_metrics = sorted(all_metrics)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['scene']
        for metric in all_metrics:
            header.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max'])
        writer.writerow(header)
        
        # Data rows
        for scene_name in sorted(scene_stats.keys()):
            if method not in scene_stats[scene_name]:
                continue
            
            row = [scene_name]
            metrics = scene_stats[scene_name][method]
            
            for metric in all_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    row.extend([
                        f"{stats['mean']:.6f}",
                        f"{stats['std']:.6f}",
                        f"{stats['min']:.6f}",
                        f"{stats['max']:.6f}"
                    ])
                else:
                    row.extend(['', '', '', ''])
            
            writer.writerow(row)
    
    print(f"\n✅ Per-scene CSV saved to: {output_path}")


def create_plots(scene_stats, output_dir, method='ours_7000'):
    """Create visualization plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠️  matplotlib not available. Skipping plots.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for key metrics
    metrics_to_plot = ['PSNR', 'SSIM', 'LPIPS']
    
    for metric in metrics_to_plot:
        scene_names = []
        means = []
        stds = []
        
        for scene_name in sorted(scene_stats.keys()):
            if method in scene_stats[scene_name]:
                if metric in scene_stats[scene_name][method]:
                    scene_names.append(scene_name)
                    stats = scene_stats[scene_name][method][metric]
                    means.append(stats['mean'])
                    stds.append(stats['std'])
        
        if not means:
            continue
        
        # Create bar plot with error bars
        fig, ax = plt.subplots(figsize=(16, 6))
        x = np.arange(len(scene_names))
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
        
        # Color bars by performance
        if metric in ['PSNR', 'SSIM']:
            colors = plt.cm.RdYlGn(np.array(means) / max(means))
        else:  # LPIPS (lower is better)
            colors = plt.cm.RdYlGn(1 - np.array(means) / max(means))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Scene', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} per Scene ({method})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(scene_names, rotation=90, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'per_scene_{metric.lower()}.png', dpi=150)
        plt.close()
        
        print(f"✅ Plot saved: {output_dir / f'per_scene_{metric.lower()}.png'}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        scene_names = []
        means = []
        
        for scene_name in sorted(scene_stats.keys()):
            if method in scene_stats[scene_name]:
                if metric in scene_stats[scene_name][method]:
                    scene_names.append(scene_name)
                    means.append(scene_stats[scene_name][method][metric]['mean'])
        
        if means:
            axes[idx].hist(means, bins=20, alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(means), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(means):.4f}')
            axes[idx].set_xlabel(metric, fontsize=12)
            axes[idx].set_ylabel('Count', fontsize=12)
            axes[idx].set_title(f'{metric} Distribution', fontsize=14)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150)
    plt.close()
    
    print(f"✅ Plot saved: {output_dir / 'metric_distributions.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed per-scene analysis of test results"
    )
    parser.add_argument('--summary', '-s', type=str, required=True,
                       help='Path to summary_statistics.json')
    parser.add_argument('--method', '-m', type=str, default='ours_7000',
                       help='Method name to analyze (default: ours_7000)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for plots and CSV')
    parser.add_argument('--scene', type=str, default=None,
                       help='Show detailed metrics for specific scene')
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top/bottom scenes to show (default: 5)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots (requires matplotlib)')
    
    args = parser.parse_args()
    
    # Load summary
    print(f"\n{'='*80}")
    print(f"Per-Scene Analysis")
    print(f"{'='*80}")
    print(f"Loading: {args.summary}")
    
    summary = load_summary(args.summary)
    scene_stats = summary['per_scene_statistics']
    overall_stats = summary['overall_statistics']
    metadata = summary['metadata']
    
    print(f"\nDataset: {metadata['num_scenes']} scenes, {metadata['num_cases']} cases")
    print(f"Method: {args.method}")
    
    # Print overall stats first
    if args.method in overall_stats:
        print(f"\n{'='*80}")
        print(f"Overall Statistics ({args.method})")
        print(f"{'='*80}")
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'-'*70}")
        
        for metric_name in sorted(overall_stats[args.method].keys()):
            stats = overall_stats[args.method][metric_name]
            print(f"{metric_name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
    
    # Per-scene analysis for key metrics
    for metric in ['PSNR', 'SSIM', 'LPIPS']:
        scene_data = print_per_scene_table(scene_stats, metric, args.method)
        if scene_data:
            print_top_bottom_scenes(scene_data, metric, args.top_n)
    
    # Specific scene details
    if args.scene:
        print_metric_comparison(scene_stats, args.scene, args.method)
    
    # Export and plot
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export CSV
        export_per_scene_csv(scene_stats, 
                            output_dir / f'per_scene_{args.method}.csv',
                            args.method)
        
        # Generate plots
        if args.plot:
            create_plots(scene_stats, output_dir / 'plots', args.method)
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

