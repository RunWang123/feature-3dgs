#!/bin/bash
# Complete analysis workflow for feature-3dgs test results
# Usage: ./run_full_analysis.sh <results_directory> [method_name]

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_directory> [method_name]"
    echo ""
    echo "Example:"
    echo "  $0 /scratch/runw/project/colmap/output/scannet_test_feature3dgs_2_to_1 ours_7000"
    exit 1
fi

RESULTS_DIR="$1"
METHOD="${2:-ours_7000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Feature-3DGS Complete Results Analysis"
echo "========================================================================"
echo "Results directory: $RESULTS_DIR"
echo "Method:            $METHOD"
echo "Scripts location:  $SCRIPT_DIR"
echo "========================================================================"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Create analysis output directory
ANALYSIS_DIR="${RESULTS_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"

# Step 1: Aggregate all results
echo "=========================================="
echo "Step 1: Aggregating Results"
echo "=========================================="
python3 "${SCRIPT_DIR}/aggregate_test_results.py" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$ANALYSIS_DIR" \
    --iteration "$(echo $METHOD | sed 's/ours_//')" \
    --formats json csv latex

if [ $? -ne 0 ]; then
    echo "❌ Error in aggregation step"
    exit 1
fi

echo ""
echo "✅ Aggregation complete!"
echo ""

# Step 2: Per-scene analysis
echo "=========================================="
echo "Step 2: Per-Scene Analysis"
echo "=========================================="
python3 "${SCRIPT_DIR}/analyze_per_scene_results.py" \
    --summary "${ANALYSIS_DIR}/summary_statistics.json" \
    --method "$METHOD" \
    --output_dir "$ANALYSIS_DIR" \
    --top_n 5

if [ $? -ne 0 ]; then
    echo "❌ Error in per-scene analysis step"
    exit 1
fi

echo ""
echo "✅ Per-scene analysis complete!"
echo ""

# Step 3: Generate plots (if matplotlib is available)
echo "=========================================="
echo "Step 3: Generating Plots (optional)"
echo "=========================================="

if python3 -c "import matplotlib" 2>/dev/null; then
    echo "matplotlib found, generating plots..."
    python3 "${SCRIPT_DIR}/analyze_per_scene_results.py" \
        --summary "${ANALYSIS_DIR}/summary_statistics.json" \
        --method "$METHOD" \
        --output_dir "$ANALYSIS_DIR" \
        --plot > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Plots generated successfully!"
    else
        echo "⚠️  Plot generation failed (non-fatal)"
    fi
else
    echo "⚠️  matplotlib not available, skipping plots"
    echo "    Install with: pip install matplotlib"
fi

echo ""
echo "========================================================================"
echo "✅ Complete Analysis Finished!"
echo "========================================================================"
echo ""
echo "Output files in: $ANALYSIS_DIR"
echo ""
echo "📁 Generated files:"
echo "  ├── summary_statistics.json      (complete statistics)"
echo "  ├── summary_statistics.csv       (CSV format)"
echo "  ├── summary_table.tex            (LaTeX table)"
echo "  ├── per_scene_${METHOD}.csv      (per-scene breakdown)"
if [ -d "${ANALYSIS_DIR}/plots" ]; then
echo "  └── plots/                       (visualization plots)"
echo "      ├── per_scene_psnr.png"
echo "      ├── per_scene_ssim.png"
echo "      ├── per_scene_lpips.png"
echo "      └── metric_distributions.png"
fi
echo ""
echo "📊 Quick Summary:"
python3 -c "
import json
with open('${ANALYSIS_DIR}/summary_statistics.json') as f:
    data = json.load(f)
    
if '$METHOD' in data['overall_statistics']:
    stats = data['overall_statistics']['$METHOD']
    print(f\"  Method: $METHOD\")
    print(f\"  Scenes: {data['metadata']['num_scenes']}\")
    print(f\"  Cases:  {data['metadata']['num_cases']}\")
    print(f\"\")
    if 'PSNR' in stats:
        print(f\"  PSNR:  {stats['PSNR']['mean']:.4f} ± {stats['PSNR']['std']:.4f}\")
    if 'SSIM' in stats:
        print(f\"  SSIM:  {stats['SSIM']['mean']:.4f} ± {stats['SSIM']['std']:.4f}\")
    if 'LPIPS' in stats:
        print(f\"  LPIPS: {stats['LPIPS']['mean']:.4f} ± {stats['LPIPS']['std']:.4f}\")
else:
    print(f\"  Method '$METHOD' not found in results\")
    print(f\"  Available methods: {list(data['overall_statistics'].keys())}\")
" 2>/dev/null || echo "  (Use summary_statistics.json for detailed metrics)"

echo ""
echo "========================================================================"
echo ""

