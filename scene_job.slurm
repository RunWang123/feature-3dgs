#!/bin/bash
#SBATCH -J feature-3dgs-scene-processing
#SBATCH -p mri2020
#SBATCH --time=24:00:00
#SBATCH --nodes=1                     # 1 nodes for mini dataset
#SBATCH --ntasks-per-node=1           # 1 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb                   # Reduced memory
#SBATCH --gpus-per-node=a100:1        # 1 A100 GPUs
#SBATCH -o /scratch/runw/project/colmap/work_dirs/palmetto_log/slurm_%j.out
#SBATCH -e /scratch/runw/project/colmap/work_dirs/palmetto_log/slurm_%j.err


# ============================================================================
# SLURM Job Template for Feature-3DGS Scene Processing
# 
# This script is called by submit_all_scenes.sh with environment variables:
#   - SCENE_NAME: Name of the scene to process
#   - JSON_SPLIT_PATH: Path to JSON split file
#   - DATA_BASE_DIR: Base directory for scene data
#   - OUTPUT_BASE_DIR: Base directory for outputs
# ============================================================================

# Print job info
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo ""

# Validate environment variables
if [ -z "${SCENE_NAME}" ] || [ -z "${JSON_SPLIT_PATH}" ] || [ -z "${DATA_BASE_DIR}" ] || [ -z "${OUTPUT_BASE_DIR}" ]; then
    echo "❌ Error: Required environment variables not set"
    echo "   Expected: SCENE_NAME, JSON_SPLIT_PATH, DATA_BASE_DIR, OUTPUT_BASE_DIR"
    exit 1
fi

echo "Scene: ${SCENE_NAME}"
echo "JSON Split: ${JSON_SPLIT_PATH}"
echo "Data Base: ${DATA_BASE_DIR}"
echo "Output Base: ${OUTPUT_BASE_DIR}"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

# Activate conda environment
if [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda/etc/profile.d/conda.sh"
    conda activate feature_3dgs
    echo "✅ Conda environment activated: feature_3dgs"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate feature_3dgs
    echo "✅ Conda environment activated: feature_3dgs"
else
    echo "⚠️  Warning: Could not find conda initialization script"
    echo "   Assuming environment is already activated"
fi

# Print environment info
echo ""
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""

# ============================================================================
# Run Processing Script
# ============================================================================

# Use SLURM_SUBMIT_DIR (directory where sbatch was called) instead of BASH_SOURCE
# because sbatch copies the script to a temp location
if [ -z "${SLURM_SUBMIT_DIR}" ]; then
    # Fallback if not running under SLURM (shouldn't happen)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
fi

PROCESS_SCRIPT="${SCRIPT_DIR}/process_single_scene.sh"

if [ ! -f "${PROCESS_SCRIPT}" ]; then
    echo "❌ Error: Processing script not found: ${PROCESS_SCRIPT}"
    echo "   SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
    echo "   SCRIPT_DIR: ${SCRIPT_DIR}"
    exit 1
fi

echo "Using processing script: ${PROCESS_SCRIPT}"
echo ""

echo "========================================"
echo "Starting Scene Processing"
echo "========================================"
echo ""

bash "${PROCESS_SCRIPT}" \
    "${SCENE_NAME}" \
    "${JSON_SPLIT_PATH}" \
    "${DATA_BASE_DIR}" \
    "${OUTPUT_BASE_DIR}"

EXIT_CODE=$?

# ============================================================================
# Job Summary
# ============================================================================

echo ""
echo "========================================"
echo "SLURM Job Complete"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Scene: ${SCENE_NAME}"
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Job completed successfully"
else
    echo "❌ Job failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}

