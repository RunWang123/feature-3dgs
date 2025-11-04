#!/bin/bash
#
# Batch LSeg Feature Extraction for All ScanNet Scenes
#
# This script processes all scenes in a directory with LSeg encoder
# to generate semantic features for training feature-3dgs.
#

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================================================

# Base directory containing all scene folders
DATASET_BASE="/scratch/runw/project/colmap/scannet_test_feature3dgs"

# LSeg model checkpoint path
LSEG_WEIGHTS="/home/runw/Project/feature-3dgs/encoders/lseg_encoder/demo_e200.ckpt"

# LSeg encoder script path
ENCODE_SCRIPT="/home/runw/Project/feature-3dgs/encoders/lseg_encoder/encode_images.py"

# Number of workers (0 = main thread only, safer for clusters)
WORKERS=0

# ============================================================================
# MAIN PROCESSING
# ============================================================================

echo "========================================"
echo "Batch LSeg Feature Extraction"
echo "========================================"
echo "Dataset base: ${DATASET_BASE}"
echo "LSeg weights: ${LSEG_WEIGHTS}"
echo ""

# Check if weights file exists
if [ ! -f "${LSEG_WEIGHTS}" ]; then
    echo "ERROR: LSeg weights not found at: ${LSEG_WEIGHTS}"
    echo "Please download demo_e200.ckpt first!"
    exit 1
fi

# Check if encode script exists
if [ ! -f "${ENCODE_SCRIPT}" ]; then
    echo "ERROR: Encode script not found at: ${ENCODE_SCRIPT}"
    exit 1
fi

# Find all scene directories
echo "Searching for scenes..."
SCENE_DIRS=($(find "${DATASET_BASE}" -mindepth 1 -maxdepth 1 -type d -name "scene*" | sort))

if [ ${#SCENE_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No scene directories found in ${DATASET_BASE}"
    echo "Looking for directories matching pattern: scene*"
    exit 1
fi

echo "Found ${#SCENE_DIRS[@]} scenes to process"
echo ""

# Process each scene
PROCESSED=0
SKIPPED=0
FAILED=0

for SCENE_DIR in "${SCENE_DIRS[@]}"; do
    SCENE_NAME=$(basename "${SCENE_DIR}")
    
    echo "========================================"
    echo "Processing: ${SCENE_NAME}"
    echo "========================================"
    
    # Define paths
    IMAGES_DIR="${SCENE_DIR}/images"
    OUTPUT_DIR="${SCENE_DIR}/rgb_feature_langseg"
    
    # Check if images directory exists
    if [ ! -d "${IMAGES_DIR}" ]; then
        echo "⚠️  WARNING: Images directory not found: ${IMAGES_DIR}"
        echo "    Skipping ${SCENE_NAME}"
        echo ""
        ((SKIPPED++))
        continue
    fi
    
    # Check if images exist
    IMG_COUNT=$(find "${IMAGES_DIR}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) | wc -l)
    if [ ${IMG_COUNT} -eq 0 ]; then
        echo "⚠️  WARNING: No images found in ${IMAGES_DIR}"
        echo "    Skipping ${SCENE_NAME}"
        echo ""
        ((SKIPPED++))
        continue
    fi
    
    echo "Found ${IMG_COUNT} images in ${IMAGES_DIR}"
    
    # Check if features already exist
    if [ -d "${OUTPUT_DIR}" ]; then
        FEATURE_COUNT=$(find "${OUTPUT_DIR}" -type f -name "*_fmap_CxHxW.pt" | wc -l)
        
        if [ ${FEATURE_COUNT} -eq ${IMG_COUNT} ]; then
            echo "✅ Features already exist (${FEATURE_COUNT} files)"
            echo "   Skipping ${SCENE_NAME} (delete ${OUTPUT_DIR} to reprocess)"
            echo ""
            ((SKIPPED++))
            continue
        elif [ ${FEATURE_COUNT} -gt 0 ]; then
            echo "⚠️  Partial features found (${FEATURE_COUNT}/${IMG_COUNT})"
            echo "   Re-processing ${SCENE_NAME}..."
        fi
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run LSeg encoder
    echo "Running LSeg encoder..."
    echo "Command: python -u ${ENCODE_SCRIPT} --backbone clip_vitl16_384 --weights ${LSEG_WEIGHTS} --widehead --no-scaleinv --outdir ${OUTPUT_DIR} --test-rgb-dir ${IMAGES_DIR} --workers ${WORKERS}"
    echo ""
    
    if python -u "${ENCODE_SCRIPT}" \
        --backbone clip_vitl16_384 \
        --weights "${LSEG_WEIGHTS}" \
        --widehead \
        --no-scaleinv \
        --outdir "${OUTPUT_DIR}" \
        --test-rgb-dir "${IMAGES_DIR}" \
        --workers ${WORKERS}; then
        
        # Verify output
        FEATURE_COUNT=$(find "${OUTPUT_DIR}" -type f -name "*_fmap_CxHxW.pt" | wc -l)
        
        if [ ${FEATURE_COUNT} -eq ${IMG_COUNT} ]; then
            echo "✅ SUCCESS: Generated ${FEATURE_COUNT} features for ${SCENE_NAME}"
            ((PROCESSED++))
        else
            echo "⚠️  WARNING: Feature count mismatch (${FEATURE_COUNT} features vs ${IMG_COUNT} images)"
            ((FAILED++))
        fi
    else
        echo "❌ FAILED: Error processing ${SCENE_NAME}"
        ((FAILED++))
    fi
    
    echo ""
done

# Summary
echo "========================================"
echo "BATCH PROCESSING COMPLETE"
echo "========================================"
echo "Total scenes: ${#SCENE_DIRS[@]}"
echo "Successfully processed: ${PROCESSED}"
echo "Skipped (already done): ${SKIPPED}"
echo "Failed: ${FAILED}"
echo "========================================"

if [ ${FAILED} -gt 0 ]; then
    exit 1
else
    exit 0
fi

