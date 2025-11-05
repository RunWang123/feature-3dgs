#!/bin/bash
#
# Process all cases for a single scene
# Usage: bash process_single_scene.sh <scene_name> <json_split_path> <data_base_dir> <output_base_dir>
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <scene_name> <json_split_path> <data_base_dir> <output_base_dir>"
    echo ""
    echo "Example:"
    echo "  $0 scene0686_01 /path/to/split.json /scratch/data /scratch/output"
    exit 1
fi

SCENE_NAME="$1"
JSON_SPLIT_PATH="$2"
DATA_BASE_DIR="$3"
OUTPUT_BASE_DIR="$4"

SCENE_DATA_DIR="${DATA_BASE_DIR}/${SCENE_NAME}"
LSEG_WEIGHTS="/home/runw/project/feature-3dgs/encoders/lseg_encoder/demo_e200.ckpt"
ENCODE_SCRIPT_DIR="/home/runw/project/feature-3dgs/encoders/lseg_encoder"
FEATURE_3DGS_DIR="/home/runw/project/feature-3dgs"

# Training parameters
ITERATIONS=7000
SEMANTIC_LABELS="wall,floor,ceiling,chair,table,sofa,bed,other"

# ============================================================================
# Validate inputs
# ============================================================================

if [ ! -d "${SCENE_DATA_DIR}" ]; then
    echo "❌ Error: Scene data directory not found: ${SCENE_DATA_DIR}"
    exit 1
fi

if [ ! -f "${JSON_SPLIT_PATH}" ]; then
    echo "❌ Error: JSON split file not found: ${JSON_SPLIT_PATH}"
    exit 1
fi

# ============================================================================
# Extract number of cases from JSON
# ============================================================================

echo "========================================"
echo "Processing Scene: ${SCENE_NAME}"
echo "========================================"
echo "Data directory: ${SCENE_DATA_DIR}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "JSON split: ${JSON_SPLIT_PATH}"
echo ""

# Get number of cases for this scene
NUM_CASES=$(python3 -c "
import json
with open('${JSON_SPLIT_PATH}', 'r') as f:
    data = json.load(f)
if '${SCENE_NAME}' in data['scenes']:
    print(len(data['scenes']['${SCENE_NAME}']))
else:
    print(0)
")

if [ "${NUM_CASES}" -eq 0 ]; then
    echo "❌ Error: Scene ${SCENE_NAME} not found in JSON or has no cases"
    exit 1
fi

echo "Found ${NUM_CASES} cases for scene ${SCENE_NAME}"
echo ""

# ============================================================================
# Step 1: Extract LSeg Features (ONCE per scene, not per case)
# ============================================================================

echo "========================================"
echo "Step 1: Extract LSeg Features"
echo "========================================"

LSEG_OUTPUT_DIR="${SCENE_DATA_DIR}/rgb_feature_langseg"

if [ -d "${LSEG_OUTPUT_DIR}" ]; then
    FEATURE_COUNT=$(find "${LSEG_OUTPUT_DIR}" -type f -name "*_fmap_CxHxW.pt" 2>/dev/null | wc -l)
    IMG_COUNT=$(find "${SCENE_DATA_DIR}/images" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    
    if [ "${FEATURE_COUNT}" -eq "${IMG_COUNT}" ] && [ "${IMG_COUNT}" -gt 0 ]; then
        echo "✅ LSeg features already exist (${FEATURE_COUNT} files)"
        echo "   Skipping feature extraction"
    else
        echo "⚠️  Incomplete features (${FEATURE_COUNT}/${IMG_COUNT}), re-extracting..."
        rm -rf "${LSEG_OUTPUT_DIR}"
        mkdir -p "${LSEG_OUTPUT_DIR}"
        
        # Run from lseg_encoder directory (needs relative paths to label_files, weights)
        cd "${ENCODE_SCRIPT_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        python -u encode_images.py \
            --backbone clip_vitl16_384 \
            --weights demo_e200.ckpt \
            --widehead \
            --no-scaleinv \
            --outdir "${LSEG_OUTPUT_DIR}" \
            --test-rgb-dir "${SCENE_DATA_DIR}/images" \
            --workers 0
        
        # Return to feature-3dgs root
        cd "${FEATURE_3DGS_DIR}" || exit 1
    fi
else
    echo "Extracting LSeg features..."
    mkdir -p "${LSEG_OUTPUT_DIR}"
    
    # Run from lseg_encoder directory (needs relative paths to label_files, weights)
    cd "${ENCODE_SCRIPT_DIR}" || exit 1
    echo "Working directory: $(pwd)"
    
    python -u encode_images.py \
        --backbone clip_vitl16_384 \
        --weights demo_e200.ckpt \
        --widehead \
        --no-scaleinv \
        --outdir "${LSEG_OUTPUT_DIR}" \
        --test-rgb-dir "${SCENE_DATA_DIR}/images" \
        --workers 0
    
    # Return to feature-3dgs root
    cd "${FEATURE_3DGS_DIR}" || exit 1
    
    echo "✅ LSeg features extracted"
fi

echo ""

# ============================================================================
# Step 2: Download timm model (once, if not already cached)
# ============================================================================

echo "========================================"
echo "Step 2: Ensure timm model is cached"
echo "========================================"

# Run from feature-3dgs root directory
cd "${FEATURE_3DGS_DIR}" || exit 1

python -c "
import timm
try:
    model = timm.create_model('vit_large_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
    print('✅ timm model available')
except Exception as e:
    print(f'⚠️  Error loading timm model: {e}')
" || echo "⚠️  Warning: timm model check failed (will try again during segmentation)"

echo ""

# ============================================================================
# Step 3-7: Process each case
# ============================================================================

SUCCESSFUL_CASES=0
FAILED_CASES=0

for CASE_ID in $(seq 0 $((NUM_CASES - 1))); do
    echo "========================================"
    echo "Processing Case ${CASE_ID} / ${NUM_CASES}"
    echo "========================================"
    
    CASE_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SCENE_NAME}_case${CASE_ID}"
    mkdir -p "${CASE_OUTPUT_DIR}"
    
    # Log file for this case
    CASE_LOG="${CASE_OUTPUT_DIR}/processing.log"
    echo "Log file: ${CASE_LOG}"
    echo ""
    
    {
        echo "Starting case ${CASE_ID} at $(date)"
        
        # ----------------------------------------------------------------
        # Step 3: Train
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 3: Training (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from feature-3dgs root directory
        cd "${FEATURE_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        python train.py \
            -s "${SCENE_DATA_DIR}" \
            -m "${CASE_OUTPUT_DIR}" \
            -f lseg \
            --speedup \
            --iterations ${ITERATIONS} \
            --eval \
            --json_split_path "${JSON_SPLIT_PATH}" \
            --case_id ${CASE_ID} \
            2>&1
        
        TRAIN_STATUS=$?
        if [ ${TRAIN_STATUS} -ne 0 ]; then
            echo "❌ Training failed for case ${CASE_ID}"
            FAILED_CASES=$((FAILED_CASES + 1))
            continue
        fi
        
        echo "✅ Training completed"
        
        # ----------------------------------------------------------------
        # Step 4: Render
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 4: Rendering (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from feature-3dgs root directory
        cd "${FEATURE_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        python render.py \
            -s "${SCENE_DATA_DIR}" \
            -m "${CASE_OUTPUT_DIR}" \
            --iteration ${ITERATIONS} \
            --eval \
            --json_split_path "${JSON_SPLIT_PATH}" \
            --case_id ${CASE_ID} \
            2>&1
        
        RENDER_STATUS=$?
        if [ ${RENDER_STATUS} -ne 0 ]; then
            echo "❌ Rendering failed for case ${CASE_ID}"
            FAILED_CASES=$((FAILED_CASES + 1))
            continue
        fi
        
        echo "✅ Rendering completed"
        
        # ----------------------------------------------------------------
        # Step 5: Metrics
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 5: Computing Metrics (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from feature-3dgs root directory
        cd "${FEATURE_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        python metrics.py -m "${CASE_OUTPUT_DIR}" 2>&1
        
        METRICS_STATUS=$?
        if [ ${METRICS_STATUS} -ne 0 ]; then
            echo "⚠️  Metrics computation failed for case ${CASE_ID}"
        else
            echo "✅ Metrics computed"
            
            # Display results
            if [ -f "${CASE_OUTPUT_DIR}/results.json" ]; then
                echo ""
                echo "Results:"
                cat "${CASE_OUTPUT_DIR}/results.json"
                echo ""
            fi
        fi
        
        # ----------------------------------------------------------------
        # Step 6: Segmentation
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 6: Segmentation (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from lseg_encoder directory (needs relative paths to modules, weights)
        cd "${FEATURE_3DGS_DIR}/encoders/lseg_encoder" || exit 1
        echo "Working directory: $(pwd)"
        
        python -u segmentation.py \
            --data "${CASE_OUTPUT_DIR}" \
            --iteration ${ITERATIONS} \
            --label_src "${SEMANTIC_LABELS}" \
            2>&1
        
        SEG_STATUS=$?
        if [ ${SEG_STATUS} -ne 0 ]; then
            echo "⚠️  Segmentation failed for case ${CASE_ID}"
        else
            echo "✅ Segmentation completed"
        fi
        
        # Return to feature-3dgs root
        cd "${FEATURE_3DGS_DIR}" || exit 1
        
        # ----------------------------------------------------------------
        # Case summary
        # ----------------------------------------------------------------
        echo ""
        echo "========================================="
        echo "Case ${CASE_ID} completed at $(date)"
        echo "========================================="
        
        SUCCESSFUL_CASES=$((SUCCESSFUL_CASES + 1))
        
    } 2>&1 | tee "${CASE_LOG}"
    
done

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "========================================"
echo "Scene ${SCENE_NAME} Processing Complete"
echo "========================================"
echo "Total cases: ${NUM_CASES}"
echo "Successful: ${SUCCESSFUL_CASES}"
echo "Failed: ${FAILED_CASES}"
echo ""
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo ""

if [ ${FAILED_CASES} -gt 0 ]; then
    echo "⚠️  Some cases failed. Check individual logs for details."
    exit 1
else
    echo "✅ All cases processed successfully!"
    exit 0
fi

