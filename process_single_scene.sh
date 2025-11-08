#!/bin/bash
#
# Process all cases for a single scene
# Usage: bash process_single_scene.sh <scene_name> <json_split_path> <data_base_dir> <output_base_dir>
#

# NOTE: We don't use 'set -e' here so that if one case fails,
# we continue processing other cases. Individual critical errors
# (like missing data directory) will still cause early exit.

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
SAVE_ITERATIONS="3000 7000"
TEST_ITERATIONS="3000 7000"
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
model = timm.create_model('vit_large_patch16_384.augreg_in21k_ft_in1k', pretrained=True)
print('✅ Model downloaded successfully!')
" || echo "⚠️  Warning: timm model download failed (will try again during segmentation)"

echo ""

# ============================================================================
# Step 3-8: Process each case
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
    
    # Track if this case has any errors (use file to avoid subshell issues)
    CASE_STATUS_FILE="${CASE_OUTPUT_DIR}/.case_status"
    echo "0" > "${CASE_STATUS_FILE}"  # 0 = success, 1 = failed
    
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
            --save_iterations ${SAVE_ITERATIONS} \
            --test_iterations ${TEST_ITERATIONS} \
            --eval \
            --json_split_path "${JSON_SPLIT_PATH}" \
            --case_id ${CASE_ID} \
            2>&1
        
        TRAIN_STATUS=$?
        if [ ${TRAIN_STATUS} -ne 0 ]; then
            echo "❌ Training failed for case ${CASE_ID}"
            echo "1" > "${CASE_STATUS_FILE}"  # Mark as failed
            exit 1  # Exit subshell to skip remaining steps
        fi
        
        echo "✅ Training completed"
        
        # ----------------------------------------------------------------
        # Step 4: Render (all saved iterations)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 4: Rendering (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from feature-3dgs root directory
        cd "${FEATURE_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        # Render all saved iterations
        for ITER in ${SAVE_ITERATIONS}; do
            echo "Rendering iteration ${ITER}..."
            
            python render.py \
                -s "${SCENE_DATA_DIR}" \
                -m "${CASE_OUTPUT_DIR}" \
                --iteration ${ITER} \
                --eval \
                --json_split_path "${JSON_SPLIT_PATH}" \
                --case_id ${CASE_ID} \
                2>&1
            
            RENDER_STATUS=$?
            if [ ${RENDER_STATUS} -ne 0 ]; then
                echo "⚠️  Rendering failed for iteration ${ITER}"
            else
                echo "✅ Rendering iteration ${ITER} completed"
            fi
        done
        
        echo "✅ All renderings completed"
        
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
        
        # Check if GT depth directory exists
        GT_DEPTH_DIR="${SCENE_DATA_DIR}/depths"
        if [ -d "${GT_DEPTH_DIR}" ]; then
            echo "Found GT depth directory: ${GT_DEPTH_DIR}"
            echo "Evaluating depth metrics on TRAINING views (LSM-aligned: 0.1m-100m, median norm)..."
            python metrics.py \
                -m "${CASE_OUTPUT_DIR}" \
                --eval_depth \
                --gt_depth_dir "${GT_DEPTH_DIR}" \
                --json_split_path "${JSON_SPLIT_PATH}" \
                --case_id ${CASE_ID} \
                2>&1
            # Note: Uses defaults (min=0.1m, max=100m) per LSM paper
            # Depth evaluated on ref_views (training), RGB on target_views (test)
        else
            echo "GT depth directory not found, computing RGB metrics only..."
            python metrics.py -m "${CASE_OUTPUT_DIR}" 2>&1
        fi
        
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
        # Step 6: Segmentation (all saved iterations)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 6: Segmentation (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from lseg_encoder directory (needs relative paths to modules, weights)
        cd "${FEATURE_3DGS_DIR}/encoders/lseg_encoder" || exit 1
        echo "Working directory: $(pwd)"
        
        # Segment all saved iterations
        for ITER in ${SAVE_ITERATIONS}; do
            echo "Segmenting iteration ${ITER}..."
            
            python -u segmentation.py \
                --data "${CASE_OUTPUT_DIR}" \
                --iteration ${ITER} \
                --label_src "${SEMANTIC_LABELS}" \
                2>&1
            
            SEG_STATUS=$?
            if [ ${SEG_STATUS} -ne 0 ]; then
                echo "⚠️  Segmentation failed for iteration ${ITER}"
            else
                echo "✅ Segmentation iteration ${ITER} completed"
            fi
        done
        
        echo "✅ All segmentations completed"
        
        # Return to feature-3dgs root
        cd "${FEATURE_3DGS_DIR}" || exit 1
        
        # ----------------------------------------------------------------
        # Step 7: Teacher-Student Metrics (all saved iterations, train + test)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 7: Teacher-Student Metrics (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Run from lseg_encoder directory
        cd "${FEATURE_3DGS_DIR}/encoders/lseg_encoder" || exit 1
        echo "Working directory: $(pwd)"
        
        mkdir -p "${CASE_OUTPUT_DIR}/results"
        
        # Compute metrics for all saved iterations and both splits
        for ITER in ${SAVE_ITERATIONS}; do
            echo "Computing Teacher-Student metrics for iteration ${ITER}..."
            
            # Test split
            echo "  - Test split..."
            python -u segmentation_metric.py \
                --backbone clip_vitl16_384 \
                --weights demo_e200.ckpt \
                --widehead \
                --no-scaleinv \
                --student-feature-dir "${CASE_OUTPUT_DIR}/test/ours_${ITER}/saved_feature/" \
                --teacher-feature-dir "${SCENE_DATA_DIR}/rgb_feature_langseg/" \
                --test-rgb-dir "${CASE_OUTPUT_DIR}/test/ours_${ITER}/renders/" \
                --workers 0 \
                --eval-mode test \
                --label_src "${SEMANTIC_LABELS}" \
                --output "${CASE_OUTPUT_DIR}/results/${SCENE_NAME}_case${CASE_ID}_iter${ITER}_teacher_student_metrics_test.json" \
                2>&1
            
            TEST_STATUS=$?
            if [ ${TEST_STATUS} -ne 0 ]; then
                echo "    ⚠️  Test split failed"
            else
                echo "    ✅ Test split completed"
            fi
            
            # Train split
            echo "  - Train split..."
            python -u segmentation_metric.py \
                --backbone clip_vitl16_384 \
                --weights demo_e200.ckpt \
                --widehead \
                --no-scaleinv \
                --student-feature-dir "${CASE_OUTPUT_DIR}/train/ours_${ITER}/saved_feature/" \
                --teacher-feature-dir "${SCENE_DATA_DIR}/rgb_feature_langseg/" \
                --test-rgb-dir "${CASE_OUTPUT_DIR}/train/ours_${ITER}/renders/" \
                --workers 0 \
                --eval-mode train \
                --label_src "${SEMANTIC_LABELS}" \
                --output "${CASE_OUTPUT_DIR}/results/${SCENE_NAME}_case${CASE_ID}_iter${ITER}_teacher_student_metrics_train.json" \
                2>&1
            
            TRAIN_STATUS=$?
            if [ ${TRAIN_STATUS} -ne 0 ]; then
                echo "    ⚠️  Train split failed"
            else
                echo "    ✅ Train split completed"
            fi
            
            # Overall status
            if [ ${TEST_STATUS} -eq 0 ] && [ ${TRAIN_STATUS} -eq 0 ]; then
                echo "✅ Teacher-Student metrics iteration ${ITER} completed (train + test)"
            else
                echo "⚠️  Teacher-Student metrics iteration ${ITER} partially failed"
            fi
        done
        
        echo "✅ All Teacher-Student metrics computed (train + test)"
        
        # ----------------------------------------------------------------
        # Step 8: Ground Truth Metrics (all saved iterations, train + test)
        # Note: segmentation_metric_gt.py internally processes both splits
        # Only run if GT labels exist (e.g., ScanNet has them, DL3DV doesn't)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 8: Ground Truth Metrics (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        # Check if GT label directory exists
        GT_LABEL_DIR="${SCENE_DATA_DIR}/label-filt"
        if [ ! -d "${GT_LABEL_DIR}" ]; then
            echo "ℹ️  Ground truth label directory not found: ${GT_LABEL_DIR}"
            echo "   Skipping GT segmentation metrics (dataset may not provide GT labels)"
            echo "   Teacher-Student metrics (Step 7) are the primary evaluation for this scene"
        else
            echo "✅ Found GT label directory: ${GT_LABEL_DIR}"
            
            # Check if label mapping file exists
            LABEL_MAPPING_FILE="${FEATURE_3DGS_DIR}/encoders/lseg_encoder/scannetv2-labels.combined.tsv"
            if [ ! -f "${LABEL_MAPPING_FILE}" ]; then
                echo "⚠️  Warning: Label mapping file not found: ${LABEL_MAPPING_FILE}"
                echo "   GT metrics will use identity mapping (may be incorrect for ScanNet)"
                LABEL_MAPPING_ARG=""
            else
                echo "✅ Using label mapping file: ${LABEL_MAPPING_FILE}"
                LABEL_MAPPING_ARG="--label_mapping_file scannetv2-labels.combined.tsv"
            fi
            
            # Run from lseg_encoder directory
            cd "${FEATURE_3DGS_DIR}/encoders/lseg_encoder" || exit 1
            echo "Working directory: $(pwd)"
            
            # Compute metrics for all saved iterations
            for ITER in ${SAVE_ITERATIONS}; do
                echo "Computing Ground Truth metrics for iteration ${ITER}..."
                
                python -u segmentation_metric_gt.py \
                    --weights demo_e200.ckpt \
                    --data "${CASE_OUTPUT_DIR}" \
                    --scene_data_path "${SCENE_DATA_DIR}" \
                    --json_split_path "${JSON_SPLIT_PATH}" \
                    --case_id ${CASE_ID} \
                    --label_src "${SEMANTIC_LABELS}" \
                    ${LABEL_MAPPING_ARG} \
                    --iteration ${ITER} \
                    --output "${CASE_OUTPUT_DIR}/results/${SCENE_NAME}_case${CASE_ID}_iter${ITER}_gt_metrics.json" \
                    2>&1
                
                GT_METRICS_STATUS=$?
                if [ ${GT_METRICS_STATUS} -ne 0 ]; then
                    echo "⚠️  Ground Truth metrics failed for iteration ${ITER}"
                else
                    echo "✅ Ground Truth metrics iteration ${ITER} computed"
                fi
            done
            
            echo "✅ All Ground Truth metrics computed"
            
            # Return to feature-3dgs root
            cd "${FEATURE_3DGS_DIR}" || exit 1
        fi
        
        # ----------------------------------------------------------------
        # Case summary
        # ----------------------------------------------------------------
        echo ""
        echo "========================================="
        echo "Case ${CASE_ID} completed at $(date)"
        echo "========================================="
        
    } 2>&1 | tee "${CASE_LOG}"
    
    # Check if case succeeded (read status from file to handle subshell)
    CASE_STATUS=$(cat "${CASE_STATUS_FILE}" 2>/dev/null || echo "1")
    if [ "${CASE_STATUS}" = "0" ]; then
        SUCCESSFUL_CASES=$((SUCCESSFUL_CASES + 1))
        echo "✅ Case ${CASE_ID}: SUCCESS"
    else
        FAILED_CASES=$((FAILED_CASES + 1))
        echo "❌ Case ${CASE_ID}: FAILED (check log: ${CASE_LOG})"
    fi
    echo ""
    
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

