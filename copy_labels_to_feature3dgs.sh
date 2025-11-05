#!/bin/bash
#
# Copy labels folders from scannet_test to scannet_test_feature3dgs
# Usage: bash copy_labels_to_feature3dgs.sh
#

set -e

# ============================================================================
# Configuration
# ============================================================================

SOURCE_DIR="/scratch/runw/project/colmap/scannet_test"
TARGET_DIR="/scratch/runw/project/colmap/scannet_test_feature3dgs"

# ============================================================================
# Validate directories
# ============================================================================

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "❌ Error: Source directory not found: ${SOURCE_DIR}"
    exit 1
fi

if [ ! -d "${TARGET_DIR}" ]; then
    echo "❌ Error: Target directory not found: ${TARGET_DIR}"
    exit 1
fi

# ============================================================================
# Process all scenes
# ============================================================================

echo "========================================"
echo "Copy Labels to Feature-3DGS Dataset"
echo "========================================"
echo "Source: ${SOURCE_DIR}"
echo "Target: ${TARGET_DIR}"
echo ""

TOTAL_SCENES=0
COPIED_SCENES=0
SKIPPED_SCENES=0
FAILED_SCENES=0

# Find all scene directories in source
for SOURCE_SCENE_DIR in "${SOURCE_DIR}"/scene*; do
    # Skip if not a directory
    if [ ! -d "${SOURCE_SCENE_DIR}" ]; then
        continue
    fi
    
    SCENE_NAME=$(basename "${SOURCE_SCENE_DIR}")
    TOTAL_SCENES=$((TOTAL_SCENES + 1))
    
    SOURCE_LABELS_DIR="${SOURCE_SCENE_DIR}/labels"
    TARGET_SCENE_DIR="${TARGET_DIR}/${SCENE_NAME}"
    TARGET_LABELS_DIR="${TARGET_SCENE_DIR}/labels"
    
    echo "----------------------------------------"
    echo "Scene ${TOTAL_SCENES}: ${SCENE_NAME}"
    echo "----------------------------------------"
    
    # Check if source labels exist
    if [ ! -d "${SOURCE_LABELS_DIR}" ]; then
        echo "⚠️  Source labels not found: ${SOURCE_LABELS_DIR}"
        SKIPPED_SCENES=$((SKIPPED_SCENES + 1))
        continue
    fi
    
    # Check if target scene directory exists
    if [ ! -d "${TARGET_SCENE_DIR}" ]; then
        echo "⚠️  Target scene directory not found: ${TARGET_SCENE_DIR}"
        SKIPPED_SCENES=$((SKIPPED_SCENES + 1))
        continue
    fi
    
    # Count source labels
    LABEL_COUNT=$(find "${SOURCE_LABELS_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    
    if [ "${LABEL_COUNT}" -eq 0 ]; then
        echo "⚠️  No label files found in ${SOURCE_LABELS_DIR}"
        SKIPPED_SCENES=$((SKIPPED_SCENES + 1))
        continue
    fi
    
    echo "Found ${LABEL_COUNT} label files"
    
    # Remove existing labels if they exist (overwrite)
    if [ -d "${TARGET_LABELS_DIR}" ]; then
        echo "Removing existing labels..."
        rm -rf "${TARGET_LABELS_DIR}"
    fi
    
    # Copy labels directory
    echo "Copying labels..."
    cp -r "${SOURCE_LABELS_DIR}" "${TARGET_LABELS_DIR}"
    
    # Verify copy
    COPIED_COUNT=$(find "${TARGET_LABELS_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    
    if [ "${COPIED_COUNT}" -eq "${LABEL_COUNT}" ]; then
        echo "✅ Successfully copied ${COPIED_COUNT} label files"
        COPIED_SCENES=$((COPIED_SCENES + 1))
    else
        echo "❌ Copy verification failed: expected ${LABEL_COUNT}, got ${COPIED_COUNT}"
        FAILED_SCENES=$((FAILED_SCENES + 1))
    fi
    
    echo ""
done

# ============================================================================
# Summary
# ============================================================================

echo "========================================"
echo "Copy Complete"
echo "========================================"
echo "Total scenes found: ${TOTAL_SCENES}"
echo "Successfully copied: ${COPIED_SCENES}"
echo "Skipped: ${SKIPPED_SCENES}"
echo "Failed: ${FAILED_SCENES}"
echo ""

if [ ${FAILED_SCENES} -gt 0 ]; then
    echo "⚠️  Some scenes failed to copy"
    exit 1
elif [ ${COPIED_SCENES} -eq 0 ]; then
    echo "⚠️  No scenes were copied"
    exit 1
else
    echo "✅ All labels copied successfully!"
    exit 0
fi

