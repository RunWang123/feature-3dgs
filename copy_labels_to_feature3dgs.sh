#!/bin/bash
#
# Copy labels and depths folders from scannet_test to scannet_test_feature3dgs
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
echo "Copy Labels & Depths to Feature-3DGS Dataset"
echo "========================================"
echo "Source: ${SOURCE_DIR}"
echo "Target: ${TARGET_DIR}"
echo ""

TOTAL_SCENES=0
COPIED_LABELS=0
COPIED_DEPTHS=0
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
    SOURCE_DEPTHS_DIR="${SOURCE_SCENE_DIR}/depths"
    TARGET_SCENE_DIR="${TARGET_DIR}/${SCENE_NAME}"
    TARGET_LABELS_DIR="${TARGET_SCENE_DIR}/labels"
    TARGET_DEPTHS_DIR="${TARGET_SCENE_DIR}/depths"
    
    echo "----------------------------------------"
    echo "Scene ${TOTAL_SCENES}: ${SCENE_NAME}"
    echo "----------------------------------------"
    
    # Check if target scene directory exists
    if [ ! -d "${TARGET_SCENE_DIR}" ]; then
        echo "⚠️  Target scene directory not found: ${TARGET_SCENE_DIR}"
        SKIPPED_SCENES=$((SKIPPED_SCENES + 1))
        continue
    fi
    
    SCENE_HAS_DATA=0
    
    # ========== Process Labels ==========
    if [ -d "${SOURCE_LABELS_DIR}" ]; then
        LABEL_COUNT=$(find "${SOURCE_LABELS_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        
        if [ "${LABEL_COUNT}" -gt 0 ]; then
            echo "📁 Labels: Found ${LABEL_COUNT} files"
            SCENE_HAS_DATA=1
            
            # Remove existing labels if they exist (overwrite)
            if [ -d "${TARGET_LABELS_DIR}" ]; then
                rm -rf "${TARGET_LABELS_DIR}"
            fi
            
            # Copy labels directory
            cp -r "${SOURCE_LABELS_DIR}" "${TARGET_LABELS_DIR}"
            
            # Verify copy
            COPIED_COUNT=$(find "${TARGET_LABELS_DIR}" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
            
            if [ "${COPIED_COUNT}" -eq "${LABEL_COUNT}" ]; then
                echo "   ✅ Copied ${COPIED_COUNT} label files"
                COPIED_LABELS=$((COPIED_LABELS + 1))
            else
                echo "   ❌ Copy failed: expected ${LABEL_COUNT}, got ${COPIED_COUNT}"
                FAILED_SCENES=$((FAILED_SCENES + 1))
            fi
        else
            echo "📁 Labels: Empty directory"
        fi
    else
        echo "📁 Labels: Not found"
    fi
    
    # ========== Process Depths ==========
    if [ -d "${SOURCE_DEPTHS_DIR}" ]; then
        DEPTH_COUNT=$(find "${SOURCE_DEPTHS_DIR}" -type f -name "*.png" 2>/dev/null | wc -l)
        
        if [ "${DEPTH_COUNT}" -gt 0 ]; then
            echo "📁 Depths: Found ${DEPTH_COUNT} files"
            SCENE_HAS_DATA=1
            
            # Remove existing depths if they exist (overwrite)
            if [ -d "${TARGET_DEPTHS_DIR}" ]; then
                rm -rf "${TARGET_DEPTHS_DIR}"
            fi
            
            # Copy depths directory
            cp -r "${SOURCE_DEPTHS_DIR}" "${TARGET_DEPTHS_DIR}"
            
            # Verify copy
            COPIED_COUNT=$(find "${TARGET_DEPTHS_DIR}" -type f -name "*.png" 2>/dev/null | wc -l)
            
            if [ "${COPIED_COUNT}" -eq "${DEPTH_COUNT}" ]; then
                echo "   ✅ Copied ${COPIED_COUNT} depth files"
                COPIED_DEPTHS=$((COPIED_DEPTHS + 1))
            else
                echo "   ❌ Copy failed: expected ${DEPTH_COUNT}, got ${COPIED_COUNT}"
                FAILED_SCENES=$((FAILED_SCENES + 1))
            fi
        else
            echo "📁 Depths: Empty directory"
        fi
    else
        echo "📁 Depths: Not found"
    fi
    
    # Check if scene had any data
    if [ "${SCENE_HAS_DATA}" -eq 0 ]; then
        echo "⚠️  No labels or depths found for this scene"
        SKIPPED_SCENES=$((SKIPPED_SCENES + 1))
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
echo "Scenes with labels copied: ${COPIED_LABELS}"
echo "Scenes with depths copied: ${COPIED_DEPTHS}"
echo "Skipped (no data): ${SKIPPED_SCENES}"
echo "Failed: ${FAILED_SCENES}"
echo ""

TOTAL_COPIED=$((COPIED_LABELS + COPIED_DEPTHS))

if [ ${FAILED_SCENES} -gt 0 ]; then
    echo "⚠️  Some scenes failed to copy"
    exit 1
elif [ ${TOTAL_COPIED} -eq 0 ]; then
    echo "⚠️  No data was copied"
    exit 1
else
    echo "✅ Copy operation completed successfully!"
    echo "   - ${COPIED_LABELS} scenes with labels"
    echo "   - ${COPIED_DEPTHS} scenes with depths"
    exit 0
fi

