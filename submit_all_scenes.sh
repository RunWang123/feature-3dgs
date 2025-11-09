#!/bin/bash
#
# Submit sbatch jobs for all scenes in JSON split file
# Usage: bash submit_all_scenes.sh <json_split_path> <data_base_dir> <output_base_dir>
#

set -e

# ============================================================================
# Configuration
# ============================================================================

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <json_split_path> <data_base_dir> <output_base_dir> [--skip-existing]"
    echo ""
    echo "Example:"
    echo "  $0 /scratch/split.json /scratch/data /scratch/output"
    echo "  $0 /scratch/split.json /scratch/data /scratch/output --skip-existing"
    exit 1
fi

JSON_SPLIT_PATH="$1"
DATA_BASE_DIR="$2"
OUTPUT_BASE_DIR="$3"
SKIP_EXISTING="false"

if [ "$#" -eq 4 ] && [ "$4" == "--skip-existing" ]; then
    SKIP_EXISTING="true"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_SCRIPT="${SCRIPT_DIR}/process_single_scene.sh"
SLURM_TEMPLATE="${SCRIPT_DIR}/scene_job.slurm"

# ============================================================================
# Validate inputs
# ============================================================================

if [ ! -f "${JSON_SPLIT_PATH}" ]; then
    echo "❌ Error: JSON split file not found: ${JSON_SPLIT_PATH}"
    exit 1
fi

if [ ! -f "${PROCESS_SCRIPT}" ]; then
    echo "❌ Error: Processing script not found: ${PROCESS_SCRIPT}"
    echo "   Expected: ${PROCESS_SCRIPT}"
    exit 1
fi

if [ ! -f "${SLURM_TEMPLATE}" ]; then
    echo "⚠️  Warning: SLURM template not found: ${SLURM_TEMPLATE}"
    echo "   Will use default sbatch settings"
fi

# ============================================================================
# Extract scene names from JSON
# ============================================================================

echo "========================================"
echo "Batch Scene Processing Submission"
echo "========================================"
echo "JSON split: ${JSON_SPLIT_PATH}"
echo "Data base: ${DATA_BASE_DIR}"
echo "Output base: ${OUTPUT_BASE_DIR}"
echo "Skip existing: ${SKIP_EXISTING}"
echo ""

# Get all scene names from JSON
SCENE_NAMES=$(python3 -c "
import json
with open('${JSON_SPLIT_PATH}', 'r') as f:
    data = json.load(f)
for scene in data['scenes'].keys():
    print(scene)
")

if [ -z "${SCENE_NAMES}" ]; then
    echo "❌ Error: No scenes found in JSON"
    exit 1
fi

# Count scenes
NUM_SCENES=$(echo "${SCENE_NAMES}" | wc -l)
echo "Found ${NUM_SCENES} scenes to process:"
echo "${SCENE_NAMES}"
echo ""

# ============================================================================
# Create output directory
# ============================================================================

mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}/logs"

# ============================================================================
# Submit jobs for each scene
# ============================================================================

echo "========================================"
echo "Submitting Jobs"
echo "========================================"
echo ""

SUBMITTED_JOBS=0
SKIPPED_JOBS=0
JOB_IDS=()

for SCENE_NAME in ${SCENE_NAMES}; do
    echo "Processing: ${SCENE_NAME}"
    
    # Check if we should skip this scene
    if [ "${SKIP_EXISTING}" == "true" ]; then
        # Check for any completed cases (case0, case1, etc.)
        # A case is complete if it has results.json (metrics computed)
        SHOULD_SKIP=true
        
        # Get number of cases for this scene from JSON
        NUM_CASES=$(python3 -c "
import json
with open('${JSON_SPLIT_PATH}', 'r') as f:
    data = json.load(f)
if '${SCENE_NAME}' in data['scenes']:
    print(len(data['scenes']['${SCENE_NAME}']))
else:
    print(0)
" 2>/dev/null)
        
        if [ -z "$NUM_CASES" ] || [ "$NUM_CASES" -eq 0 ]; then
            NUM_CASES=1  # Default to 1 case
        fi
        
        # Check if ALL cases are completed
        for CASE_ID in $(seq 0 $((NUM_CASES - 1))); do
            CASE_OUTPUT="${OUTPUT_BASE_DIR}/${SCENE_NAME}_case${CASE_ID}"
            
            # Check multiple completion indicators
            if [ ! -d "${CASE_OUTPUT}" ]; then
                SHOULD_SKIP=false
                break
            fi
            
            # Check for results.json (metrics completed) OR final checkpoint
            if [ ! -f "${CASE_OUTPUT}/results.json" ]; then
                # No results.json, check for any final checkpoint
                FOUND_CHECKPOINT=false
                for PLY in "${CASE_OUTPUT}"/point_cloud/iteration_*/point_cloud.ply; do
                    if [ -f "$PLY" ]; then
                        FOUND_CHECKPOINT=true
                        break
                    fi
                done
                
                if [ "$FOUND_CHECKPOINT" = false ]; then
                    SHOULD_SKIP=false
                    break
                fi
            fi
        done
        
        if [ "$SHOULD_SKIP" = true ]; then
            echo "  ⏭️  Skipping (all ${NUM_CASES} case(s) already completed)"
            SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            echo ""
            continue
        fi
    fi
    
    echo "  Submitting job..."
    
    # Job name
    JOB_NAME="feature3dgs_${SCENE_NAME}"
    
    # Log files
    LOG_DIR="${OUTPUT_BASE_DIR}/logs"
    STDOUT_LOG="${LOG_DIR}/${SCENE_NAME}_%j.out"
    STDERR_LOG="${LOG_DIR}/${SCENE_NAME}_%j.err"
    
    # Submit job
    if [ -f "${SLURM_TEMPLATE}" ]; then
        # Use template with placeholders
        JOB_ID=$(sbatch \
            --job-name="${JOB_NAME}" \
            --output="${STDOUT_LOG}" \
            --error="${STDERR_LOG}" \
            --export=ALL,SCENE_NAME="${SCENE_NAME}",JSON_SPLIT_PATH="${JSON_SPLIT_PATH}",DATA_BASE_DIR="${DATA_BASE_DIR}",OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
            "${SLURM_TEMPLATE}" \
            | awk '{print $4}')
    else
        # Direct sbatch with inline script
        JOB_ID=$(sbatch \
            --job-name="${JOB_NAME}" \
            --output="${STDOUT_LOG}" \
            --error="${STDERR_LOG}" \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem=32G \
            --gres=gpu:1 \
            --time=24:00:00 \
            --wrap="bash ${PROCESS_SCRIPT} ${SCENE_NAME} ${JSON_SPLIT_PATH} ${DATA_BASE_DIR} ${OUTPUT_BASE_DIR}" \
            | awk '{print $4}')
    fi
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Job submitted: ${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
    else
        echo "  ❌ Failed to submit job for ${SCENE_NAME}"
    fi
    
    echo ""
done

# ============================================================================
# Summary
# ============================================================================

echo "========================================"
echo "Submission Complete"
echo "========================================"
echo "Total scenes: ${NUM_SCENES}"
echo "Jobs submitted: ${SUBMITTED_JOBS}"
if [ "${SKIP_EXISTING}" == "true" ]; then
    echo "Jobs skipped: ${SKIPPED_JOBS}"
fi
echo ""
echo "Job IDs:"
for JOB_ID in "${JOB_IDS[@]}"; do
    echo "  - ${JOB_ID}"
done
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel all jobs: scancel ${JOB_IDS[@]}"
echo ""
echo "Logs directory: ${OUTPUT_BASE_DIR}/logs"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo ""

# ============================================================================
# Create monitoring script
# ============================================================================

MONITOR_SCRIPT="${OUTPUT_BASE_DIR}/monitor_jobs.sh"
cat > "${MONITOR_SCRIPT}" << 'EOF'
#!/bin/bash
# Monitor submitted jobs

echo "========================================"
echo "Job Status Monitor"
echo "========================================"
echo ""

# Your job IDs
JOB_IDS=(${JOB_IDS_PLACEHOLDER})

echo "Checking status of ${#JOB_IDS[@]} jobs..."
echo ""

squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R" --jobs=$(IFS=,; echo "${JOB_IDS[*]}")

echo ""
echo "Detailed status:"
echo ""

for JOB_ID in "${JOB_IDS[@]}"; do
    STATUS=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null || echo "NOT_FOUND")
    if [ "${STATUS}" == "NOT_FOUND" ]; then
        STATUS="COMPLETED or FAILED (check logs)"
    fi
    echo "Job ${JOB_ID}: ${STATUS}"
done

echo ""
echo "========================================"
EOF

# Replace placeholder with actual job IDs
sed -i "s/JOB_IDS_PLACEHOLDER/${JOB_IDS[*]}/" "${MONITOR_SCRIPT}"
chmod +x "${MONITOR_SCRIPT}"

echo "Created monitoring script: ${MONITOR_SCRIPT}"
echo "Run it with: bash ${MONITOR_SCRIPT}"
echo ""

