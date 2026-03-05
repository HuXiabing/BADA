set -e  # Exit on error

# ==================== Configuration Section ====================
LOCAL_WORK_DIR="$(pwd)"             # Local working directory
# Default parameters for Python commands
DEFAULT_MODEL_TYPE="transformer"
DEFAULT_TRAIN_DATA="datasets/u74.json"
DEFAULT_VAL_DATA="datasets/u74.json"
DEFAULT_TEST_DATA="datasets/u74_10.json"
DEFAULT_EXPERIMENT_NAME="test"
DEFAULT_EPOCH=3
DEFAULT_BATCH_SIZE=8
DEFAULT_INCREMENTAL_EXPERIMENT_NAME="incre"
DEFAULT_NO_IMPROVEMENT_LIMIT=2
DEFAULT_SIMULATOR="sifive-u74"

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handling function
error_exit() {
    log_error "$1"
    exit 1
}

# Check if file exists
check_file() {
    if [[ ! -f "$1" ]]; then
        error_exit "File not found: $1"
    fi
}

# Check if directory exists
check_dir() {
    if [[ ! -d "$1" ]]; then
        error_exit "Directory not found: $1"
    fi
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --model_type TYPE              Model type (default: $DEFAULT_MODEL_TYPE)
    --train_data PATH              Path to training data JSON file (default: $DEFAULT_TRAIN_DATA)
    --val_data PATH                Path to validation data JSON file (default: $DEFAULT_VAL_DATA)
    --test_data PATH               Path to test data JSON file (default: $DEFAULT_TEST_DATA)
    --experiment_name NAME         Initial experiment name (default: $DEFAULT_EXPERIMENT_NAME)
    --epoch NUM                    Number of epochs (default: $DEFAULT_EPOCH)
    --batch_size NUM               Batch size (default: $DEFAULT_BATCH_SIZE)
    --incremental_exp_name NAME     Incremental experiment name (default: $DEFAULT_INCREMENTAL_EXPERIMENT_NAME)
    --no_improvement_limit NUM     Number of consecutive iterations without improvement before stopping (default: $DEFAULT_NO_IMPROVEMENT_LIMIT)
    --simulator NAME               LLVM-MCA simulator type (default: $DEFAULT_SIMULATOR)
    --help                         Display this help message

Example:
    $0 --model_type transformer --train_data datasets/u74.json --epoch 5 --batch_size 16 --simulator sifive-u74
EOF
    exit 0
}

# Parse command line arguments
MODEL_TYPE="$DEFAULT_MODEL_TYPE"
TRAIN_DATA="$DEFAULT_TRAIN_DATA"
VAL_DATA="$DEFAULT_VAL_DATA"
TEST_DATA="$DEFAULT_TEST_DATA"
EXPERIMENT_NAME="$DEFAULT_EXPERIMENT_NAME"
EPOCH="$DEFAULT_EPOCH"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
INCREMENTAL_EXPERIMENT_NAME="$DEFAULT_INCREMENTAL_EXPERIMENT_NAME"
NO_IMPROVEMENT_LIMIT="$DEFAULT_NO_IMPROVEMENT_LIMIT"
SIMULATOR="$DEFAULT_SIMULATOR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --val_data)
            VAL_DATA="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --incremental_exp_name)
            INCREMENTAL_EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --no_improvement_limit)
            NO_IMPROVEMENT_LIMIT="$2"
            shift 2
            ;;
        --simulator)
            SIMULATOR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

log_info "=== Starting Automated Training Process with LLVM-MCA Simulator ==="
log_info "Configuration:"
log_info "  Model Type: $MODEL_TYPE"
log_info "  Training Data: $TRAIN_DATA"
log_info "  Validation Data: $VAL_DATA"
log_info "  Test Data: $TEST_DATA"
log_info "  Initial Experiment Name: $EXPERIMENT_NAME"
log_info "  Epochs: $EPOCH"
log_info "  Batch Size: $BATCH_SIZE"
log_info "  Incremental Experiment Name: $INCREMENTAL_EXPERIMENT_NAME"
log_info "  No Improvement Limit: $NO_IMPROVEMENT_LIMIT"
log_info "  Simulator: $SIMULATOR"

# Step 1: Run initial training command
log_step "1. Running initial training command"
TRAIN_CMD="python main.py train --model_type $MODEL_TYPE --train_data $TRAIN_DATA --val_data $VAL_DATA --experiment_name $EXPERIMENT_NAME --epoch $EPOCH --batch_size $BATCH_SIZE"

log_info "Executing command: $TRAIN_CMD"

# Capture output and display
TRAIN_OUTPUT=$(mktemp)
if ! $TRAIN_CMD 2>&1 | tee "$TRAIN_OUTPUT"; then
    error_exit "Initial training command execution failed"
fi

# Step 2: Extract experiment name from output
log_step "2. Extracting experiment name"
EXPERIMENT_NAME=$(grep -oP "Experiment created: \K[^,]*" "$TRAIN_OUTPUT" | tail -1)

if [[ -z "$EXPERIMENT_NAME" ]]; then
    error_exit "Failed to extract experiment name from output"
fi

log_info "Experiment name: $EXPERIMENT_NAME"

# Step 3: Analyze experiment logs
log_step "3. Analyzing experiment logs"
LOG_FILE="../experiments/${EXPERIMENT_NAME}/logs/experiment.log"

# Wait for log file to be generated (max 60 seconds)
WAIT_COUNT=0
while [[ ! -f "$LOG_FILE" && $WAIT_COUNT -lt 60 ]]; do
    log_warn "Waiting for log file to be generated... ($WAIT_COUNT/60)"
    sleep 1
    ((WAIT_COUNT++))
done

check_file "$LOG_FILE"

# Get training data file information from the third line
log_info "Extracting information from log file: $LOG_FILE"
TRAIN_DATA_LINE=$(sed -n '3p' "$LOG_FILE")
TRAIN_DATA_FILE=$(echo "$TRAIN_DATA_LINE" | grep -oP "Training data: \K[^,]*")

if [[ -z "$TRAIN_DATA_FILE" ]]; then
    error_exit "Failed to extract training data file information from log line 3"
fi

log_info "Training data file: $TRAIN_DATA_FILE"

# Get Epoch information from the last line
LAST_LINE=$(tail -n 1 "$LOG_FILE")
EPOCH=$(echo "$LAST_LINE" | grep -oP "at Epoch \K\d+")

if [[ -z "$EPOCH" ]]; then
    error_exit "Failed to extract Epoch information from log last line"
fi

log_info "Best Epoch: $EPOCH"

# Get initial validation loss
INITIAL_LOSS=$(echo "$LAST_LINE" | grep -oP "Best validation loss: \K[0-9.]+")
if [[ -z "$INITIAL_LOSS" ]]; then
    error_exit "Failed to extract initial validation loss from log last line"
fi

log_info "Initial validation loss: $INITIAL_LOSS"

# Step 4: Run inference on initial training
log_step "4. Running inference on initial training (this may take a while...)"
if ! python scripts/inference.py --file "$EXPERIMENT_NAME" --test_data "$TEST_DATA" > /dev/null; then
    error_exit "Inference failed for experiment: $EXPERIMENT_NAME"
fi
INITIAL_TEST_LOSS=$(python3 -c "import json; print(f\"{json.load(open('../experiments/${EXPERIMENT_NAME}/test_result.json'))['loss']:.6f}\")")
log_info "Inference completed. Test loss: $INITIAL_TEST_LOSS"

# Initialize loop variables
CURRENT_EXPERIMENT_NAME="$EXPERIMENT_NAME"
CURRENT_TRAIN_DATA_FILE="$TRAIN_DATA_FILE"
CURRENT_EPOCH="$EPOCH"
BEST_LOSS="$INITIAL_LOSS"
PREV_LOSS="$INITIAL_LOSS"
NO_IMPROVEMENT_COUNT=0
ITERATION=1

# Arrays to record iteration results
ITERATION_EXPERIMENTS=()
ITERATION_LOSSES=()
ITERATION_EPOCHS=()
ITERATION_TEST_LOSSES=()

# Record initial results
ITERATION_EXPERIMENTS[0]="$EXPERIMENT_NAME"
ITERATION_LOSSES[0]="$INITIAL_LOSS"
ITERATION_EPOCHS[0]="$EPOCH"
ITERATION_TEST_LOSSES[0]="$INITIAL_TEST_LOSS"

log_info "=== Starting Iterative Training Loop ==="

# Iteration loop: steps 5-9
while [[ $NO_IMPROVEMENT_COUNT -lt $NO_IMPROVEMENT_LIMIT ]]; do
    log_info "=== Iteration $ITERATION ==="

    # Step 5: Run fuzzing
    log_step "5.$ITERATION Running bada/tool.py"
    GEN_CMD="python bada/tool.py --exp $CURRENT_EXPERIMENT_NAME --epoch $CURRENT_EPOCH --train $CURRENT_TRAIN_DATA_FILE"

    log_info "Executing command: $GEN_CMD"
    if ! $GEN_CMD; then
        error_exit "bada/tool.py execution failed (iteration ${ITERATION})"
    fi
    
    # Step 6: Run LLVM-MCA simulator
    log_step "6.$ITERATION Running LLVM-MCA simulator"
    check_file "./run_llvm_mca.sh"

    log_info "Running simulator: $SIMULATOR"
    if ! ./run_llvm_mca.sh --exp_name "$CURRENT_EXPERIMENT_NAME" --simulator "$SIMULATOR"; then
        error_exit "LLVM-MCA simulator execution failed (iteration ${ITERATION})"
    fi

    log_info "Simulator execution completed"

    # Output file from simulator
    OUTPUT_FILE="../experiments/${CURRENT_EXPERIMENT_NAME}/output_generated.json"

    if [ ! -f "$OUTPUT_FILE" ]; then
        log_error "Failed to find $OUTPUT_FILE after simulator execution!"
        exit 1
    fi

    log_info "Result file generated successfully"

    # Step 7: Run preprocess.py
    log_step "7.$ITERATION Running preprocess.py"

    # Generate new training data file name
    TRAIN_BASE=$(basename "$CURRENT_TRAIN_DATA_FILE" .json)
    TRAIN_DIR=$(dirname "$CURRENT_TRAIN_DATA_FILE")
    NEW_TRAIN_FILE="${TRAIN_DIR}/${TRAIN_BASE}_2k.json"

    log_info "New training data file: $NEW_TRAIN_FILE"

    # Check file
    check_file "$OUTPUT_FILE"

    PREPROCESS_CMD="python scripts/preprocess.py --cycle_jsons $OUTPUT_FILE --existing_train_json $CURRENT_TRAIN_DATA_FILE --existing_val_json $VAL_DATA --train_json $NEW_TRAIN_FILE"

    log_info "Executing command: $PREPROCESS_CMD"
    if ! $PREPROCESS_CMD; then
        error_exit "preprocess.py execution failed (iteration ${ITERATION})"
    fi

    # Step 8: Run incremental training
    log_step "8.$ITERATION Running incremental training"
    MODEL_PATH="../experiments/${CURRENT_EXPERIMENT_NAME}/checkpoints/model_best.pth"
    check_file "$MODEL_PATH"

    INCREMENTAL_CMD="python main.py incremental --train_data $NEW_TRAIN_FILE --model_path $MODEL_PATH --experiment_name $INCREMENTAL_EXPERIMENT_NAME --epoch $EPOCH --batch_size $BATCH_SIZE --val_data $VAL_DATA"

    log_info "Executing command: $INCREMENTAL_CMD"

    # Capture incremental training output
    INCREMENTAL_OUTPUT=$(mktemp)
    if ! $INCREMENTAL_CMD 2>&1 | tee "$INCREMENTAL_OUTPUT"; then
        error_exit "Incremental training execution failed (iteration ${ITERATION})"
    fi

    # Extract new experiment name from incremental training output
    NEW_EXPERIMENT_NAME=$(grep -oP "Experiment created: \K[^,]*" "$INCREMENTAL_OUTPUT" | tail -1)
    if [[ -z "$NEW_EXPERIMENT_NAME" ]]; then
        error_exit "Failed to extract experiment name from incremental training output (iteration ${ITERATION})"
    fi

    log_info "New experiment name: $NEW_EXPERIMENT_NAME"

    # Wait for new experiment log file to be generated
    NEW_LOG_FILE="../experiments/${NEW_EXPERIMENT_NAME}/logs/experiment.log"
    WAIT_COUNT=0
    while [[ ! -f "$NEW_LOG_FILE" && $WAIT_COUNT -lt 60 ]]; do
        log_warn "Waiting for new experiment log file to be generated... ($WAIT_COUNT/60)"
        sleep 1
        ((WAIT_COUNT++))
    done

    check_file "$NEW_LOG_FILE"

    # Get information from new experiment log
    NEW_LAST_LINE=$(tail -n 1 "$NEW_LOG_FILE")
    NEW_LOSS=$(echo "$NEW_LAST_LINE" | grep -oP "Best validation loss: \K[0-9.]+")
    NEW_EPOCH=$(echo "$NEW_LAST_LINE" | grep -oP "at Epoch \K\d+")

    if [[ -z "$NEW_LOSS" ]]; then
        error_exit "Failed to extract validation loss from new experiment log (iteration ${ITERATION})"
    fi

    if [[ -z "$NEW_EPOCH" ]]; then
        error_exit "Failed to extract Epoch information from new experiment log (iteration ${ITERATION})"
    fi

    log_info "Iteration ${ITERATION} result - Loss: $NEW_LOSS, Epoch: $NEW_EPOCH"

    # Step 9: Run inference on incremental training
    log_step "9.${ITERATION} Running inference on incremental training (this may take a while...)"
    if ! python scripts/inference.py --file "$NEW_EXPERIMENT_NAME" --test_data "$TEST_DATA" > /dev/null; then
        error_exit "Inference failed for experiment: $NEW_EXPERIMENT_NAME"
    fi
    NEW_TEST_LOSS=$(python3 -c "import json; print(f\"{json.load(open('../experiments/${NEW_EXPERIMENT_NAME}/test_result.json'))['loss']:.6f}\")")
    log_info "Inference completed. Test loss: $NEW_TEST_LOSS"

    # Record this iteration result
    ITERATION_EXPERIMENTS[$ITERATION]="$NEW_EXPERIMENT_NAME"
    ITERATION_LOSSES[$ITERATION]="$NEW_LOSS"
    ITERATION_EPOCHS[$ITERATION]="$NEW_EPOCH"
    ITERATION_TEST_LOSSES[$ITERATION]="$NEW_TEST_LOSS"

    # Check if there is improvement
    IMPROVEMENT=$(python3 -c "print(1 if float('$NEW_LOSS') < float('$BEST_LOSS') else 0)")

    if [[ "$IMPROVEMENT" == "1" ]]; then
        log_info "Validation loss improved! ($BEST_LOSS -> $NEW_LOSS)"
        BEST_LOSS="$NEW_LOSS"
        NO_IMPROVEMENT_COUNT=0
    else
        NO_IMPROVEMENT_COUNT=$((NO_IMPROVEMENT_COUNT + 1))
        log_warn "Validation loss did not improve ($PREV_LOSS -> $NEW_LOSS), consecutive no improvement count: $NO_IMPROVEMENT_COUNT"
    fi

    # Update loop variables for next iteration
    CURRENT_EXPERIMENT_NAME="$NEW_EXPERIMENT_NAME"
    CURRENT_TRAIN_DATA_FILE="$NEW_TRAIN_FILE"
    CURRENT_EPOCH="$NEW_EPOCH"
    PREV_LOSS="$NEW_LOSS"
    ITERATION=$((ITERATION + 1))

    # Clean up temporary files
    rm -f "$INCREMENTAL_OUTPUT"

    # Check termination condition
    if [[ $NO_IMPROVEMENT_COUNT -ge $NO_IMPROVEMENT_LIMIT ]]; then
        log_info "Reached no improvement limit, exiting iteration loop"
        break
    fi

    log_info "Preparing for iteration $ITERATION..."
    echo "----------------------------------------"
done

log_info "=== Iterative Training Loop Ended ==="
log_info "Total iterations performed: $((ITERATION - 1))"
log_info "Best validation loss: $BEST_LOSS"
log_info "Final experiment name: $CURRENT_EXPERIMENT_NAME"

# Clean up temporary files
rm -f "$TRAIN_OUTPUT"

log_info "=== Automated Training Process Completed ==="
log_info "Initial experiment name: $EXPERIMENT_NAME"
log_info "Final experiment name: $CURRENT_EXPERIMENT_NAME"
log_info "Initial training data file: $TRAIN_DATA_FILE"
log_info "Final training data file: $CURRENT_TRAIN_DATA_FILE"
log_info "Initial validation loss: $INITIAL_LOSS"
log_info "Best validation loss: $BEST_LOSS"
log_info "Total iterations: $((ITERATION - 1))"

echo
log_info "=== Detailed Iteration History ==="
echo -e "${BLUE}┌─────────────┬────────────────────────────────────┬─────────────┬─────────┬─────────────┐${NC}"
echo -e "${BLUE}│   Round     │         Experiment Name            │ Validation  │  Epoch  │   Test      │${NC}"
echo -e "${BLUE}│             │                                    │    Loss     │         │    Loss     │${NC}"
echo -e "${BLUE}├─────────────┼────────────────────────────────────┼─────────────┼─────────┼─────────────┤${NC}"

# Output initial training result
printf "${BLUE}│${NC} %-11s ${BLUE}│${NC} %-34s ${BLUE}│${NC} %-11s ${BLUE}│${NC} %-7s ${BLUE}│${NC} %-11s ${BLUE}│${NC}\n" \
    "Initial" "${ITERATION_EXPERIMENTS[0]}" "${ITERATION_LOSSES[0]}" "${ITERATION_EPOCHS[0]}" "${ITERATION_TEST_LOSSES[0]}"

# Output each iteration result
for ((i=1; i<ITERATION; i++)); do
    ROUND_LABEL="Round $i"
    # Check if this is the best result
    if [[ "${ITERATION_LOSSES[i]}" == "$BEST_LOSS" ]]; then
        # Highlight best result in green
        printf "${GREEN}│ %-11s │ %-34s │ %-11s │ %-7s │ %-11s │${NC}\n" \
            "$ROUND_LABEL" "${ITERATION_EXPERIMENTS[i]}" "${ITERATION_LOSSES[i]}" "${ITERATION_EPOCHS[i]}" "${ITERATION_TEST_LOSSES[i]} ★"
    else
        printf "${BLUE}│${NC} %-11s ${BLUE}│${NC} %-34s ${BLUE}│${NC} %-11s ${BLUE}│${NC} %-7s ${BLUE}│${NC} %-11s ${BLUE}│${NC}\n" \
            "$ROUND_LABEL" "${ITERATION_EXPERIMENTS[i]}" "${ITERATION_LOSSES[i]}" "${ITERATION_EPOCHS[i]}" "${ITERATION_TEST_LOSSES[i]}"
    fi
done

echo -e "${BLUE}└─────────────┴────────────────────────────────────┴─────────────┴─────────┴─────────────┘${NC}"

echo
# Find the best result and display it specially
BEST_INDEX=0
for ((i=0; i<ITERATION; i++)); do
    if [[ "${ITERATION_LOSSES[i]}" == "$BEST_LOSS" ]]; then
        BEST_INDEX=$i
        break
    fi
done

if [[ $BEST_INDEX -eq 0 ]]; then
    log_info "Best result: Initial training (Loss: $BEST_LOSS)"
else
    log_info "Best result: Round ${BEST_INDEX} iteration (Loss: $BEST_LOSS)"
fi

echo
log_info "All steps completed successfully!"