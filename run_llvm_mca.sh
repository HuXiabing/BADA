#!/bin/bash

# Unified LLVM-MCA Processing Script
# Description: Split input file, process in parallel, and merge results
# Usage: ./run_llvm_mca.sh --exp_name NAME --simulator NAME

# Default values
DEFAULT_SIMULATOR="sifive-u74"
DEFAULT_NUM_PARTS=10
TEMP_DIR="./temp_mca_processing"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --exp_name NAME             Experiment name (required)
    --simulator NAME             LLVM-MCA simulator type (default: $DEFAULT_SIMULATOR)
    --num_parts NUM              Number of parts to split into (default: $DEFAULT_NUM_PARTS)
    --temp_dir PATH              Temporary directory (default: $TEMP_DIR)
    --help                      Display this help message

Workflow:
    1. Split input file into multiple parts
    2. Process each part in parallel
    3. Merge all results into single output file
    4. Clean up temporary files

Paths:
    Input:  ../experiments/{exp_name}/input_generated.json
    Output: ../experiments/{exp_name}/output_generated.json

Examples:
    $0 --exp_name test_exp --simulator sifive-u74
    $0 --exp_name test_exp --num_parts 8
EOF
    exit 0
}

# Parse command line arguments
EXP_NAME=""
SIMULATOR="$DEFAULT_SIMULATOR"
NUM_PARTS="$DEFAULT_NUM_PARTS"
TEMP_DIR="$TEMP_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --simulator)
            SIMULATOR="$2"
            shift 2
            ;;
        --num_parts)
            NUM_PARTS="$2"
            shift 2
            ;;
        --temp_dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$EXP_NAME" ]]; then
    echo "Error: --exp_name is required"
    usage
fi

# Build input and output paths
INPUT="../experiments/${EXP_NAME}/input_generated.json"
OUTPUT="../experiments/${EXP_NAME}/output_generated.json"

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed. Please install jq first."
    exit 1
fi

# Check if llvm-mca is executable
if ! command -v llvm-mca &> /dev/null; then
    echo "Error: 'llvm-mca' does not exist or is not executable"
    exit 1
fi

# Get absolute path of temporary directory
TEMP_DIR_ABS=$(realpath "$TEMP_DIR" 2>/dev/null || echo "$TEMP_DIR")

echo "Processing data..."
# Step 1: Create temporary directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Step 2: Split input file into multiple parts
# Get total number of entries
total_entries=$(jq '. | length' "$INPUT")
# Calculate entries per part
entries_per_part=$((total_entries / NUM_PARTS))
remainder=$((total_entries % NUM_PARTS))

# Split the input file
start=0
for ((i=0; i<NUM_PARTS; i++)); do
    end=$((start + entries_per_part))
    
    # Add remainder to last part
    if [[ $i -eq $((NUM_PARTS - 1)) ]]; then
        end=$((end + remainder))
    fi
    
    # Extract entries for this part
    part_file="${TEMP_DIR}/part_${i}.json"
    jq --argjson start "$start" --argjson end "$end" \
       '.[$start:$end]' "$INPUT" > "$part_file"
    
    part_entries=$(jq '. | length' "$part_file")    
    start=$end
done

# Step 3: Process each part in parallel
pids=()
output_files=()

for ((i=0; i<NUM_PARTS; i++)); do
    part_file="${TEMP_DIR}/part_${i}.json"
    output_file="${TEMP_DIR}/output_${i}.json"
    output_files+=("$output_file")
    
    if [ -f "$part_file" ]; then        
        # Process this part in background
        (
            TEMP_ASM=$(mktemp -p "$TEMP_DIR" -t "asm_${i}_XXXXXX.S")
            
            # Initialize output JSON
            echo "[" > "$output_file"
            first_entry=true
            
            # Get number of entries in this part
            part_entries=$(jq '. | length' "$part_file")
            
            # Process each entry
            for ((j=0; j<part_entries; j++)); do
                # Extract asm content
                asm_content=$(jq -r ".[$j].asm" "$part_file")
                
                # Write asm to temporary file
                echo -e "$asm_content" > "$TEMP_ASM"
                
                # Run llvm-mca with specified simulator
                mca_output=$(llvm-mca -mcpu="$SIMULATOR" -iterations=1000 --instruction-info=0 -resource-pressure=0 "$TEMP_ASM" 2>&1)
                
                # Check if command was successful
                if [ $? -ne 0 ]; then
                    mca_output="Error running llvm-mca: $mca_output"
                fi
                
                # Add comma separator if not first entry
                if [ "$first_entry" = false ]; then
                    echo "," >> "$output_file"
                fi
                first_entry=false
                
                # Write result to output JSON
                jq -n --arg asm "$asm_content" --arg mca "$mca_output" \
                   '{"asm": $asm, "mca_result": $mca}' >> "$output_file"
            done
            
            # Close the JSON array
            echo "]" >> "$output_file"
            
            rm -f "$TEMP_ASM"
        ) &
        
        pids+=($!)
    else
        echo "Warning: Part file $part_file does not exist, skipping"
    fi
done

# Wait for all processes to complete
all_success=true
for ((i=0; i<${#pids[@]}; i++)); do
    pid=${pids[$i]}
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        echo "Process $pid (part $i) failed with status $status"
        all_success=false
    fi
done

if [ "$all_success" = false ]; then
    echo "Warning: Some processes failed. Check the output above."
fi

# Step 4: Merge all output files
# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Initialize merged output
echo "[" > "$OUTPUT"
first_entry=true

# Merge all parts
for ((i=0; i<NUM_PARTS; i++)); do
    output_file="${TEMP_DIR}/output_${i}.json"
    
    if [ -f "$output_file" ]; then
        # Read the output file and remove the outer brackets
        content=$(cat "$output_file")
        # Remove leading '[' and trailing ']'
        content=${content#[}
        content=${content%]}
        
        # Skip empty parts
        if [[ -z "$content" || "$content" == "[]" ]]; then
            continue
        fi
        # Add comma separator if not first entry
        if [ "$first_entry" = false ]; then
            echo "," >> "$OUTPUT"
        fi
        first_entry=false
        # Write the content (without brackets)
        echo "$content" >> "$OUTPUT"
    fi
done

# Close the merged JSON array
echo "]" >> "$OUTPUT"
# Step 5: Clean up temporary files

rm -rf "$TEMP_DIR"

echo "Total entries processed: $total_entries"