# Enable debug mode
DEBUG=0

# Check number of arguments
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 <input JSON file> <output JSON file>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_JSON="$2"
TIMEOUT_SECONDS=10

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist"
    exit 1
fi

# Check if test program exists and is executable
if [ ! -x "./test" ]; then
    echo "Error: './test' does not exist or is not executable"
    exit 1
fi

# Debug function
debug_print() {
    if [ $DEBUG -eq 1 ]; then
        echo -e "[DEBUG] $1" >&2
    fi
}

# Counters
processed_tests=0
success_count=0
timeout_count=0
failed_count=0

# Create temporary directory for individual test outputs
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Timeout function
run_with_timeout() {
    local cmd="$1"
    local timeout=$2
    local output_file="$3"

    SECONDS=0
    eval "$cmd" > "$output_file" 2>&1 &
    local pid=$!

    local step=0.1
    local elapsed=0
    while kill -0 $pid 2>/dev/null; do
        sleep $step
        elapsed=$(echo "$SECONDS" | awk '{printf "%.1f", $1}')
        if (( $(echo "$elapsed >= $timeout" | bc -l) )); then
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            echo "  Error: Execution timeout (>${timeout}s), terminated" | tee -a "$output_file"
            return 124
        fi
    done

    wait $pid
    return $?
}

# Function to process a single JSON object
process_single_object() {
    local obj="$1"
    local test_index="$2"
    
    # Extract binary field
    local binary=""
    if [[ "$obj" =~ \"binary\"[[:space:]]*:[[:space:]]*\"([^\"]+)\" ]]; then
        binary="${BASH_REMATCH[1]}"
    else
        debug_print "Binary field not found, skipping this object"
        return 1
    fi

    # Extract asm field
    local asm=""
    if [[ "$obj" =~ \"asm\"[[:space:]]*:[[:space:]]*\"([^\"]+)\" ]]; then
        asm="${BASH_REMATCH[1]}"
        # Process escape characters
        asm=$(echo -e "$asm")
    else
        debug_print "Asm field not found, using empty string"
    fi

    # Generate test name
    local current_test="test_$test_index"

    # Prepare output file
    local temp_output="$TEMP_DIR/${current_test}.txt"

    # Execute test
    run_with_timeout "./test $binary" $TIMEOUT_SECONDS "$temp_output"
    local exit_code=$?

    # Check command execution status
    if [ $exit_code -eq 0 ]; then
        ((success_count++))
    elif [ $exit_code -eq 124 ]; then
        debug_print "Test execution timeout"
        echo "  Error: Execution timeout" >> "$temp_output"
        ((timeout_count++))
    else
        debug_print "Test execution failed, exit code: $exit_code"
        echo "  Warning: Command execution failed (exit code: $exit_code)" >> "$temp_output"
        ((failed_count++))
    fi

    # Read output file content and escape JSON special characters
    local output_content=$(<"$temp_output")
    output_content=${output_content//\\/\\\\}
    output_content=${output_content//\"/\\\"}
    output_content=${output_content//$'\n'/\\n}
    output_content=${output_content//$'\t'/\\t}

    # Write JSON entry
    if [ "$first_entry" = false ]; then
        echo "," >> "$OUTPUT_JSON"
    fi
    first_entry=false

    # Escape special characters in asm content for JSON output
    local asm_json=${asm//\\/\\\\}
    asm_json=${asm_json//\"/\\\"}
    asm_json=${asm_json//$'\n'/\\n}
    asm_json=${asm_json//$'\t'/\\t}

    # Include original input and test output
    echo -n "  {" >> "$OUTPUT_JSON"
    echo -n "\"asm\": \"$asm_json\", " >> "$OUTPUT_JSON"
    echo -n "\"binary\": \"$binary\", " >> "$OUTPUT_JSON"
    echo -n "\"result\": \"$output_content\"" >> "$OUTPUT_JSON"
    echo -n "}" >> "$OUTPUT_JSON"

    ((processed_tests++))

    # Clean up temporary file
    rm -f "$temp_output"

    # Free memory and CPU
    if (( processed_tests % 500 == 0 )); then
        echo "Processed $processed_tests tests, taking a short break..." >&2
        sleep 50
        ps aux | grep "./test" | grep -v grep | awk '{print $1}' | xargs kill -9  2>/dev/null
        sleep 10
    fi

    return 0
}

# Function for streaming JSON file processing
stream_process_json() {
    local input_file="$1"
    local buffer=""
    local in_object=false
    local bracket_count=0
    local in_string=false
    local escape_next=false
    local test_index=0
    local line_num=0
    
    # Use while read loop to process line by line, avoiding loading entire file into memory
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))
        
        # Show progress every 1000 lines
        if (( line_num % 1000 == 0 )); then
            echo "Read $line_num lines..." >&2
        fi
        
        # Process current line character by character
        local i=0
        while [ $i -lt ${#line} ]; do
            local char="${line:$i:1}"
            
            # Handle escape characters
            if [ "$escape_next" = true ]; then
                buffer+="$char"
                escape_next=false
                ((i++))
                continue
            fi
            
            if [ "$char" = "\\" ]; then
                buffer+="$char"
                escape_next=true
                ((i++))
                continue
            fi
            
            # Handle string state
            if [ "$char" = "\"" ]; then
                if [ "$in_string" = true ]; then
                    in_string=false
                else
                    in_string=true
                fi
                buffer+="$char"
                ((i++))
                continue
            fi
            
            # If inside string, directly add character
            if [ "$in_string" = true ]; then
                buffer+="$char"
                ((i++))
                continue
            fi
            
            # Handle JSON structure characters
            case "$char" in
                "{")
                    if [ "$in_object" = false ]; then
                        in_object=true
                        buffer=""
                    fi
                    buffer+="$char"
                    ((bracket_count++))
                    ;;
                "}")
                    buffer+="$char"
                    ((bracket_count--))
                    
                    # If brackets are balanced and in object, found complete JSON object
                    if [ $bracket_count -eq 0 ] && [ "$in_object" = true ]; then
                        # Process this complete JSON object
                        process_single_object "$buffer" "$test_index"
                        ((test_index++))
                        
                        # Reset state
                        buffer=""
                        in_object=false
                        
                        # Check if long break is needed
                        if (( SECONDS >= 3600 )); then
                            echo "Runtime reached 1 hour, starting rest and cleanup..." >&2
                            sleep 60
                            ps aux | grep "./test" | grep -v grep | awk '{print $1}' | xargs kill -9  2>/dev/null
                            sleep 60
                            SECONDS=0
                        fi
                    fi
                    ;;
                " "|$'\t'|$'\n'|$'\r'|","|"["|"]")
                    # Ignore JSON array start/end symbols and whitespace characters
                    if [ "$in_object" = true ]; then
                        buffer+="$char"
                    fi
                    ;;
                *)
                    if [ "$in_object" = true ]; then
                        buffer+="$char"
                    fi
                    ;;
            esac
            
            ((i++))
        done
        
        # Add newline at end of line (if inside object)
        if [ "$in_object" = true ]; then
            buffer+=$'\n'
        fi
        
    done < "$input_file"
    
    echo "File processing complete, read $line_num lines total" >&2
}

# Initialize output JSON file
echo "[" > "$OUTPUT_JSON"
first_entry=true
SECONDS=0

echo "Starting streaming parsing and processing of test cases..."

# Stream process JSON file
stream_process_json "$INPUT_FILE"

# Complete JSON file
echo -e "\n]" >> "$OUTPUT_JSON"

echo "Processing complete: Total of $processed_tests tests processed"
echo "  Success: $success_count"
echo "  Timeout: $timeout_count"
echo "  Failed: $failed_count"
echo "Results saved to $OUTPUT_JSON"

exit 0