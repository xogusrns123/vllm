#!/bin/bash

# Define parameters
MODEL="facebook/opt-1.3b"
MAX_NUM_BATCHED_TOKENS=65536
ENFORCE_EAGER="--enforce-eager"
OUTPUT_LEN=3
DTYPE="float16"
OUTPUT_DIR="/home/kth/brl/profile_data/opt1.3b/2080Ti"

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Batch sizes and prompt lengths to iterate over
BATCH_SIZES=(1 2 4 8 16 32 64)
PROMPT_LENGTHS=(128 256 512)

# Iterate over batch sizes and prompt lengths
for PROMPT_LEN in "${PROMPT_LENGTHS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        CSV_FILE="${OUTPUT_DIR}/batch${BATCH_SIZE}_prompt${PROMPT_LEN}.csv"
        JSON_FILE="${OUTPUT_DIR}/batch${BATCH_SIZE}_prompt${PROMPT_LEN}.json"
        
        echo "Running inference with batch size ${BATCH_SIZE} and prompt length ${PROMPT_LEN}..."
        
        python examples/offline_profile.py \
            --model $MODEL \
            --batch-size $BATCH_SIZE \
            --prompt-len $PROMPT_LEN \
            --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
            $ENFORCE_EAGER \
            --output-len $OUTPUT_LEN \
            --dtype $DTYPE \
            --csv $CSV_FILE \
            --json $JSON_FILE
        
        echo "Finished batch size ${BATCH_SIZE} and prompt length ${PROMPT_LEN}. Results saved to ${CSV_FILE} and ${JSON_FILE}."
        
        # Wait for 5 seconds before the next iteration
        echo "Waiting for 5 seconds..."
        sleep 5
    done
done

echo "All combinations of batch sizes and prompt lengths completed."
