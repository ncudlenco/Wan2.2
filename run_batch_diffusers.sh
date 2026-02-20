#!/bin/bash
# Batch video generation script for Wan2.2 (Diffusers + BitsAndBytes int8)
# This is a convenience wrapper for batch_generate_videos_diffusers.py
set -e  # Exit on error

# Configuration - EDIT THESE VALUES
DATASET_PATH="/mnt/g/My Drive/Archive - PhD/sa_video_story_engine"
MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
TEXT_KEY="gpt-4o_withGEST_t-1.0"

# Generation parameters (optimized for RTX 3090 24GB with int8 quantization)
HEIGHT=704
WIDTH=1280
FRAME_NUM=121
SAMPLE_STEPS=30
GUIDANCE_SCALE=4.0
SEED=42

# Quantization (default: int8)
QUANT_FLAG="--quantize_8bit"

# System parameters
LOG_FILE="batch_generation_diffusers_$(date +%Y%m%d_%H%M%S).log"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Activating..."
    source .venv/bin/activate
fi

# Print configuration
echo "========================================"
echo "Batch Video Generation - Wan2.2 Diffusers"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_ID"
echo "Text Key: $TEXT_KEY"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Frames: $FRAME_NUM (~$(echo "scale=1; $FRAME_NUM/24" | bc)s at 24fps)"
echo "Steps: $SAMPLE_STEPS"
echo "Quantization: int8 (BitsAndBytes)"
echo "Log File: $LOG_FILE"
echo "========================================"
echo ""

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Parse command line arguments
MAX_VIDEOS=""
OVERWRITE=""
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            MAX_VIDEOS="--max_videos 1"
            echo "SINGLE MODE: Processing only 1 video (for testing)"
            shift
            ;;
        --test)
            MAX_VIDEOS="--max_videos 5"
            echo "TEST MODE: Processing only 5 videos"
            shift
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            echo "OVERWRITE MODE: Will regenerate existing videos"
            shift
            ;;
        --4bit)
            QUANT_FLAG="--quantize_4bit"
            echo "4-BIT MODE: Using int4 quantization (more aggressive)"
            shift
            ;;
        --no-quant)
            QUANT_FLAG="--no_quantize"
            echo "NO QUANT MODE: Using bf16 (may need --offload)"
            shift
            ;;
        --offload)
            EXTRA_FLAGS="$EXTRA_FLAGS --use_cpu_offload"
            echo "OFFLOAD MODE: Using smart CPU offloading"
            shift
            ;;
        --ref-frame)
            EXTRA_FLAGS="$EXTRA_FLAGS --use_reference_frame"
            echo "REF FRAME MODE: Extracting reference frames from raw.mp4"
            shift
            ;;
        --max)
            MAX_VIDEOS="--max_videos $2"
            echo "Processing maximum $2 videos"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--single] [--test] [--overwrite] [--4bit] [--no-quant] [--offload] [--ref-frame] [--max N]"
            echo "  --single    : Process only 1 video (quick test)"
            echo "  --test      : Process only 5 videos"
            echo "  --overwrite : Regenerate existing videos"
            echo "  --4bit      : Use int4 quantization (more VRAM savings, slight quality loss)"
            echo "  --no-quant  : Disable quantization (bf16, may OOM on 24GB)"
            echo "  --offload   : Enable smart CPU offloading (fallback if OOM)"
            echo "  --ref-frame : Use reference frames from raw.mp4 (image-to-video mode)"
            echo "  --max N     : Process maximum N videos"
            exit 1
            ;;
    esac
done

echo "Starting batch generation..."
echo "Press Ctrl+C to cancel"
echo ""

# Run the batch generation
python batch_generate_videos_diffusers.py \
    --dataset_path "$DATASET_PATH" \
    --model_id "$MODEL_ID" \
    --text_key "$TEXT_KEY" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --frame_num "$FRAME_NUM" \
    --sample_steps "$SAMPLE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --log_file "$LOG_FILE" \
    $QUANT_FLAG \
    $MAX_VIDEOS \
    $OVERWRITE \
    $EXTRA_FLAGS

echo ""
echo "========================================"
echo "Batch generation complete!"
echo "Check log file: $LOG_FILE"
echo "========================================"