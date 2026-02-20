#!/bin/bash
# Batch video generation script for Wan2.2
# This is a convenience wrapper for batch_generate_videos.py

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
DATASET_PATH="/mnt/g/My Drive/Archive - PhD/sa_video_story_engine"
CKPT_DIR="Wan2.2-TI2V-5B"
TEXT_KEY="gpt-4o_withGEST_t-1.0"

# Generation parameters (optimized for RTX 3090 24GB)
TASK="ti2v-5B"
SIZE="832*480"
FRAME_NUM=121
SAMPLE_STEPS=30
SAMPLE_SOLVER="unipc"
SEED=42

# System parameters
LOG_FILE="batch_generation_$(date +%Y%m%d_%H%M%S).log"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Activating..."
    source .venv/bin/activate
fi

# Print configuration
echo "========================================"
echo "Batch Video Generation - Wan2.2"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Checkpoint: $CKPT_DIR"
echo "Text Key: $TEXT_KEY"
echo "Task: $TASK"
echo "Size: $SIZE"
echo "Frames: $FRAME_NUM (~$(echo "scale=1; $FRAME_NUM/24" | bc)s at 24fps)"
echo "Steps: $SAMPLE_STEPS"
echo "Log File: $LOG_FILE"
echo "========================================"
echo ""

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

# Parse command line arguments
MAX_VIDEOS=""
OVERWRITE=""

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
        --fast)
            SAMPLE_STEPS=30
            SIZE="832*480"
            echo "FAST MODE: Using 30 steps and 832*480 resolution"
            shift
            ;;
        --max)
            MAX_VIDEOS="--max_videos $2"
            echo "Processing maximum $2 videos"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--single] [--test] [--overwrite] [--fast] [--max N]"
            echo "  --single    : Process only 1 video (quick test)"
            echo "  --test      : Process only 5 videos (for testing)"
            echo "  --overwrite : Regenerate existing videos"
            echo "  --fast      : Use faster settings (30 steps, 832*480)"
            echo "  --max N     : Process maximum N videos"
            exit 1
            ;;
    esac
done

echo "Starting batch generation..."
echo "Press Ctrl+C to cancel"
echo ""

# Run the batch generation
# Note: t5_cpu, convert_model_dtype, offload_model, and low_vram_mode are enabled by default
python batch_generate_videos.py \
    --dataset_path "$DATASET_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --text_key "$TEXT_KEY" \
    --task "$TASK" \
    --size "$SIZE" \
    --frame_num "$FRAME_NUM" \
    --sample_steps "$SAMPLE_STEPS" \
    --sample_solver "$SAMPLE_SOLVER" \
    --seed "$SEED" \
    --log_file "$LOG_FILE" \
    $MAX_VIDEOS \
    $OVERWRITE

echo ""
echo "========================================"
echo "Batch generation complete!"
echo "Check log file: $LOG_FILE"
echo "========================================"
