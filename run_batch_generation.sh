#!/bin/bash
# Batch video generation script for Wan2.2
# This is a convenience wrapper for batch_generate_videos.py

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
CONDA_ENV="wan2"
PYTHON="/home/hpc/miniconda3/envs/$CONDA_ENV/bin/python"
TORCHRUN="/home/hpc/miniconda3/envs/$CONDA_ENV/bin/torchrun"
DATASET_PATH="/home/hpc/captioning/data"
# On RTX3090 machine, override to GDrive mount path:
# DATASET_PATH="/mnt/g/My Drive/Archive - PhD/sa_video_story_engine"
CKPT_DIR="Wan2.2-TI2V-5B"
TEXT_KEY="gpt-4o_withGEST_t-1.0"

# Cross-machine sharding: split work with hpc (run_8gpu.sh takes shards 0+1 of 3).
# RTX3090 should use SHARD_IDX=2 NUM_SHARDS=3.
# Default (0/1): process full dataset on this machine alone.
SHARD_IDX=0
NUM_SHARDS=1

# Multi-GPU: set NUM_GPUS=2 (or more) to use torchrun with FSDP
# Set NUM_GPUS=1 for single-GPU mode (default, backwards compatible)
NUM_GPUS=4

# P100 (no BF16 hardware): "--fp16"; RTX3090 (native BF16): ""
FP16_FLAG="--fp16"

# Offload DiT to CPU before VAE decode (required for P100 16GB; skip on RTX3090 24GB): "--offload_model" / ""
OFFLOAD_FLAG="--offload_model"

# Generation parameters (optimized for RTX 3090 24GB; use --fp16 for P100)
TASK="ti2v-5B"
SIZE="832*480"
FRAME_NUM=121
SAMPLE_STEPS=30
SAMPLE_SOLVER="unipc"
SEED=42

# System parameters
LOG_FILE="batch_generation_$(date +%Y%m%d_%H%M%S).log"

# Check if virtual environment is activated
# if [[ -z "$VIRTUAL_ENV" ]]; then
#     echo "Virtual environment not activated. Activating..."
#     source .venv/bin/activate
# fi

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
echo "GPUs: $NUM_GPUS"
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
COMMON_ARGS=(
    --dataset_path "$DATASET_PATH"
    --ckpt_dir "$CKPT_DIR"
    --text_key "$TEXT_KEY"
    --task "$TASK"
    --size "$SIZE"
    --frame_num "$FRAME_NUM"
    --sample_steps "$SAMPLE_STEPS"
    --sample_solver "$SAMPLE_SOLVER"
    --seed "$SEED"
    --log_file "$LOG_FILE"
    --shard_idx "$SHARD_IDX"
    --num_shards "$NUM_SHARDS"
)
# Append optional flags only when non-empty
[ -n "$MAX_VIDEOS" ]   && COMMON_ARGS+=($MAX_VIDEOS)
[ -n "$OVERWRITE" ]    && COMMON_ARGS+=($OVERWRITE)
[ -n "$FP16_FLAG" ]    && COMMON_ARGS+=($FP16_FLAG)
[ -n "$OFFLOAD_FLAG" ] && COMMON_ARGS+=($OFFLOAD_FLAG)

# Reduce CUDA memory fragmentation (recovers ~300 MB of reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix NVML driver/library version mismatch (kernel 580.95.05 vs userspace 580.126.09).
# LD_PRELOAD is process-local: only this script and its children are affected.
# Remove this block once the server is rebooted (the new kernel module will match).
NVML_COMPAT=/tmp/nvidia9505/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
if [ -f "$NVML_COMPAT" ]; then
    export LD_PRELOAD="$NVML_COMPAT"
    echo "NVML compat: preloading $NVML_COMPAT"
else
    echo "WARNING: NVML compat library not found at $NVML_COMPAT — NCCL may fail."
    echo "Run: cd /tmp && apt-get download libnvidia-compute-580=580.95.05-0ubuntu1 && dpkg-deb -x libnvidia-compute-580_580.95.05-0ubuntu1_amd64.deb /tmp/nvidia9505/"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU mode: launching with torchrun --nproc_per_node=$NUM_GPUS"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" batch_generate_videos.py "${COMMON_ARGS[@]}"
else
    echo "Single-GPU mode: launching with python"
    "$PYTHON" batch_generate_videos.py "${COMMON_ARGS[@]}"
fi

echo ""
echo "========================================"
echo "Batch generation complete!"
echo "Check log file: $LOG_FILE"
echo "========================================"
