#!/bin/bash
# 8-GPU launcher: 2 parallel instances × 4 GPUs each (FSDP per instance)
# Each instance processes a disjoint shard of the dataset.
# 2 GPUs per instance OOMs at 832×480 — 4 GPUs needed for activation memory headroom.
#
# Usage:
#   ./run_8gpu.sh               # full run
#   ./run_8gpu.sh --test        # test with ~1 video per shard
#   ./run_8gpu.sh --overwrite   # regenerate existing videos

set -e

# ── Configuration ────────────────────────────────────────────────────────────
CONDA_ENV="wan2"
PYTHON="/home/hpc/miniconda3/envs/$CONDA_ENV/bin/python"
TORCHRUN="/home/hpc/miniconda3/envs/$CONDA_ENV/bin/torchrun"

DATASET_PATH="/home/hpc/captioning/data"
CKPT_DIR="Wan2.2-TI2V-5B"
TEXT_KEY="gpt-4o_withGEST_t-1.0"

GPUS_PER_INSTANCE=4   # GPUs per FSDP group (2 OOMs at 832×480)
NUM_INSTANCES=2       # 2 × 4 = 8 GPUs total
NUM_SHARDS=2          # total shards across ALL machines; set to 3 when coordinating with RTX3090
BASE_PORT=29500       # torchrun master ports: 29500, 29501

TASK="ti2v-5B"
SIZE="832*480"
FRAME_NUM=121
SAMPLE_STEPS=30
SAMPLE_SOLVER="unipc"
SEED=42
FP16_FLAG="--fp16"    # P100 has no BF16
OFFLOAD_FLAG="--offload_model"  # offload DiT to CPU before VAE decode (required for P100 16GB)

# Fix NVML driver/library version mismatch (temporary until server is rebooted)
NVML_COMPAT=/tmp/nvidia9505/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
if [ -f "$NVML_COMPAT" ]; then
    export LD_PRELOAD="$NVML_COMPAT"
    echo "NVML compat: preloading $NVML_COMPAT"
else
    echo "WARNING: NVML compat library not found at $NVML_COMPAT — NCCL may fail."
    echo "Run: cd /tmp && apt-get download libnvidia-compute-580=580.95.05-0ubuntu1 && dpkg-deb -x libnvidia-compute-580_580.95.05-0ubuntu1_amd64.deb /tmp/nvidia9505/"
fi
# ─────────────────────────────────────────────────────────────────────────────

# Reduce CUDA memory fragmentation (recovers ~150 MiB of reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)      EXTRA_ARGS+=(--max_videos 4);  echo "TEST MODE: ~2 videos per shard"; shift ;;
        --single)    EXTRA_ARGS+=(--max_videos 2);  echo "SINGLE MODE: ~1 video per shard"; shift ;;
        --overwrite) EXTRA_ARGS+=(--overwrite);     echo "OVERWRITE MODE"; shift ;;
        --max)       EXTRA_ARGS+=(--max_videos "$2"); echo "Max $2 videos total"; shift 2 ;;
        *)
            echo "Usage: $0 [--single] [--test] [--overwrite] [--max N]"
            exit 1 ;;
    esac
done

echo "========================================"
echo "8-GPU Batch Video Generation - Wan2.2"
echo "========================================"
echo "Instances: $NUM_INSTANCES  (${GPUS_PER_INSTANCE} GPUs each)"
echo "GPU layout:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    START=$((i * GPUS_PER_INSTANCE))
    END=$((START + GPUS_PER_INSTANCE - 1))
    echo "  Shard $i → GPU $START-$END  port $((BASE_PORT + i))  log: $LOG_DIR/shard_${i}.log"
done
echo "========================================"
echo ""

# Launch all instances in parallel
PIDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    START=$((i * GPUS_PER_INSTANCE))
    END=$((START + GPUS_PER_INSTANCE - 1))
    GPU_LIST=$(seq -s, $START $END)
    PORT=$((BASE_PORT + i))
    LOG="$LOG_DIR/shard_${i}.log"

    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    "$TORCHRUN" \
        --nproc_per_node="$GPUS_PER_INSTANCE" \
        --master_port="$PORT" \
        batch_generate_videos.py \
            --dataset_path "$DATASET_PATH" \
            --ckpt_dir "$CKPT_DIR" \
            --text_key "$TEXT_KEY" \
            --task "$TASK" \
            --size "$SIZE" \
            --frame_num "$FRAME_NUM" \
            --sample_steps "$SAMPLE_STEPS" \
            --sample_solver "$SAMPLE_SOLVER" \
            --seed "$SEED" \
            --shard_idx "$i" \
            --num_shards "$NUM_SHARDS" \
            --log_file "$LOG" \
            $FP16_FLAG \
            $OFFLOAD_FLAG \
            "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/shard_${i}_stdout.log" 2>&1 &

    PIDS+=($!)
    echo "Launched shard $i (GPU $GPU_LIST, port $PORT, PID ${PIDS[-1]})"
done

echo ""
echo "All $NUM_INSTANCES shards running. Waiting for completion..."
echo "  Tail logs:  tail -f $LOG_DIR/shard_0.log"
echo "  Kill all:   kill ${PIDS[*]}"
echo ""

# Wait for all and report exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "Shard $i (PID $PID): done"
    else
        echo "Shard $i (PID $PID): FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
if [ "$FAILED" -eq 0 ]; then
    echo "All shards completed successfully!"
else
    echo "$FAILED shard(s) failed. Check logs in $LOG_DIR/"
fi
echo "Logs: $LOG_DIR/"
echo "========================================"
