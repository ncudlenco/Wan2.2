# Performance Optimization Guide for RTX 3090 (24GB)

## Your Current Issue
- **4 hours per 5-second video** with T2V-A14B
- **Cause:** T2V-A14B requires 60-80GB VRAM, your 3090 has 24GB → severe CPU swapping
- **Solution:** Use TI2V-5B with optimizations → **10-20 minutes per video**

## Key Optimizations for RTX 3090 (24GB VRAM)

### 1. **CRITICAL: Use ONLY TI2V-5B**
```bash
# ✅ WORKS on 24GB (TI2V-5B model, 10-20 min per video)
--task ti2v-5B --ckpt_dir ./Wan2.2-TI2V-5B

# ❌ WILL NOT WORK on 24GB (requires 60-80GB VRAM)
--task t2v-A14B --ckpt_dir ./Wan2.2-T2V-A14B
```

**Why TI2V-5B is the ONLY option for 24GB:**
- ✅ 5B parameters (fits in 24GB with optimizations)
- ✅ Efficient Wan2.2-VAE (16×16×4 compression)
- ✅ Designed for consumer GPUs like RTX 3090
- ✅ Still supports text-only generation
- ❌ T2V-A14B/I2V-A14B: 14B MoE models need 60-80GB minimum

### 2. **Enable BF16 Conversion (REQUIRED)**
```bash
--convert_model_dtype  # Now enabled by default
```
Converts model to bfloat16: **2x speedup** + **50% VRAM reduction**. Without this, you'll OOM.

### 3. **Offload T5 to CPU (REQUIRED)**
```bash
--t5_cpu  # Now enabled by default
```
T5 encoder uses ~8GB VRAM. Moving to CPU frees critical VRAM. Small speed impact since T5 only runs once per video.

### 4. **Model Offloading (REQUIRED)**
```bash
--offload_model  # Now enabled by default
```
Moves inactive model components to CPU/RAM. Essential for 24GB GPUs.

### 5. **Low VRAM Mode (REQUIRED)**
```bash
--low_vram_mode  # Now enabled by default
```
Aggressive garbage collection + CUDA synchronization after each video.

## Recommended Command for RTX 3090 (24GB)

### **Conservative (Always works, ~10-15 min/video)**
```bash
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/" \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --task ti2v-5B \
    --size 832*480 \
    --sample_steps 30 \
    --max_videos 5
```
**All critical flags (t5_cpu, convert_model_dtype, offload_model, low_vram_mode) are now enabled by default.**

### **High Quality (May work, ~15-25 min/video)**
```bash
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/" \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --task ti2v-5B \
    --size 1280*720 \
    --sample_steps 40 \
    --max_videos 2
```
**Warning:** 720p may OOM on some scenes. Monitor with `nvidia-smi -l 1`.

## Sampling Steps Trade-off (RTX 3090)

Balance quality vs speed:

| Steps | Quality      | Speed     | Time/Video (24GB) | VRAM   |
|-------|--------------|-----------|-------------------|--------|
| 50    | Best         | Slow      | 20-25 min         | ~23GB  |
| 40    | Very Good    | Medium    | 15-20 min         | ~22GB  |
| 30    | Good         | Fast      | 10-15 min         | ~21GB  |
| 25    | Acceptable   | Very Fast | 8-12 min          | ~20GB  |

```bash
--sample_steps 30  # Recommended balance for 3090
```

## Multi-GPU Setup (If you have 2x 3090s)

With 2x RTX 3090 (48GB combined), you can use FSDP:

```bash
# NOTE: Requires script modifications for FSDP args
torchrun --nproc_per_node=2 batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/" \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --task ti2v-5B \
    --size 1280*720 \
    --sample_steps 40
```

This enables 720p high quality generation reliably.

## Video Length Optimization

For 5-second videos at 24fps:
```bash
--frame_num 121  # 5 seconds at 24fps (default for TI2V-5B)
```

For shorter videos (faster generation):
```bash
--frame_num 61   # 2.5 seconds at 24fps
--frame_num 81   # 3.3 seconds at 24fps
```

## Expected Performance on RTX 3090 (24GB)

| Configuration                           | Time/Video | VRAM  | Status      |
|-----------------------------------------|------------|-------|-------------|
| TI2V-5B + 832*480 + 30 steps (optimized)| 10-15 min  | 18-21GB | ✅ Recommended |
| TI2V-5B + 1280*720 + 40 steps          | 15-25 min  | 21-24GB | ⚠️ May OOM    |
| T2V-A14B (any settings)                 | N/A        | 60-80GB | ❌ Won't fit   |
| **Your old setup** (T2V-A14B, no opt)  | 4 hours    | OOM + Swap | ❌ CPU swapping |

## Monitoring Performance

The script now logs timing information:
```
Generation completed in 480.5s (8.0 minutes)
Total time: 495.2s (8.3 minutes)
```

## Additional Tips for RTX 3090

1. **Monitor VRAM**: Run `watch -n 1 nvidia-smi` in another terminal during generation
2. **Use Reference Frames**: Extract frames from your MP4s - TI2V can use them for consistency
3. **Test First**: Use `--max_videos 2` to test settings before full batch
4. **Room Temperature**: 3090s run hot - ensure good airflow
5. **Gradual Increase**: Start with 832×480/30 steps, then try 720p if stable
6. **Close Other Apps**: Chrome/browsers can use 1-2GB VRAM
7. **One Video at a Time**: Script processes sequentially to avoid memory fragmentation

## Common Issues on RTX 3090

### Out of Memory (OOM)
```bash
# All these are now DEFAULT, but verify:
--t5_cpu \
--offload_model \
--convert_model_dtype \
--low_vram_mode

# If still OOM, reduce resolution:
--size 832*480  # or even 640*368

# Or reduce frames:
--frame_num 61  # ~2.5 seconds
```

### Still Too Slow (>30 min/video)
```bash
# Reduce sampling steps (biggest impact):
--sample_steps 25

# Reduce resolution:
--size 832*480

# Shorter videos:
--frame_num 61
```

### CUDA Out of Memory Error
If you see "CUDA out of memory" errors:
1. Close all other GPU applications (browsers, other Python scripts)
2. Restart Python and try again with lower settings
3. Use `--size 640*368` as minimum viable resolution

## Extracting Reference Frames from MP4

To use your existing MP4 files as reference frames:

```bash
# Extract first frame from each video
cd /mnt/g/My\ Drive/Archive\ -\ PhD/sa_video_story_engine/
find segmentations_balanced_* -name "raw.mp4" -type f | while read video; do
    dir=$(dirname "$video")
    ffmpeg -i "$video" -vf "select=eq(n\,0)" -frames:v 1 "$dir/frame.png" -y
done
```

Then TI2V will automatically pick up `frame.png` and use it!
