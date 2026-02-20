# Batch Video Generation with Wan2.2

This script automates video generation for your dataset by iterating through folders, reading text descriptions from `texts.json` files, and generating videos using Wan2.2.

## Overview

The script will:
1. Scan through `segmentations_balanced_1`, `segmentations_balanced_2`, and `segmentations_balanced_3` folders
2. Look for `texts.json` files in subdirectories
3. Extract descriptions from the specified JSON key (default: `gpt-4o_withGEST_t-1.0`)
4. Optionally look for reference images in each folder (for TI2V mode)
5. Generate a video and save it as `wan2.2.mp4` in each folder

## Quick Start

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run batch generation
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --size 1280*704 \
    --sample_steps 50
```

### Test Run (First 5 Videos)

Test on a small subset first to verify everything works:

```bash
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --max_videos 5 \
    --log_file test_generation.log
```

## Command-Line Arguments

### Required Arguments

- `--dataset_path`: Base path to your dataset
  - Example: `"/mnt/g/My Drive/Archive - PhD/sa_video_story_engine"`
  
- `--ckpt_dir`: Path to Wan2.2 model checkpoint
  - Example: `"Wan2.2-TI2V-5B"`

### Dataset Arguments

- `--text_key`: JSON key to extract text from (default: `gpt-4o_withGEST_t-1.0`)

### Model Arguments

- `--task`: Model type to use
  - Options: `ti2v-5B` (text+image to video), `t2v-A14B` (text to video)
  - Default: `ti2v-5B`
  - **Recommended**: `ti2v-5B` (works with or without reference images)

### Generation Parameters

- `--size`: Video resolution (default: `1280*704`)
  - Other options: `832*480`, `768*768`, etc.
  
- `--frame_num`: Number of frames to generate
  - Default: Model's default (usually 121 for ti2v-5B)
  
- `--sample_steps`: Number of sampling steps (default: 50)
  - Higher = better quality but slower
  - Range: 20-100
  
- `--sample_solver`: Solver algorithm
  - Options: `unipc` (default), `dpm++`
  
- `--seed`: Random seed for reproducibility (default: 42)

### System Arguments

- `--offload_model`: Offload model to CPU when not in use (default: True)
  - Saves GPU memory between generations
  
- `--overwrite`: Overwrite existing `wan2.2.mp4` files
  - Default: False (skip existing videos)
  
- `--max_videos`: Limit number of videos to generate
  - Useful for testing
  
- `--log_file`: Path to log file (default: `batch_generation.log`)

## Example Commands

### Full Production Run

```bash
# Generate all videos with high quality
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --size 1280*704 \
    --sample_steps 50 \
    --sample_solver unipc \
    --seed 42 \
    --log_file full_batch_generation.log
```

### Fast Test Run (Lower Quality)

```bash
# Quick test with fewer steps
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --size 832*480 \
    --sample_steps 20 \
    --max_videos 3
```

### Resume Failed Generations

```bash
# Skip existing videos, only generate missing ones
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --log_file resume_generation.log
```

### Regenerate All (Overwrite)

```bash
# Regenerate all videos even if they exist
python batch_generate_videos.py \
    --dataset_path "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" \
    --ckpt_dir "Wan2.2-TI2V-5B" \
    --task ti2v-5B \
    --overwrite \
    --log_file regenerate_all.log
```

## Reference Images

The script automatically looks for reference images in each folder with these filenames (in order):
- `frame.png`, `frame.jpg`, `frame.jpeg`
- `reference.png`, `reference.jpg`, `reference.jpeg`
- `image.png`, `image.jpg`, `image.jpeg`
- `ref.png`, `ref.jpg`, `ref.jpeg`
- `input.png`, `input.jpg`, `input.jpeg`
- Or any other `.png`, `.jpg`, `.jpeg` file

If a reference image is found, it will be used with the text description for TI2V generation.
If no image is found, only the text description will be used.

## Monitoring Progress

### View Log in Real-Time

```bash
# In another terminal
tail -f batch_generation.log
```

### Check Generated Videos

```bash
# Count how many wan2.2.mp4 files exist
find "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" -name "wan2.2.mp4" | wc -l

# List all generated videos
find "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" -name "wan2.2.mp4"
```

## Expected Performance

With your RTX 3090 (24GB VRAM):
- **Time per video**: ~10-15 minutes at 50 steps
- **GPU Memory**: ~18-20GB
- **Video size**: ~8-10MB per video

For large datasets, consider:
1. Running overnight or over multiple days
2. Using `--max_videos` to process in batches
3. Using `--sample_steps 30` for faster (but slightly lower quality) generation

## Troubleshooting

### Out of Memory

If you get CUDA out of memory errors:
```bash
# Use smaller resolution
--size 832*480

# Enable model offloading (should be on by default)
--offload_model
```

### Dataset Path Not Found

Make sure the G: drive is mounted in WSL2:
```bash
# Check if drive is accessible
ls "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine"

# If not, check mounted drives
ls /mnt/
```

### No Folders Found

The script looks for folders with `texts.json` containing the specified key. Check:
```bash
# Find texts.json files
find "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine" -name "texts.json" | head -20

# Check a texts.json file structure
cat "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/segmentations_balanced_1/<some-folder>/texts.json" | jq .
```

## Directory Structure Expected

```
G:\My Drive\Archive - PhD\sa_video_story_engine\
├── segmentations_balanced_1\
│   ├── house_max5actors_max2regions_2action_chains_41c2af58\
│   │   ├── texts.json              # Contains "gpt-4o_withGEST_t-1.0" key
│   │   ├── frame.png               # Optional reference image
│   │   └── wan2.2.mp4              # Generated output
│   └── ...
├── segmentations_balanced_2\
│   └── ...
└── segmentations_balanced_3\
    └── ...
```

## Output

Each folder will contain:
- `wan2.2.mp4`: Generated video
- Logs will be written to `batch_generation.log` (or specified log file)

## Notes

- The script uses the **ti2v-5B** model by default (text+image to video)
- Videos are generated at **1280x704** resolution by default
- The script **skips folders that already have wan2.2.mp4** unless `--overwrite` is used
- Generation is **sequential** (one video at a time) to avoid memory issues
- GPU memory is cleared between videos with `torch.cuda.empty_cache()`
