#!/usr/bin/env python3
"""
Batch video generation script for Wan2.2
Iterates through dataset folders and generates videos from text descriptions
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import traceback

import torch
from PIL import Image
import cv2
import numpy as np

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video


def setup_logging(log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=handlers
    )


def find_dataset_folders(base_path, text_key="gpt-4o_withGEST_t-1.0", max_limit=None, skip_existing=True):
    """
    Find folders containing texts.json with the specified key
    
    Args:
        base_path: Base path like "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/"
        text_key: JSON key to look for in texts.json
        max_limit: Stop searching after finding this many folders (for efficiency)
        skip_existing: Skip folders that already have wan2.2.mp4
        
    Returns:
        List of (folder_path, text_description) tuples
    """
    folders_to_process = []
    
    # Iterate through segmentations_balanced_1, 2, and 3
    for seg_num in [1, 2, 3]:
        seg_path = os.path.join(base_path, f"segmentations_balanced_{seg_num}")
        
        if not os.path.exists(seg_path):
            logging.warning(f"Path does not exist: {seg_path}")
            continue
        
        logging.info(f"Scanning {seg_path}...")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(seg_path):
            if "texts.json" in files:
                texts_json_path = os.path.join(root, "texts.json")
                
                # Check if video already exists
                if skip_existing and os.path.exists(os.path.join(root, "wan2.2.mp4")):
                    logging.debug(f"Skipping {root} - wan2.2.mp4 already exists")
                    continue
                
                try:
                    with open(texts_json_path, 'r', encoding='utf-8') as f:
                        texts_data = json.load(f)
                    
                    # Check if the key exists
                    if text_key in texts_data:
                        description = texts_data[text_key]
                        if description and isinstance(description, str) and len(description.strip()) > 0:
                            folders_to_process.append((root, description.strip()))
                            logging.info(f"Found: {root}")
                            
                            # Early exit if we have enough folders
                            if max_limit and len(folders_to_process) >= max_limit:
                                logging.info(f"Reached limit of {max_limit} folders, stopping search")
                                return folders_to_process
                        else:
                            logging.warning(f"Empty or invalid text in {texts_json_path}")
                    else:
                        logging.debug(f"Key '{text_key}' not found in {texts_json_path}")
                        
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON in {texts_json_path}: {e}")
                except Exception as e:
                    logging.error(f"Error reading {texts_json_path}: {e}")
    
    return folders_to_process


def extract_frame_from_video(video_path, frame_number, output_path=None):
    """
    Extract a specific frame from a video file
    
    Args:
        video_path: Path to video file (e.g., raw.mp4)
        frame_number: Frame number to extract (0-indexed)
        output_path: Optional path to save frame as image
        
    Returns:
        PIL Image or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return None
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logging.error(f"Failed to read frame {frame_number} from {video_path}")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Optionally save
        if output_path:
            img.save(output_path)
            logging.info(f"Saved extracted frame to: {output_path}")
        
        return img
        
    except Exception as e:
        logging.error(f"Error extracting frame from video: {e}")
        return None


def get_reference_frame_from_mapping(folder_path):
    """
    Extract reference frame from raw.mp4 based on event_frame_mapping.json
    
    Searches for:
    - folder_path/simulations/take1_sim1/event_frame_mapping.json
    - folder_path/simulations/take1_sim1/camera1/raw.mp4
    
    Args:
        folder_path: Base folder containing texts.json
        
    Returns:
        PIL Image or None
    """
    # Look for simulations folder
    sim_base = os.path.join(folder_path, "simulations")
    if not os.path.exists(sim_base):
        logging.debug(f"No simulations folder found in {folder_path}")
        return None
    
    # Find first simulation folder (take1_sim1, etc.)
    sim_folders = [d for d in os.listdir(sim_base) if os.path.isdir(os.path.join(sim_base, d))]
    if not sim_folders:
        logging.debug(f"No simulation folders found in {sim_base}")
        return None
    
    sim_folder = os.path.join(sim_base, sim_folders[0])  # Use first sim folder
    
    # Load event_frame_mapping.json
    mapping_path = os.path.join(sim_folder, "event_frame_mapping.json")
    if not os.path.exists(mapping_path):
        logging.debug(f"No event_frame_mapping.json found at {mapping_path}")
        return None
    
    try:
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Extract frame number - use first event's start frame, or middle frame
        if not mapping_data or not isinstance(mapping_data, list) or len(mapping_data) == 0:
            logging.warning(f"Invalid event_frame_mapping.json format in {mapping_path}")
            return None
        
        events = mapping_data[0].get('events', [])
        fps = mapping_data[0].get('fps', 60)
        
        if not events:
            logging.warning(f"No events found in {mapping_path}")
            return None
        
        # Choose a representative frame - use the middle of the first event
        first_event = events[0]
        if 'startFrame' in first_event and 'endFrame' in first_event:
            start_frame = first_event['startFrame']
            end_frame = first_event['endFrame']
            reference_frame = (start_frame + end_frame) // 2
        elif 'startFrame' in first_event:
            reference_frame = first_event['startFrame']
        else:
            logging.warning(f"Event missing frame information in {mapping_path}")
            return None
        
        logging.info(f"Using frame {reference_frame} from event_frame_mapping.json (FPS: {fps})")
        
        # Find raw.mp4 in camera folders
        camera_folders = ['camera1', 'camera2', 'camera3']  # Common camera folder names
        video_path = None
        
        for camera_folder in camera_folders:
            potential_path = os.path.join(sim_folder, camera_folder, "raw.mp4")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if not video_path:
            logging.warning(f"No raw.mp4 found in simulation folder {sim_folder}")
            return None
        
        logging.info(f"Extracting frame {reference_frame} from {video_path}")
        
        # Extract frame
        img = extract_frame_from_video(video_path, reference_frame)
        
        if img:
            logging.info(f"Successfully extracted reference frame: {img.size}")
        
        return img
        
    except Exception as e:
        logging.error(f"Error processing event_frame_mapping.json: {e}")
        logging.error(traceback.format_exc())
        return None


def find_reference_image(folder_path):
    """
    Extract reference frame from raw.mp4 using event_frame_mapping.json
    
    Returns:
        PIL Image object or None (for text-only mode)
    """
    # Extract frame from video using event mapping
    img = get_reference_frame_from_mapping(folder_path)
    
    if img is None:
        logging.info("No raw.mp4 or event_frame_mapping.json found - will use text-only mode")
    
    return img


def generate_video_for_folder(folder_path, description, pipeline, cfg, args):
    """
    Generate a video for a specific folder
    
    Args:
        folder_path: Path to the folder
        description: Text description
        pipeline: Wan pipeline object (WanTI2V or WanT2V)
        cfg: Model config
        args: Arguments
    """
    output_path = os.path.join(folder_path, "wan2.2.mp4")
    
    # Skip if already exists
    if os.path.exists(output_path) and not args.overwrite:
        logging.info(f"Skipping {folder_path} - video already exists")
        return True
    
    try:
        start_time = time.time()
        logging.info(f"Processing: {folder_path}")
        logging.info(f"Description: {description[:100]}...")
        
        # Get reference image/frame (extracted from raw.mp4 via event_frame_mapping.json)
        img = None
        if args.use_reference_frame:
            img = find_reference_image(folder_path)
            if img is not None:
                logging.info(f"Using extracted reference frame: {img.size}")
            else:
                logging.info("No reference frame available, using text-only mode")
        else:
            logging.info("Reference frame disabled, using text-only mode")
        
        # Generate video
        logging.info("Generating video...")
        gen_start = time.time()
        
        if hasattr(pipeline, 'generate'):
            if img is not None:
                # TI2V with image
                video = pipeline.generate(
                    description,
                    img=img,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.seed,
                    offload_model=args.offload_model
                )
            else:
                # T2V without image (if ti2v pipeline can handle it)
                video = pipeline.generate(
                    description,
                    img=None,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.seed,
                    offload_model=args.offload_model
                )
        else:
            logging.error("Pipeline does not have generate method")
            return False
        
        gen_time = time.time() - gen_start
        logging.info(f"Generation completed in {gen_time:.1f}s ({gen_time/60:.1f} minutes)")
        
        # Save video
        logging.info(f"Saving video to: {output_path}")
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        # Clean up (aggressive for low VRAM)
        del video
        if img is not None:
            del img
        torch.cuda.empty_cache()
        
        # Extra cleanup for 24GB GPU
        if args.low_vram_mode:
            import gc
            gc.collect()
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        logging.info(f"Successfully generated: {output_path}")
        logging.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate video for {folder_path}")
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch generate videos with Wan2.2")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help='Base path to dataset, e.g., "/mnt/g/My Drive/Archive - PhD/sa_video_story_engine/"'
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="gpt-4o_withGEST_t-1.0",
        help="JSON key in texts.json to extract description from"
    )
    
    # Model arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to Wan2.2 checkpoint directory"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ti2v-5B",
        choices=["ti2v-5B", "t2v-A14B", "i2v-A14B"],
        help="Task type (ti2v-5B ONLY option for 24GB GPU like RTX 3090)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        help="Video size (832*480 for 480p, 1280*704 for HD)"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=121,
        help="Number of frames (121 = 5s at 24fps, 81 = 3.4s, 193 = 8s)"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help="Number of sampling steps (25 for speed, 30 balanced, 40 for quality)"
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor"
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale"
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="Solver for sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # System arguments (optimized for 24GB GPU)
    parser.add_argument(
        "--offload_model",
        action="store_true",
        default=False,
        help="Offload model to CPU when not in use (REQUIRED for 24GB GPU)"
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=True,
        help="Keep T5 encoder on CPU to save VRAM (REQUIRED for 24GB GPU)"
    )
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=True,
        help="Convert model to bfloat16 for faster inference (REQUIRED for 24GB GPU)"
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        default=True,
        help="Enable aggressive VRAM optimization for 24GB GPUs"
    )
    parser.add_argument(
        "--use_reference_frame",
        action="store_true",
        default=False,
        help="Extract and use reference frame from raw.mp4 (disabled by default for text-only mode)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing wan2.2.mp4 files"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to generate (for testing)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="batch_generation.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logging.info("="*80)
    logging.info("Starting batch video generation")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Checkpoint: {args.ckpt_dir}")
    logging.info(f"Task: {args.task}")
    logging.info("="*80)
    
    # Check paths
    if not os.path.exists(args.dataset_path):
        logging.error(f"Dataset path does not exist: {args.dataset_path}")
        return
    
    if not os.path.exists(args.ckpt_dir):
        logging.error(f"Checkpoint directory does not exist: {args.ckpt_dir}")
        return
    
    # Find all folders to process
    logging.info("Scanning for folders with texts.json...")
    folders_to_process = find_dataset_folders(
        args.dataset_path, 
        args.text_key,
        max_limit=args.max_videos,  # Stop scanning early if limit is set
        skip_existing=not args.overwrite  # Skip folders with existing videos unless overwrite
    )
    
    if not folders_to_process:
        logging.warning("No folders found to process!")
        return
    
    logging.info(f"Found {len(folders_to_process)} folders to process")
    
    # Initialize model
    logging.info("Initializing Wan2.2 pipeline...")
    cfg = WAN_CONFIGS[args.task]
    
    # Set defaults from config
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    
    logging.info(f"Config: {cfg}")
    
    # Warn about memory constraints
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total VRAM: {total_vram:.1f} GB")
        
        if total_vram < 40 and args.task == "t2v-A14B":
            logging.error("ERROR: T2V-A14B requires >40GB VRAM. You have {:.1f}GB.".format(total_vram))
            logging.error("Use --task ti2v-5B instead!")
            return
        
        if total_vram < 40:
            # Check if optimizations are enabled
            missing_opts = []
            if not args.t5_cpu:
                missing_opts.append("--t5_cpu")
            if not args.offload_model:
                missing_opts.append("--offload_model")
            if not args.convert_model_dtype:
                missing_opts.append("--convert_model_dtype")
            
            if missing_opts:
                logging.warning(f"WARNING: {total_vram:.1f}GB VRAM detected. Consider enabling: {', '.join(missing_opts)}")
            else:
                logging.info(f"Memory optimizations enabled (T5-CPU, model offloading, BF16) for {total_vram:.1f}GB VRAM")
    
    # Create pipeline
    device_id = 0
    rank = 0
    
    if "ti2v" in args.task:
        pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            init_on_cpu=True,
        )
    elif "t2v" in args.task:
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            init_on_cpu=True,
        )
    elif "i2v" in args.task:
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            init_on_cpu=True,
        )
    else:
        logging.error(f"Unsupported task: {args.task}")
        return
    
    logging.info("Pipeline initialized successfully")
    
    # Process each folder
    success_count = 0
    fail_count = 0
    
    for idx, (folder_path, description) in enumerate(folders_to_process, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing {idx}/{len(folders_to_process)}: {folder_path}")
        logging.info(f"{'='*80}")
        
        success = generate_video_for_folder(
            folder_path, description, pipeline, cfg, args
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        logging.info(f"Progress: {success_count} succeeded, {fail_count} failed")
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("BATCH GENERATION COMPLETE")
    logging.info(f"Total folders processed: {len(folders_to_process)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {fail_count}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
