#!/usr/bin/env python3
"""
Batch video generation script for Wan2.2 using HuggingFace Diffusers + BitsAndBytes int8
Optimized for RTX 3090 24GB - eliminates offloading overhead via quantization

Requirements:
    pip install diffusers transformers accelerate bitsandbytes sentencepiece ftfy -U

Model:
    Wan-AI/Wan2.2-TI2V-5B-Diffusers (auto-downloaded from HuggingFace on first run)
    Or pre-download: huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
"""

import argparse
import json
import logging
import os
import sys
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
import cv2
import numpy as np

from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.utils import export_to_video


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


def create_pipeline(args):
    """
    Create the diffusers pipeline with optional quantization.
    
    Returns:
        (pipe, fps) tuple
    """
    model_id = args.model_id
    
    logging.info(f"Loading model: {model_id}")
    logging.info(f"Quantization: {'int8' if args.quantize_8bit else 'int4' if args.quantize_4bit else 'none (bf16)'}")
    
    # VAE must stay in float32 for quality
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    
    # Setup quantization config for the transformer
    if args.quantize_8bit:
        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        logging.info(f"Transformer loaded with int8 quantization")
    elif args.quantize_4bit:
        quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        logging.info(f"Transformer loaded with int4 quantization")
    else:
        transformer = None  # Let pipeline load it normally
        logging.info(f"Transformer loaded in bf16 (no quantization)")
    
    # Load pipeline with pre-quantized transformer
    pipe_kwargs = {
        "vae": vae,
        "torch_dtype": torch.bfloat16,
    }
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer
    
    pipe = WanPipeline.from_pretrained(model_id, **pipe_kwargs)
    
    # Memory strategy: try to keep everything on GPU (the whole point of quantization)
    # If quantized, model should fit in 24GB without offloading
    if args.use_cpu_offload:
        logging.info("Using enable_model_cpu_offload() (smart offloading)")
        pipe.enable_model_cpu_offload()
    else:
        logging.info("Moving pipeline to CUDA (no offloading)")
        pipe.to("cuda")
    
    # TI2V-5B runs at 24 FPS
    fps = 24
    
    return pipe, fps


def generate_video_for_folder(folder_path, description, pipe, fps, args):
    """
    Generate a video for a specific folder using diffusers pipeline
    
    Args:
        folder_path: Path to the folder
        description: Text description
        pipe: Diffusers WanPipeline
        fps: Frames per second for output
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
        
        # Build generation kwargs
        gen_kwargs = {
            "prompt": description,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frame_num,
            "num_inference_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
        }
        
        # Add image for TI2V mode (image-to-video)
        if img is not None:
            gen_kwargs["image"] = img
        
        # Add negative prompt if provided
        if args.negative_prompt:
            gen_kwargs["negative_prompt"] = args.negative_prompt
        
        # Set seed for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        gen_kwargs["generator"] = generator
        
        # Generate video
        logging.info(f"Generating video ({args.width}x{args.height}, {args.frame_num} frames, {args.sample_steps} steps)...")
        gen_start = time.time()
        
        output = pipe(**gen_kwargs)
        video_frames = output.frames[0]  # List of PIL Images
        
        gen_time = time.time() - gen_start
        logging.info(f"Generation completed in {gen_time:.1f}s ({gen_time/60:.1f} minutes)")
        
        # Save video
        logging.info(f"Saving video to: {output_path}")
        export_to_video(video_frames, output_path, fps=fps)
        
        # Clean up
        del video_frames, output
        if img is not None:
            del img
        torch.cuda.empty_cache()
        gc.collect()
        
        total_time = time.time() - start_time
        logging.info(f"Successfully generated: {output_path}")
        logging.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OOM for {folder_path}! Try --quantize_4bit or --use_cpu_offload")
        logging.error(f"Error: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        return False
        
    except Exception as e:
        logging.error(f"Failed to generate video for {folder_path}")
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        torch.cuda.empty_cache()
        gc.collect()
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch generate videos with Wan2.2 (Diffusers + BitsAndBytes)")
    
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
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="HuggingFace model ID or local path to diffusers model"
    )
    
    # Quantization (mutually exclusive)
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--quantize_8bit",
        action="store_true",
        default=True,
        help="Use int8 quantization via BitsAndBytes (default, recommended for 24GB GPU)"
    )
    quant_group.add_argument(
        "--quantize_4bit",
        action="store_true",
        default=False,
        help="Use int4 quantization via BitsAndBytes (more aggressive, slight quality loss)"
    )
    quant_group.add_argument(
        "--no_quantize",
        action="store_true",
        default=False,
        help="Disable quantization (bf16 only, may OOM on 24GB without offloading)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height (704 for TI2V-5B 720p)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (1280 for TI2V-5B 720p)"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=121,
        help="Number of frames (121 = 5s at 24fps, 81 = 3.4s)"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help="Number of sampling steps (25 for speed, 30 balanced, 40+ for quality)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # System arguments
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        default=False,
        help="Use smart CPU offloading (fallback if OOM without it)"
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
        default=None,
        help="Path to log file (default: batch_generation_diffusers_TIMESTAMP.log)"
    )
    
    args = parser.parse_args()
    
    # Handle mutually exclusive quantization args
    if args.no_quantize:
        args.quantize_8bit = False
        args.quantize_4bit = False
    elif args.quantize_4bit:
        args.quantize_8bit = False
    
    # Default log file with timestamp
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"batch_generation_diffusers_{timestamp}.log"
    
    # Setup logging
    setup_logging(args.log_file)
    logging.info("=" * 80)
    logging.info("Starting batch video generation (Diffusers + BitsAndBytes)")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Model: {args.model_id}")
    quant_mode = "int8" if args.quantize_8bit else "int4" if args.quantize_4bit else "bf16 (no quantization)"
    logging.info(f"Quantization: {quant_mode}")
    logging.info(f"Resolution: {args.width}x{args.height}")
    logging.info(f"Frames: {args.frame_num} ({args.frame_num / 24:.1f}s at 24fps)")
    logging.info(f"Steps: {args.sample_steps}")
    logging.info(f"CPU offload: {args.use_cpu_offload}")
    logging.info("=" * 80)
    
    # Check paths
    if not os.path.exists(args.dataset_path):
        logging.error(f"Dataset path does not exist: {args.dataset_path}")
        return
    
    # GPU info
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total VRAM: {total_vram:.1f} GB")
    else:
        logging.error("No CUDA GPU available!")
        return
    
    # Find all folders to process
    logging.info("Scanning for folders with texts.json...")
    folders_to_process = find_dataset_folders(
        args.dataset_path, 
        args.text_key,
        max_limit=args.max_videos,
        skip_existing=not args.overwrite
    )
    
    if not folders_to_process:
        logging.warning("No folders found to process!")
        return
    
    logging.info(f"Found {len(folders_to_process)} folders to process")
    
    # Initialize pipeline
    logging.info("Initializing diffusers pipeline...")
    init_start = time.time()
    
    pipe, fps = create_pipeline(args)
    
    init_time = time.time() - init_start
    logging.info(f"Pipeline initialized in {init_time:.1f}s ({init_time/60:.1f} minutes)")
    
    # Log VRAM usage after init
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logging.info(f"VRAM after init: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Process each folder
    success_count = 0
    fail_count = 0
    total_gen_time = 0
    batch_start = time.time()
    
    for idx, (folder_path, description) in enumerate(folders_to_process, 1):
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Processing {idx}/{len(folders_to_process)}: {folder_path}")
        logging.info(f"{'=' * 80}")
        
        video_start = time.time()
        success = generate_video_for_folder(
            folder_path, description, pipe, fps, args
        )
        video_time = time.time() - video_start
        
        if success:
            success_count += 1
            total_gen_time += video_time
            
            # ETA calculation
            avg_time = total_gen_time / success_count
            remaining = len(folders_to_process) - idx
            eta_seconds = avg_time * remaining
            eta_hours = eta_seconds / 3600
            logging.info(f"Avg time/video: {avg_time:.1f}s ({avg_time/60:.1f} min)")
            logging.info(f"ETA for remaining {remaining} videos: {eta_hours:.1f} hours ({eta_hours/24:.1f} days)")
        else:
            fail_count += 1
        
        logging.info(f"Progress: {success_count} succeeded, {fail_count} failed, {len(folders_to_process) - idx} remaining")
    
    # Summary
    total_time = time.time() - batch_start
    logging.info("\n" + "=" * 80)
    logging.info("BATCH GENERATION COMPLETE")
    logging.info(f"Total folders processed: {len(folders_to_process)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {fail_count}")
    logging.info(f"Total time: {total_time/3600:.1f} hours ({total_time/86400:.1f} days)")
    if success_count > 0:
        logging.info(f"Average time per video: {total_gen_time/success_count:.1f}s ({total_gen_time/success_count/60:.1f} min)")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()