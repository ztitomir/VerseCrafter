"""
VerseCrafter Standalone API Server

A Flask-based REST API server that provides endpoints for the VerseCrafter
video generation pipeline. This can be used as an alternative to ComfyUI.

Usage:
    # Multi-GPU mode (recommended) - model loads on GPUs 1-7 at startup, GPU 0 for other tasks
    python api_server.py --port 8188 --num_gpus 8
    
    # Single GPU mode - model loads after render, unloads after generation
    python api_server.py --port 8188 --num_gpus 1

GPU Allocation:
    Multi-GPU (num_gpus > 1):
        - GPU 0: Reserved for preprocessing (Step 1-3) and rendering (Step 5)
        - GPUs 1-N: Model server for video generation (Step 6)
        - Model starts loading immediately at server startup (parallel)
    
    Single GPU (num_gpus == 1):
        - GPU 0: Shared for all tasks (sequential)
        - Step 1-3 (preprocess) → Step 5 (render) → load model → Step 6 (generate) → unload model
        - Model loads AFTER render completes, unloads AFTER generation to free GPU

Endpoints:
    POST /api/preprocess - Run Steps 1-3 (depth, segmentation, gaussian fitting)
    POST /api/render     - Run Step 5 (render 4D control maps)
    POST /api/generate   - Run Step 6 (generate video)
    POST /api/workflow   - Run full postprocess workflow (Steps 5-6)
    GET  /api/status/<task_id> - Check task status
    GET  /api/download/<path>  - Download output file
"""

import os
import sys
import json
import uuid
import time
import threading
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(PROJECT_ROOT, 'inference')
VIDEOX_FUN_PATH = os.path.join(PROJECT_ROOT, 'third_party/VideoX-Fun')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, INFERENCE_DIR)
if VIDEOX_FUN_PATH not in sys.path:
    sys.path.insert(0, VIDEOX_FUN_PATH)

# Conda environment configuration
# Set CONDA_ENV to specify which conda environment to use for inference scripts
# If not set, uses the current Python interpreter
CONDA_ENV = os.environ.get('VERSECRAFTER_CONDA_ENV', '')  # e.g., 'videoxfun' or 'versecrafter'
CONDA_PREFIX = os.environ.get('CONDA_PREFIX', '')

def get_python_cmd():
    """Get the Python command to use for subprocess calls."""
    if CONDA_ENV:
        # Use conda run to execute in specific environment
        return ['conda', 'run', '-n', CONDA_ENV, '--no-capture-output', 'python']
    else:
        # Use current Python interpreter
        return [sys.executable]


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    type: str
    status: TaskStatus
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0
    completed_at: Optional[float] = None


# Global task storage
tasks: Dict[str, Task] = {}
task_lock = threading.Lock()

# ============================================================================
# Global Model State (for preloaded model mode)
# ============================================================================
_pipeline = None
_vae = None
_device = None
_weight_dtype = None
_model_config = {}
_model_loaded = False
_model_loading = False  # True when model is being loaded in background
_model_load_error = None  # Error message if loading failed
_model_load_thread = None  # Background loading thread
_loading_lock = threading.Lock()

# Startup configuration (stored for async loading)
_startup_config = {
    'num_gpus': 1,
    'model_path': 'model/VerseCrafter',
    'base_model_path': 'model/Wan2.1-T2V-14B',
    'config_path': 'config/wan2.1/wan_civitai.yaml',
    'gpu_memory_mode': 'model_full_load',
}

# Model server configuration (for multi-GPU mode)
_model_server_url = None  # Set when model server is started
_model_server_process = None  # Subprocess running the model server
_model_server_port = 8189


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,User-Agent')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


def create_task(task_type: str) -> Task:
    """Create a new task."""
    task = Task(
        id=str(uuid.uuid4()),
        type=task_type,
        status=TaskStatus.PENDING,
        progress=0.0,
        message="Task created",
        created_at=time.time()
    )
    with task_lock:
        tasks[task.id] = task
    return task


def update_task(task_id: str, **kwargs):
    """Update task status."""
    with task_lock:
        if task_id in tasks:
            task = tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)


def run_preprocess(task_id: str, params: Dict[str, Any]):
    """Run preprocessing pipeline (Steps 1-3)."""
    try:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.0, message="Starting preprocessing...")
        
        import os
        # IMPORTANT: Use GPU 0 for preprocessing to avoid conflict with model server
        # Model server uses GPUs 1-7, preprocessing uses GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logger.info(f"[Preprocess Task {task_id}] Using GPU 0 for preprocessing (CUDA_VISIBLE_DEVICES=0)")
        
        import numpy as np
        import torch
        from PIL import Image
        
        # Import inference modules
        from moge.model import import_model_class_by_version
        from moge.utils.vis import colorize_depth
        from grounded_sam2_infer import ImageSegmenter
        from fit_3D_gaussian import (
            get_point_cloud_from_depth, fit_3d_gaussian, tensor_to_json_serializable
        )
        import cv2
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract parameters
        image_path = params['image_path']
        output_dir = params['output_dir']
        text_prompt = params.get('text_prompt', 'person . car .')
        model_version = params.get('model_version', 'v2')
        use_fp16 = params.get('use_fp16', True)
        resolution_level = params.get('resolution_level', 9)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Depth Estimation
        update_task(task_id, progress=0.1, message="Step 1: Running depth estimation...")
        
        DEFAULT_MODELS = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        model_class = import_model_class_by_version(model_version)
        depth_model = model_class.from_pretrained(DEFAULT_MODELS[model_version]).to(device).eval()
        if use_fp16:
            depth_model.half()
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        with torch.no_grad():
            output = depth_model.infer(image_tensor, resolution_level=resolution_level, use_fp16=use_fp16)
        
        depth = output['depth'].cpu().numpy()
        intrinsics = output['intrinsics'].cpu().numpy()
        
        # Replace infinities
        valid_mask = np.isfinite(depth) & (depth > 0)
        if np.any(valid_mask):
            depth[~valid_mask] = float(np.max(depth[valid_mask])) + 10.0
        
        # Save depth data
        depth_dir = output_path / "estimated_depth"
        depth_dir.mkdir(exist_ok=True)
        np.savez_compressed(str(depth_dir / "depth_intrinsics.npz"),
                           depth=depth.astype(np.float16), intrinsic=intrinsics.astype(np.float16))
        cv2.imwrite(str(depth_dir / "depth_vis.png"), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
        
        update_task(task_id, progress=0.4, message="Step 2: Running segmentation...")
        
        # Step 2: Segmentation
        segmenter = ImageSegmenter(device=str(device))
        image_source, detections, masks = segmenter.segment_image(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=params.get('box_threshold', 0.4),
            min_area_ratio=params.get('min_area_ratio', 0.003),
            max_area_ratio=params.get('max_area_ratio', 0.2)
        )
        
        if not detections or len(detections.get('masks', [])) == 0:
            update_task(task_id, status=TaskStatus.FAILED, error="No objects detected")
            return
        
        # Save masks
        masks_dir = output_path / "object_mask" / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (mask, label) in enumerate(zip(detections['masks'], detections['labels'])):
            mask_path = masks_dir / f"mask_{idx+1:02d}_{label}.png"
            Image.fromarray(mask).save(str(mask_path))
        
        # Save visualization
        vis_image = segmenter.visualize_results(image_source, detections)
        Image.fromarray(vis_image).save(str(output_path / "object_mask" / "visualization.png"))
        
        update_task(task_id, progress=0.7, message="Step 3: Fitting 3D Gaussians...")
        
        # Step 3: 3D Gaussian Fitting
        depth_tensor = torch.from_numpy(depth.astype(np.float32)).to(device)
        intrinsic_tensor = torch.from_numpy(intrinsics.astype(np.float32)).to(device)
        
        # Denormalize intrinsics
        if intrinsic_tensor[0, 0].item() < 10:
            intrinsic_tensor[0, 0] *= width
            intrinsic_tensor[1, 1] *= height
            intrinsic_tensor[0, 2] *= width
            intrinsic_tensor[1, 2] *= height
        
        extrinsic = torch.eye(4, device=device, dtype=torch.float32)
        
        gaussian_params = {}
        for idx, (mask, label) in enumerate(zip(detections['masks'], detections['labels'])):
            obj_id = idx + 1
            
            mask_binary = (mask > 127).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_eroded = cv2.erode(mask_binary * 255, kernel, iterations=1)
            mask_tensor = torch.from_numpy(mask_eroded > 127).to(device)
            
            if mask_tensor.shape != (height, width):
                mask_resized = cv2.resize(mask_eroded, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(mask_resized > 127).to(device)
            
            points = get_point_cloud_from_depth(depth_tensor, intrinsic_tensor, extrinsic, mask_tensor)
            
            if len(points) < 10:
                continue
            
            mean, cov = fit_3d_gaussian(points, device)
            if mean is None:
                continue
            
            gaussian_params[obj_id] = {
                "label": label,
                "mean": tensor_to_json_serializable(mean),
                "cov": tensor_to_json_serializable(cov),
                "num_points": len(points),
            }
        
        # Save Gaussian parameters
        gaussian_dir = output_path / "fitted_3D_gaussian"
        gaussian_dir.mkdir(exist_ok=True)
        
        output_data = {
            "image_info": {"resolution": [width, height]},
            "camera_info": {
                "intrinsic": tensor_to_json_serializable(intrinsic_tensor),
                "extrinsic": tensor_to_json_serializable(extrinsic)
            },
            "gaussian_params": gaussian_params,
            "num_objects": len(gaussian_params),
        }
        
        with open(gaussian_dir / "gaussian_params.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save input image copy
        Image.fromarray(image).save(str(output_path / "input_image.png"))
        
        update_task(task_id, 
                   status=TaskStatus.COMPLETED, 
                   progress=1.0, 
                   message="Preprocessing complete!",
                   result={
                       "output_dir": str(output_path),
                       "depth_npz": str(depth_dir / "depth_intrinsics.npz"),
                       "masks_dir": str(masks_dir),
                       "gaussian_json": str(gaussian_dir / "gaussian_params.json"),
                       "num_objects": len(gaussian_params)
                   },
                   completed_at=time.time())
        
        # Note: Model loading now starts at server startup (async), not here.
        # This allows model to load in parallel with preprocessing from the beginning.
        logger.info("Preprocessing complete!")
        
    except Exception as e:
        import traceback
        update_task(task_id, status=TaskStatus.FAILED, error=f"{str(e)}\n{traceback.format_exc()}")


def run_render(task_id: str, params: Dict[str, Any]):
    """Run 4D control map rendering (Step 5)."""
    try:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.0, message="Checking input files...")
        
        import subprocess
        
        logger.info(f"[Render Task {task_id}] Starting with params: {params}")
        
        # Validate required files exist
        required_files = [
            ('image_path', params.get('image_path')),
            ('depth_npz_path', params.get('depth_npz_path')),
            ('camera_trajectory_path', params.get('camera_trajectory_path')),
            ('gaussian_trajectory_path', params.get('gaussian_trajectory_path')),
        ]
        
        for name, path in required_files:
            if path and not os.path.exists(path):
                error_msg = f"Required file not found: {name} = {path}"
                logger.error(f"[Render Task {task_id}] {error_msg}")
                update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
                return
        
        # Check masks directory
        masks_dir = params.get('masks_dir')
        if masks_dir and not os.path.exists(masks_dir):
            error_msg = f"Masks directory not found: {masks_dir}"
            logger.error(f"[Render Task {task_id}] {error_msg}")
            update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
            return
        
        update_task(task_id, progress=0.05, message="Starting rendering...")
        
        # Build command using appropriate Python environment
        python_cmd = get_python_cmd()
        cmd = python_cmd + [
            os.path.join(INFERENCE_DIR, "rendering_4D_control_maps.py"),
            "--png_path", params['image_path'],
            "--npz_path", params['depth_npz_path'],
            "--mask_dir", params['masks_dir'],
            "--trajectory_npz", params['camera_trajectory_path'],
            "--ellipsoid_json", params['gaussian_trajectory_path'],
            "--output_dir", params['output_dir'],
        ]
        
        if 'fps' in params:
            cmd.extend(["--fps", str(params['fps'])])
        
        # Note: video_length is NOT passed to rendering script - it's determined by trajectory file
        # The trajectory file (custom_camera_trajectory.npz) contains the extrinsics for all frames
        
        logger.info(f"[Render Task {task_id}] Running command: {' '.join(cmd)}")
        logger.info(f"[Render Task {task_id}] Working directory: {PROJECT_ROOT}")
        
        # Set environment variables to fix MKL threading issue
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'  # Fix MKL + libgomp compatibility
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        # IMPORTANT: Use GPU 0 for rendering to avoid conflict with model server
        # Model server uses GPUs 1-7, rendering uses GPU 0
        env['CUDA_VISIBLE_DEVICES'] = '0'
        logger.info(f"[Render Task {task_id}] Using GPU 0 for rendering (CUDA_VISIBLE_DEVICES=0)")
        
        # Run rendering with real-time output logging
        # Important: Set cwd to PROJECT_ROOT so relative imports work correctly
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            cwd=PROJECT_ROOT,  # Set working directory to project root
            env=env  # Use modified environment
        )
        
        # Read output line by line for progress tracking
        output_lines = []
        frame_count = params.get('video_length', 49)
        
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            logger.info(f"[Render Task {task_id}] {line}")
            
            # Try to parse progress from output
            if 'frame' in line.lower() or 'rendering' in line.lower():
                # Try to extract frame number for progress estimation
                import re
                match = re.search(r'(\d+)\s*/\s*(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    progress = current / total
                    update_task(task_id, progress=progress, message=f"Rendering frame {current}/{total}")
                else:
                    # Generic progress update
                    update_task(task_id, message=f"Rendering: {line[:50]}...")
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = '\n'.join(output_lines[-20:])  # Last 20 lines
            logger.error(f"[Render Task {task_id}] Failed with code {process.returncode}: {error_msg}")
            update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
            return
        
        logger.info(f"[Render Task {task_id}] Completed successfully")
        update_task(task_id,
                   status=TaskStatus.COMPLETED,
                   progress=1.0,
                   message="Rendering complete!",
                   result={"output_dir": params['output_dir']},
                   completed_at=time.time())
        
        # For single GPU mode: Start loading model after render completes
        # (GPU is now free since render finished)
        num_gpus = _startup_config.get('num_gpus', 1)
        if num_gpus == 1 and not _model_loaded and not _model_loading:
            logger.info("=" * 60)
            logger.info("[Single GPU] Render complete. Starting model loading...")
            logger.info("=" * 60)
            start_model_loading_async()
        
    except Exception as e:
        import traceback
        update_task(task_id, status=TaskStatus.FAILED, error=f"{str(e)}\n{traceback.format_exc()}")


# ============================================================================
# Model Loading and In-Process Generation (for preload mode)
# ============================================================================

def load_model(
    model_name: str = "model/Wan2.1-T2V-14B",
    transformer_path: str = "model/VerseCrafter",
    config_path: str = "config/wan2.1/wan_civitai.yaml",
    num_gpus: int = 1,
    gpu_memory_mode: str = "model_full_load",
    enable_teacache: bool = True,
    teacache_threshold: float = 0.10,
    num_skip_start_steps: int = 5,
):
    """Load the VerseCrafter model into memory for fast repeated generation.
    
    Args:
        model_name: Path to the base Wan model
        transformer_path: Path to VerseCrafter transformer weights
        config_path: Path to config file
        num_gpus: Number of GPUs to use (1 for single GPU, >1 for multi-GPU)
        gpu_memory_mode: Memory optimization mode
        enable_teacache: Whether to enable TeaCache acceleration
        teacache_threshold: TeaCache threshold (higher = faster but less accurate)
        num_skip_start_steps: Number of initial steps to skip TeaCache
    """
    global _pipeline, _vae, _device, _weight_dtype, _model_config, _model_loaded
    
    if _model_loaded:
        logger.info("Model already loaded, skipping...")
        return True
    
    with _loading_lock:
        if _model_loaded:
            return True
        
        try:
            logger.info("=" * 60)
            logger.info("Loading VerseCrafter model...")
            logger.info(f"  num_gpus: {num_gpus}")
            logger.info(f"  gpu_memory_mode: {gpu_memory_mode}")
            logger.info("=" * 60)
            
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            import torch
            import numpy as np
            from diffusers import FlowMatchEulerDiscreteScheduler
            from omegaconf import OmegaConf
            from transformers import AutoTokenizer
            
            from videox_fun.dist import set_multi_gpus_devices, shard_model
            from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel
            from videox_fun.utils.fp8_optimization import (
                convert_model_weight_to_float8,
                convert_weight_dtype_wrapper,
                replace_parameters_by_name
            )
            from videox_fun.utils.utils import filter_kwargs
            from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
            from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            from versecrafter.models import VerseCrafterWanTransformer3DModel
            from versecrafter.pipeline import WanVerseCrafterPipeline
            
            # Calculate ulysses_degree and ring_degree from num_gpus
            # IMPORTANT: ulysses_degree × ring_degree MUST equal num_gpus
            if num_gpus == 1:
                ulysses_degree = 1
                ring_degree = 1
            elif num_gpus == 2:
                ulysses_degree = 1
                ring_degree = 2
            elif num_gpus == 3:
                ulysses_degree = 1
                ring_degree = 3
            elif num_gpus == 4:
                ulysses_degree = 2
                ring_degree = 2
            elif num_gpus == 5:
                ulysses_degree = 1
                ring_degree = 5
            elif num_gpus == 6:
                ulysses_degree = 2
                ring_degree = 3
            elif num_gpus == 7:
                ulysses_degree = 1
                ring_degree = 7
            elif num_gpus == 8:
                ulysses_degree = 2
                ring_degree = 4
            else:
                # Find optimal factorization: ulysses × ring = num_gpus
                import math
                sqrt_n = int(math.sqrt(num_gpus))
                for i in range(sqrt_n, 0, -1):
                    if num_gpus % i == 0:
                        ulysses_degree = i
                        ring_degree = num_gpus // i
                        break
                else:
                    ulysses_degree = 1
                    ring_degree = num_gpus
            
            logger.info(f"  ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}")
            
            # Configuration
            _weight_dtype = torch.bfloat16
            geoada_context_scale = 1.00
            geoada_in_dim = 128
            sampler_name = "Flow_Unipc"
            shift = 16
            fsdp_dit = False
            fsdp_text_encoder = True
            compile_dit = False
            teacache_offload = False
            cfg_skip_ratio = 0
            
            # Set up device for multi-GPU
            _device = set_multi_gpus_devices(ulysses_degree, ring_degree)
            logger.info(f"Using device: {_device}")
            
            # Load config
            config = OmegaConf.load(config_path)
            
            transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
            if geoada_in_dim is not None:
                transformer_additional_kwargs['geoada_in_dim'] = geoada_in_dim
            
            # Load transformer
            logger.info(f"Loading transformer from: {transformer_path}")
            if os.path.isdir(transformer_path):
                transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
                    transformer_path,
                    transformer_additional_kwargs=transformer_additional_kwargs,
                    low_cpu_mem_usage=True,
                    torch_dtype=_weight_dtype,
                )
            else:
                transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
                    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                    transformer_additional_kwargs=transformer_additional_kwargs,
                    low_cpu_mem_usage=True,
                    torch_dtype=_weight_dtype,
                )
                if transformer_path is not None:
                    logger.info(f"Loading weights from checkpoint: {transformer_path}")
                    if transformer_path.endswith("safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(transformer_path)
                    else:
                        state_dict = torch.load(transformer_path, map_location="cpu")
                    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                    m, u = transformer.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded weights: missing={len(m)}, unexpected={len(u)}")
            
            # Load VAE
            logger.info("Loading VAE...")
            _vae = AutoencoderKLWan.from_pretrained(
                os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            ).to(_weight_dtype)
            
            # Load Tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )
            
            # Load Text encoder
            logger.info("Loading text encoder...")
            text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=_weight_dtype,
            )
            text_encoder = text_encoder.eval()
            
            # Get Scheduler
            logger.info("Loading scheduler...")
            scheduler_dict = {
                "Flow": FlowMatchEulerDiscreteScheduler,
                "Flow_Unipc": FlowUniPCMultistepScheduler,
                "Flow_DPM++": FlowDPMSolverMultistepScheduler,
            }
            Chosen_Scheduler = scheduler_dict[sampler_name]
            if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
                config['scheduler_kwargs']['shift'] = 1
            scheduler = Chosen_Scheduler(
                **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
            
            # Create Pipeline
            logger.info("Creating pipeline...")
            _pipeline = WanVerseCrafterPipeline(
                transformer=transformer,
                vae=_vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
            )
            
            # Multi-GPU setup
            if ulysses_degree > 1 or ring_degree > 1:
                from functools import partial
                transformer.enable_multi_gpus_inference()
                if fsdp_dit:
                    shard_fn = partial(shard_model, device_id=_device, param_dtype=_weight_dtype)
                    _pipeline.transformer = shard_fn(_pipeline.transformer)
                    logger.info("Enabled FSDP for DIT")
                if fsdp_text_encoder:
                    shard_fn = partial(shard_model, device_id=_device, param_dtype=_weight_dtype)
                    _pipeline.text_encoder = shard_fn(_pipeline.text_encoder)
                    logger.info("Enabled FSDP for text encoder")
            
            if compile_dit:
                for i in range(len(_pipeline.transformer.blocks)):
                    _pipeline.transformer.blocks[i] = torch.compile(_pipeline.transformer.blocks[i])
                logger.info("Enabled torch.compile")
            
            # GPU memory mode
            if gpu_memory_mode == "sequential_cpu_offload":
                replace_parameters_by_name(transformer, ["modulation",], device=_device)
                transformer.freqs = transformer.freqs.to(device=_device)
                _pipeline.enable_sequential_cpu_offload(device=_device)
            elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
                convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=_device)
                convert_weight_dtype_wrapper(transformer, _weight_dtype)
                _pipeline.enable_model_cpu_offload(device=_device)
            elif gpu_memory_mode == "model_cpu_offload":
                _pipeline.enable_model_cpu_offload(device=_device)
            elif gpu_memory_mode == "model_full_load_and_qfloat8":
                convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=_device)
                convert_weight_dtype_wrapper(transformer, _weight_dtype)
                _pipeline.to(device=_device)
            else:
                _pipeline.to(device=_device)
            
            # Store config for later use
            _model_config = {
                'shift': shift,
                'geoada_context_scale': geoada_context_scale,
                'enable_teacache': enable_teacache,
                'teacache_threshold': teacache_threshold,
                'num_skip_start_steps': num_skip_start_steps,
                'teacache_offload': teacache_offload,
                'cfg_skip_ratio': cfg_skip_ratio,
                'ulysses_degree': ulysses_degree,
                'ring_degree': ring_degree,
                'num_gpus': num_gpus,
            }
            
            _model_loaded = True
            logger.info("=" * 60)
            logger.info("Model loaded successfully!")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False


def _async_load_model_worker():
    """Worker function for background model loading."""
    global _model_loading, _model_loaded, _model_load_error
    
    try:
        logger.info("=" * 60)
        logger.info("Background model loading started...")
        logger.info("=" * 60)
        
        success = load_model(
            model_name=_startup_config['base_model_path'],
            transformer_path=_startup_config['model_path'],
            config_path=_startup_config['config_path'],
            num_gpus=_startup_config['num_gpus'],
            gpu_memory_mode=_startup_config['gpu_memory_mode'],
        )
        
        if success:
            logger.info("=" * 60)
            logger.info("Background model loading completed successfully!")
            logger.info("=" * 60)
        else:
            _model_load_error = "Model loading returned False"
            logger.error("=" * 60)
            logger.error("Background model loading FAILED!")
            logger.error("Video generation will fall back to subprocess mode (slower)")
            logger.error("=" * 60)
            
    except Exception as e:
        import traceback
        _model_load_error = f"{str(e)}\n{traceback.format_exc()}"
        logger.error("=" * 60)
        logger.error(f"Background model loading FAILED with exception: {e}")
        logger.error("Video generation will fall back to subprocess mode (slower)")
        logger.error("=" * 60)
    finally:
        _model_loading = False


def start_model_loading_async():
    """Start model loading in background (non-blocking).
    
    This is called automatically after preprocessing completes.
    The model loading happens in parallel with user's trajectory editing.
    
    For single GPU: Loads model directly in a background thread.
    For multi-GPU: Starts a model server via torchrun.
    """
    global _model_loading, _model_load_thread
    
    with _loading_lock:
        # Skip if already loaded or loading
        if _model_loaded:
            logger.info("Model already loaded, skipping async load")
            return
        if _model_loading:
            logger.info("Model already loading in background, skipping")
            return
        if _model_server_url is not None:
            logger.info("Model server already started, skipping")
            return
        
        num_gpus = _startup_config.get('num_gpus', 1)
        
        if num_gpus > 1:
            # Multi-GPU: Start model server via torchrun
            logger.info(f"Multi-GPU mode (num_gpus={num_gpus}), starting model server...")
            start_model_server_async()
        else:
            # Single GPU: Load in background thread
            _model_loading = True
            _model_load_thread = threading.Thread(target=_async_load_model_worker, daemon=True)
            _model_load_thread.start()
            logger.info("Started background model loading thread (single GPU mode)")


def wait_for_model_loaded(timeout: float = None) -> bool:
    """Wait for background model loading to complete.
    
    Args:
        timeout: Maximum seconds to wait. None means wait forever.
        
    Returns:
        True if model is loaded successfully, False otherwise.
    """
    global _model_load_thread
    
    if _model_loaded:
        return True
    
    if _model_load_thread is not None and _model_load_thread.is_alive():
        logger.info(f"Waiting for background model loading to complete (timeout={timeout})...")
        _model_load_thread.join(timeout)
        
        if _model_load_thread.is_alive():
            logger.warning("Timeout waiting for model loading")
            return False
    
    if _model_loaded:
        logger.info("Model loading completed, ready for generation")
        return True
    else:
        if _model_load_error:
            logger.error(f"Model loading failed: {_model_load_error}")
        return False


def get_model_loading_status() -> dict:
    """Get current model loading status."""
    return {
        "loaded": _model_loaded,
        "loading": _model_loading,
        "error": _model_load_error,
        "model_server_url": _model_server_url,
    }


def unload_model():
    """Unload the model from GPU memory (for single GPU mode).
    
    This frees GPU memory so other tasks (preprocessing, rendering) can use it.
    """
    global _pipeline, _vae, _device, _weight_dtype, _model_config, _model_loaded
    
    if not _model_loaded:
        logger.info("Model not loaded, nothing to unload")
        return
    
    with _loading_lock:
        if not _model_loaded:
            return
        
        try:
            logger.info("=" * 60)
            logger.info("Unloading model to free GPU memory...")
            logger.info("=" * 60)
            
            import torch
            import gc
            
            # Delete pipeline and VAE
            if _pipeline is not None:
                del _pipeline
                _pipeline = None
            
            if _vae is not None:
                del _vae
                _vae = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            _model_loaded = False
            _model_config.clear()
            
            logger.info("Model unloaded successfully, GPU memory freed")
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Model Server Management (for multi-GPU mode via torchrun)
# ============================================================================

def start_model_server_async():
    """Start the model server via torchrun in background (for multi-GPU).
    
    This launches model_server.py with torchrun, which supports multi-GPU.
    The model server provides an HTTP API for video generation.
    
    GPU Allocation:
    - GPU 0: Reserved for preprocessing and rendering
    - GPUs 1-N: Used by model server for video generation
    """
    global _model_server_url, _model_server_process
    
    if _model_server_url is not None:
        logger.info("Model server already started, skipping")
        return
    
    num_gpus = _startup_config.get('num_gpus', 1)
    if num_gpus <= 1:
        logger.info("Single GPU mode, no need for model server")
        return
    
    import subprocess
    
    # Reserve GPU 0 for rendering, use remaining GPUs for model server
    # e.g., num_gpus=8 → model_gpus=7 (GPUs 1-7), GPU 0 for rendering
    model_gpus = num_gpus - 1
    if model_gpus < 1:
        model_gpus = 1
    
    # Create CUDA_VISIBLE_DEVICES string: "1,2,3,4,5,6,7" (excluding GPU 0)
    gpu_list = ','.join(str(i) for i in range(1, num_gpus))
    
    logger.info("=" * 60)
    logger.info(f"Starting model server with {model_gpus} GPUs via torchrun...")
    logger.info(f"  GPU allocation: GPU 0 reserved for rendering")
    logger.info(f"  Model server GPUs: {gpu_list}")
    logger.info("=" * 60)
    
    # Build torchrun command
    model_server_script = os.path.join(PROJECT_ROOT, 'model_server.py')
    
    cmd = [
        'torchrun',
        f'--nproc_per_node={model_gpus}',
        model_server_script,
        '--port', str(_model_server_port),
        '--model_path', _startup_config['model_path'],
        '--base_model_path', _startup_config['base_model_path'],
        '--config_path', _startup_config['config_path'],
        '--gpu_memory_mode', _startup_config['gpu_memory_mode'],
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Set environment variables
    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'
    env['CUDA_VISIBLE_DEVICES'] = gpu_list  # Exclude GPU 0
    
    try:
        # Start the model server process
        _model_server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
            env=env
        )
        
        # Start a thread to log the output
        def log_output():
            for line in _model_server_process.stdout:
                logger.info(f"[ModelServer] {line.strip()}")
        
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        _model_server_url = f"http://127.0.0.1:{_model_server_port}"
        logger.info(f"Model server starting at {_model_server_url}")
        
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        import traceback
        traceback.print_exc()


def wait_for_model_server_ready(timeout: float = 600) -> bool:
    """Wait for the model server to be ready.
    
    Args:
        timeout: Maximum seconds to wait (default 10 minutes for model loading)
        
    Returns:
        True if server is ready, False otherwise
    """
    import urllib.request
    import urllib.error
    
    if _model_server_url is None:
        return False
    
    health_url = f"{_model_server_url}/health"
    start_time = time.time()
    
    logger.info(f"Waiting for model server to be ready (timeout={timeout}s)...")
    
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('model_loaded', False):
                    logger.info("Model server is ready!")
                    return True
                else:
                    logger.info("Model server responding but model not yet loaded...")
        except urllib.error.URLError:
            pass  # Server not ready yet
        except Exception as e:
            logger.warning(f"Health check error: {e}")
        
        time.sleep(5)  # Check every 5 seconds
    
    logger.error("Timeout waiting for model server")
    return False


def run_generate_via_model_server(task_id: str, params: Dict[str, Any]):
    """Run video generation via the model server (for multi-GPU)."""
    import urllib.request
    import urllib.error
    
    try:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.0, 
                   message="Connecting to model server...")
        
        # Wait for model server to be ready
        if not wait_for_model_server_ready(timeout=600):
            raise Exception("Model server not ready after timeout")
        
        # Send generation request
        generate_url = f"{_model_server_url}/generate"
        
        request_data = json.dumps(params).encode('utf-8')
        req = urllib.request.Request(
            generate_url,
            data=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        if 'task_id' not in result:
            raise Exception(f"Invalid response: {result}")
        
        server_task_id = result['task_id']
        logger.info(f"[Generate Task {task_id}] Model server task started: {server_task_id}")
        
        # Poll for completion
        status_url = f"{_model_server_url}/status/{server_task_id}"
        
        while True:
            time.sleep(2)
            
            try:
                with urllib.request.urlopen(status_url, timeout=10) as response:
                    status = json.loads(response.read().decode('utf-8'))
                
                server_status = status.get('status', '')
                progress = status.get('progress', 0)
                message = status.get('message', '')
                
                update_task(task_id, progress=progress, message=message)
                
                if server_status == 'completed':
                    result = status.get('result', {})
                    update_task(task_id,
                               status=TaskStatus.COMPLETED,
                               progress=1.0,
                               message="Video generation complete!",
                               result=result,
                               completed_at=time.time())
                    logger.info(f"[Generate Task {task_id}] Completed via model server")
                    return
                    
                elif server_status == 'failed':
                    error = status.get('error', 'Unknown error')
                    update_task(task_id, status=TaskStatus.FAILED, error=error)
                    return
                    
            except Exception as poll_error:
                logger.warning(f"Poll error (retrying): {poll_error}")
                continue
                
    except Exception as e:
        import traceback
        error_msg = f"Model server error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Generate Task {task_id}] {error_msg}")
        update_task(task_id, status=TaskStatus.FAILED, error=error_msg)


def run_generate_with_loaded_model(task_id: str, params: Dict[str, Any]):
    """Run video generation using the preloaded model (fast, no model reload)."""
    global _pipeline, _vae, _device, _weight_dtype, _model_config
    
    try:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.05, message="Preparing generation...")
        
        import torch
        import numpy as np
        from PIL import Image
        from videox_fun.utils.utils import (
            get_image_latent,
            get_video_to_video_latent,
            save_videos_grid
        )
        
        # Extract parameters
        prompt = params.get('prompt', '')
        image_path = params.get('image_path', '')
        rendering_maps_path = params.get('rendering_maps_path', '')
        output_dir = params.get('output_dir', 'outputs/generated')
        
        num_inference_steps = params.get('num_inference_steps', 50)
        guidance_scale = params.get('guidance_scale', 5.0)
        seed = params.get('seed', 2025)
        fps = params.get('fps', 16)
        sample_size_str = params.get('sample_size', '720,1280')
        if isinstance(sample_size_str, str):
            sample_size = [int(x) for x in sample_size_str.split(',')]
        else:
            sample_size = sample_size_str
        video_length = params.get('video_length', 81)
        
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality, JPEG "
            "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        )
        
        logger.info(f"[Task {task_id}] Starting generation with preloaded model...")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}, FPS: {fps}")
        
        # Setup TeaCache if enabled
        if _model_config.get('enable_teacache', True):
            coefficients = [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02]
            _pipeline.transformer.enable_teacache(
                coefficients, 
                num_inference_steps, 
                _model_config.get('teacache_threshold', 0.10),
                num_skip_start_steps=_model_config.get('num_skip_start_steps', 5),
                offload=_model_config.get('teacache_offload', False)
            )
        
        cfg_skip_ratio = _model_config.get('cfg_skip_ratio', 0)
        if cfg_skip_ratio:
            _pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        
        generator = torch.Generator(device=_device).manual_seed(seed)
        
        update_task(task_id, progress=0.1, message="Loading control videos...")
        
        with torch.no_grad():
            # Adjust video length
            video_length = (
                int((video_length - 1) // _vae.config.temporal_compression_ratio * 
                    _vae.config.temporal_compression_ratio) + 1 
                if video_length != 1 else 1
            )
            
            inpaint_video = None
            
            # Load control videos
            if os.path.isdir(rendering_maps_path):
                control_filenames = [
                    "background_RGB.mp4", 
                    "background_depth.mp4", 
                    "3D_gaussian_RGB.mp4", 
                    "3D_gaussian_depth.mp4"
                ]
                control_videos = []
                
                for i, control_filename in enumerate(control_filenames):
                    control_video_path_full = os.path.join(rendering_maps_path, control_filename)
                    if os.path.exists(control_video_path_full):
                        input_video, _, _, _ = get_video_to_video_latent(
                            control_video_path_full, 
                            video_length=video_length, 
                            sample_size=sample_size, 
                            fps=fps, 
                            ref_image=None
                        )
                        control_videos.append(input_video)
                    else:
                        logger.warning(f"Control video not found: {control_video_path_full}")
                        if len(control_videos) > 0:
                            control_videos.append(torch.zeros_like(control_videos[0]))
                    
                    update_task(task_id, progress=0.1 + 0.1 * (i + 1) / len(control_filenames), 
                               message=f"Loaded {control_filename}")
                
                # Load mask
                mask_path = os.path.join(rendering_maps_path, "merged_mask.mp4")
                if os.path.exists(mask_path):
                    input_video_mask, _, _, _ = get_video_to_video_latent(
                        mask_path, 
                        video_length=video_length, 
                        sample_size=sample_size, 
                        fps=fps, 
                        ref_image=None
                    )
                    input_video_mask = input_video_mask[:, :1]
                    input_video_mask[:, :, 0] = 0.0
                else:
                    input_video_mask = torch.ones_like(control_videos[0][:, :1]) * 255
                
                # Load input image
                img_latent = get_image_latent(image_path, sample_size=sample_size)
                
                # Set first frame of first control video to input image
                control_videos[0][:, :, 0] = img_latent.squeeze(2)
                
                control_video = control_videos
            else:
                raise ValueError(f"Rendering maps path not found: {rendering_maps_path}")
            
            update_task(task_id, progress=0.3, message="Running inference...")
            
            # Run pipeline
            sample = _pipeline(
                prompt, 
                num_frames=video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                video=inpaint_video,
                mask_video=input_video_mask,
                control_video=control_video,
                subject_ref_images=None,
                shift=_model_config.get('shift', 16),
                geoada_context_scale=_model_config.get('geoada_context_scale', 1.0),
            ).videos
        
        update_task(task_id, progress=0.9, message="Saving video...")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        index = len([p for p in os.listdir(output_dir) if p.startswith("generated_video_")])
        
        if video_length == 1:
            video_path = os.path.join(output_dir, f"generated_video_{index}.png")
            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(output_dir, f"generated_video_{index}.mp4")
            save_videos_grid(sample, video_path, fps=fps)
        
        logger.info(f"[Task {task_id}] Video saved to: {video_path}")
        
        # Handle multi-GPU case - only rank 0 should report success
        ulysses_degree = _model_config.get('ulysses_degree', 1)
        ring_degree = _model_config.get('ring_degree', 1)
        if ulysses_degree * ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() != 0:
                return  # Only rank 0 updates task status
        
        update_task(task_id, 
                    status=TaskStatus.COMPLETED, 
                    progress=1.0, 
                    message="Video generation complete!",
                    result={"output_dir": output_dir, "video_path": video_path},
                    completed_at=time.time())
        
        # For single GPU mode: Unload model after generation to free GPU memory
        # This allows other tasks (preprocessing, rendering) to use the GPU
        num_gpus = _model_config.get('num_gpus', 1)
        if num_gpus == 1:
            logger.info("=" * 60)
            logger.info("[Single GPU] Generation complete. Unloading model to free GPU...")
            logger.info("=" * 60)
            unload_model()
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Task {task_id}] Generation failed: {error_msg}")
        update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
        
        # For single GPU mode: Also unload on failure to free GPU
        num_gpus = _startup_config.get('num_gpus', 1)
        if num_gpus == 1 and _model_loaded:
            logger.info("[Single GPU] Generation failed. Unloading model to free GPU...")
            unload_model()


def run_generate_via_subprocess(task_id: str, params: Dict[str, Any]):
    """Run video generation via subprocess (model reloads each time)."""
    try:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.0, message="Starting video generation...")
        
        import subprocess
        
        logger.info(f"[Generate Task {task_id}] Starting with params: {params}")
        
        # Build torchrun command for multi-GPU
        num_gpus = params.get('num_gpus', 8)
        ulysses_degree = params.get('ulysses_degree', 2)
        ring_degree = params.get('ring_degree', 4)
        
        # If using conda environment, wrap torchrun with conda run
        if CONDA_ENV:
            cmd = [
                'conda', 'run', '-n', CONDA_ENV, '--no-capture-output',
                'torchrun',
            ]
        else:
            cmd = ['torchrun']
        
        cmd.extend([
            f"--nproc-per-node={num_gpus}",
            os.path.join(INFERENCE_DIR, "versecrafter_inference2.py"),
            "--transformer_path", params.get('model_path', 'model/VerseCrafter'),
            "--num_inference_steps", str(params.get('num_inference_steps', 50)),
            "--guidance_scale", str(params.get('guidance_scale', 5.0)),
            "--seed", str(params.get('seed', 2025)),
            "--fps", str(params.get('fps', 16)),
            "--sample_size", params.get('sample_size', '720,1280'),
            "--ulysses_degree", str(ulysses_degree),
            "--ring_degree", str(ring_degree),
            "--prompt", params['prompt'],
            "--input_image_path", params['image_path'],
            "--save_path", params['output_dir'],
            "--rendering_maps_path", params['rendering_maps_path'],
        ])
        
        logger.info(f"[Generate Task {task_id}] Running command: {' '.join(cmd)}")
        
        # Set environment variables to fix MKL threading issue
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'  # Fix MKL + libgomp compatibility
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        # Run generation with real-time logging
        # Important: Set cwd to PROJECT_ROOT so relative imports work correctly
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,  # Set working directory to project root
            env=env  # Use modified environment
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            logger.info(f"[Generate Task {task_id}] {line}")
            
            # Parse progress from output
            if 'step' in line.lower() or '%' in line:
                import re
                match = re.search(r'(\d+)\s*/\s*(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    progress = current / total
                    update_task(task_id, progress=progress, message=f"Generating: step {current}/{total}")
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = '\n'.join(output_lines[-20:])
            logger.error(f"[Generate Task {task_id}] Failed with code {process.returncode}: {error_msg}")
            update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
            return
        
        logger.info(f"[Generate Task {task_id}] Completed successfully")
        
        # Find the generated video file
        output_dir = params['output_dir']
        video_path = None
        if os.path.exists(output_dir):
            video_files = [f for f in os.listdir(output_dir) 
                          if f.startswith('generated_video_') and f.endswith('.mp4')]
            if video_files:
                # Get the most recently created video file
                video_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
                video_path = os.path.join(output_dir, video_files[0])
                logger.info(f"[Generate Task {task_id}] Found video: {video_path}")
        
        update_task(task_id,
                   status=TaskStatus.COMPLETED,
                   progress=1.0,
                   message="Video generation complete!",
                   result={"output_dir": output_dir, "video_path": video_path},
                   completed_at=time.time())
        
    except Exception as e:
        import traceback
        update_task(task_id, status=TaskStatus.FAILED, error=f"{str(e)}\n{traceback.format_exc()}")


def run_generate(task_id: str, params: Dict[str, Any]):
    """Run video generation (Step 6).
    
    Priority order:
    1. If model server is running (multi-GPU), use it
    2. If model is loaded in-process (single GPU), use it
    3. Otherwise, spawn subprocess (model reloads each time, slower)
    """
    # Option 1: Model server (multi-GPU mode)
    if _model_server_url is not None:
        logger.info(f"[Generate Task {task_id}] Using model server at {_model_server_url}")
        run_generate_via_model_server(task_id, params)
        return
    
    # Option 2: In-process model (single GPU, loaded in background)
    if _model_loading and not _model_loaded:
        update_task(task_id, status=TaskStatus.RUNNING, progress=0.0, 
                   message="Waiting for model to finish loading...")
        logger.info(f"[Generate Task {task_id}] Model is loading in background, waiting...")
        
        # Wait for background loading to complete
        wait_for_model_loaded(timeout=None)
        
        if _model_loaded:
            logger.info(f"[Generate Task {task_id}] Model loading completed, proceeding with generation")
        else:
            logger.warning(f"[Generate Task {task_id}] Model loading failed, falling back to subprocess")
            if _model_load_error:
                logger.error(f"[Generate Task {task_id}] Load error: {_model_load_error}")
    
    if _model_loaded:
        # Use preloaded model - fastest option
        logger.info(f"[Generate Task {task_id}] Using preloaded model for fast generation")
        run_generate_with_loaded_model(task_id, params)
        return
    
    # Option 3: Fall back to subprocess (model will reload, slower but works)
    logger.info(f"[Generate Task {task_id}] Model not loaded, using subprocess mode (slower)")
    run_generate_via_subprocess(task_id, params)


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    """Run preprocessing pipeline (Steps 1-3)."""
    logger.info("Received preprocess request")
    
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    
    if data is None:
        logger.error("No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400
    
    required = ['image_path', 'output_dir']
    for field in required:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate image path exists
    if not os.path.exists(data['image_path']):
        logger.error(f"Image not found: {data['image_path']}")
        return jsonify({"error": f"Image not found: {data['image_path']}"}), 400
    
    task = create_task("preprocess")
    logger.info(f"Created task: {task.id}")
    
    thread = threading.Thread(target=run_preprocess, args=(task.id, data))
    thread.start()
    
    response = {"task_id": task.id, "status": "started"}
    logger.info(f"Response: {response}")
    return jsonify(response)


@app.route('/api/render', methods=['POST'])
def api_render():
    """Run 4D control map rendering (Step 5)."""
    data = request.get_json()
    
    required = ['image_path', 'depth_npz_path', 'masks_dir', 
                'camera_trajectory_path', 'gaussian_trajectory_path', 'output_dir']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    task = create_task("render")
    thread = threading.Thread(target=run_render, args=(task.id, data))
    thread.start()
    
    return jsonify({"task_id": task.id, "status": "started"})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Run video generation (Step 6)."""
    data = request.get_json()
    
    required = ['prompt', 'image_path', 'output_dir', 'rendering_maps_path']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    task = create_task("generate")
    thread = threading.Thread(target=run_generate, args=(task.id, data))
    thread.start()
    
    return jsonify({"task_id": task.id, "status": "started"})


@app.route('/api/workflow', methods=['POST'])
def api_workflow():
    """Run full postprocess workflow (Steps 5-6)."""
    data = request.get_json()
    
    # This would chain render and generate tasks
    # For now, just run render first
    task = create_task("workflow")
    
    def run_workflow():
        try:
            # Step 5: Render
            update_task(task.id, message="Step 5: Rendering control maps...")
            run_render(task.id, data)
            
            # Check if render succeeded
            with task_lock:
                if tasks[task.id].status == TaskStatus.FAILED:
                    return
            
            # Step 6: Generate
            update_task(task.id, message="Step 6: Generating video...")
            data['rendering_maps_path'] = data['output_dir']
            run_generate(task.id, data)
            
        except Exception as e:
            update_task(task.id, status=TaskStatus.FAILED, error=str(e))
    
    thread = threading.Thread(target=run_workflow)
    thread.start()
    
    return jsonify({"task_id": task.id, "status": "started"})


@app.route('/api/status/<task_id>', methods=['GET'])
def api_status(task_id: str):
    """Get task status."""
    with task_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = tasks[task_id]
        return jsonify({
            "id": task.id,
            "type": task.type,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "result": task.result,
            "error": task.error,
        })


@app.route('/api/model_status', methods=['GET'])
def api_model_status():
    """Get video generation model loading status.
    
    Returns:
        - loaded: True if model is ready for generation
        - loading: True if model is currently being loaded in background
        - error: Error message if loading failed
    """
    return jsonify(get_model_loading_status())


@app.route('/api/download/<path:filepath>', methods=['GET'])
def api_download(filepath: str):
    """Download a file from the server."""
    # Security: only allow downloading from allowed directories
    allowed_prefixes = ['/tmp/', os.path.join(PROJECT_ROOT, 'demo_data')]
    
    if not any(filepath.startswith(prefix) for prefix in allowed_prefixes):
        return jsonify({"error": "Access denied"}), 403
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(filepath)


@app.route('/system_stats', methods=['GET'])
def system_stats():
    """Get system stats (ComfyUI compatibility)."""
    import torch
    
    stats = {
        "system": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    }
    
    if torch.cuda.is_available():
        stats["system"]["vram"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        stats["system"]["gpu_name"] = torch.cuda.get_device_name(0)
    
    return jsonify(stats)


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint."""
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"status": "ok", "server": "VerseCrafter API", "version": "1.0"})


@app.route('/test_proxy', methods=['GET'])
def test_proxy():
    """Simple test endpoint that returns plain text for debugging proxy issues."""
    # Return plain JSON string to make it easy to debug
    return '{"test": "ok", "message": "If you see this, the proxy is working!"}', 200, {'Content-Type': 'application/json'}


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with helpful info."""
    return jsonify({
        "name": "VerseCrafter API Server",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/test_proxy": "Proxy test",
            "/api/preprocess_get": "Run preprocessing (GET)",
            "/api/render_get": "Run rendering (GET)",
            "/api/generate_get": "Run video generation (GET)",
            "/api/status/<task_id>": "Check task status",
            "/api/list_files?path=<dir>": "List files in directory"
        }
    })


# ============================================================================
# File Upload/Download Endpoints
# ============================================================================

# Default upload folder is outputs directory (when target_dir not specified)
# Normally, Blender addon always specifies target_dir, so this is just a fallback
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'outputs')

def get_safe_filepath(target_dir: str, filename: str) -> tuple[str, str]:
    """
    Validate and return safe filepath to prevent path traversal attacks.
    
    Args:
        target_dir: The target directory (must be under PROJECT_ROOT)
        filename: The filename (will be sanitized)
    
    Returns:
        Tuple of (safe_filepath, safe_filename)
    
    Raises:
        ValueError: If the path is outside allowed directories
    """
    # Sanitize filename to remove path traversal characters
    safe_filename = secure_filename(filename)
    if not safe_filename:
        raise ValueError("Invalid filename")
    
    # Resolve target_dir to absolute path
    abs_target_dir = os.path.abspath(target_dir)
    abs_project_root = os.path.abspath(PROJECT_ROOT)
    
    # Ensure target_dir is under PROJECT_ROOT
    if not abs_target_dir.startswith(abs_project_root + os.sep) and abs_target_dir != abs_project_root:
        raise ValueError(f"Target directory must be under project root: {PROJECT_ROOT}")
    
    # Construct and validate final filepath
    filepath = os.path.join(abs_target_dir, safe_filename)
    abs_filepath = os.path.abspath(filepath)
    
    # Double-check the final path is still under PROJECT_ROOT
    if not abs_filepath.startswith(abs_project_root + os.sep):
        raise ValueError("Path traversal detected")
    
    return abs_filepath, safe_filename

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload a file to the server."""
    logger.info("Received file upload request")
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Get target directory from form data
    target_dir = request.form.get('target_dir', UPLOAD_FOLDER)
    
    # Validate and sanitize filepath to prevent path traversal
    try:
        filepath, safe_filename = get_safe_filepath(target_dir, file.filename)
    except ValueError as e:
        logger.warning(f"Path traversal attempt blocked: {e}")
        return jsonify({"error": str(e)}), 400
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    logger.info(f"File saved to: {filepath}")
    return jsonify({"success": True, "path": filepath, "filename": safe_filename})


@app.route('/api/upload_base64', methods=['GET', 'POST'])
def api_upload_base64():
    """Upload a file as base64 (works with GET for proxy compatibility)."""
    import base64
    
    if request.method == 'GET':
        # Get parameters from URL
        data_b64 = request.args.get('data')
        filename = request.args.get('filename', 'uploaded_file')
        target_dir = request.args.get('target_dir', UPLOAD_FOLDER)
    else:
        data = request.get_json()
        data_b64 = data.get('data')
        filename = data.get('filename', 'uploaded_file')
        target_dir = data.get('target_dir', UPLOAD_FOLDER)
    
    if not data_b64:
        return jsonify({"error": "No data provided"}), 400
    
    # Validate and sanitize filepath to prevent path traversal
    try:
        filepath, safe_filename = get_safe_filepath(target_dir, filename)
    except ValueError as e:
        logger.warning(f"Path traversal attempt blocked: {e}")
        return jsonify({"error": str(e)}), 400
    
    try:
        file_data = base64.b64decode(data_b64)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Base64 file saved to: {filepath}")
        return jsonify({"success": True, "path": filepath})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Storage for chunked uploads
_chunk_uploads = {}

@app.route('/api/upload_chunk', methods=['GET'])
def api_upload_chunk():
    """Upload a file chunk (for large files via GET requests)."""
    import base64
    
    upload_id = request.args.get('upload_id')
    filename = request.args.get('filename')
    target_dir = request.args.get('target_dir', UPLOAD_FOLDER)
    chunk_index = int(request.args.get('chunk_index', 0))
    total_chunks = int(request.args.get('total_chunks', 1))
    data_b64 = request.args.get('data')
    
    if not all([upload_id, filename, data_b64]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Validate and sanitize filepath to prevent path traversal
    try:
        safe_filepath, safe_filename = get_safe_filepath(target_dir, filename)
    except ValueError as e:
        logger.warning(f"Path traversal attempt blocked: {e}")
        return jsonify({"error": str(e)}), 400
    
    try:
        chunk_data = base64.b64decode(data_b64)
        
        # Initialize upload entry if needed
        if upload_id not in _chunk_uploads:
            _chunk_uploads[upload_id] = {
                'filename': safe_filename,
                'filepath': safe_filepath,
                'total_chunks': total_chunks,
                'chunks': {},
                'created_at': time.time()
            }
        
        # Store chunk
        _chunk_uploads[upload_id]['chunks'][chunk_index] = chunk_data
        
        logger.info(f"Received chunk {chunk_index + 1}/{total_chunks} for {safe_filename}")
        
        # Check if all chunks received
        if len(_chunk_uploads[upload_id]['chunks']) == total_chunks:
            # Assemble file using validated filepath
            filepath = _chunk_uploads[upload_id]['filepath']
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                for i in range(total_chunks):
                    f.write(_chunk_uploads[upload_id]['chunks'][i])
            
            # Cleanup
            del _chunk_uploads[upload_id]
            
            logger.info(f"Chunked upload complete: {filepath}")
            return jsonify({"success": True, "path": filepath, "complete": True})
        
        return jsonify({
            "success": True, 
            "complete": False,
            "received_chunks": len(_chunk_uploads[upload_id]['chunks']),
            "total_chunks": total_chunks
        })
        
    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download_file', methods=['GET'])
def api_download_file():
    """Download a file from the server."""
    filepath = request.args.get('path')
    
    if not filepath:
        return jsonify({"error": "No path provided"}), 400
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 404
    
    # Return file
    return send_file(filepath, as_attachment=True)


@app.route('/api/download_base64', methods=['GET'])
def api_download_base64():
    """Download a file as base64 (works through proxy)."""
    import base64
    
    filepath = request.args.get('path')
    
    if not filepath:
        return jsonify({"error": "No path provided"}), 400
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 404
    
    try:
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        data_b64 = base64.b64encode(file_data).decode('utf-8')
        filename = os.path.basename(filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "data": data_b64,
            "size": len(file_data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/list_files', methods=['GET'])
def api_list_files():
    """List files in a directory."""
    dirpath = request.args.get('path')
    
    if not dirpath:
        return jsonify({"error": "No path provided"}), 400
    
    if not os.path.exists(dirpath):
        return jsonify({"error": f"Directory not found: {dirpath}"}), 404
    
    files = []
    for item in os.listdir(dirpath):
        item_path = os.path.join(dirpath, item)
        files.append({
            "name": item,
            "path": item_path,
            "is_dir": os.path.isdir(item_path),
            "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0
        })
    
    return jsonify({"files": files, "path": dirpath})


# ============================================================================
# GET-based API Endpoints (for proxies that block POST)
# ============================================================================

@app.route('/api/preprocess_get', methods=['GET'])
def api_preprocess_get():
    """
    Run preprocessing via GET request.
    
    Usage:
    /api/preprocess_get?image_path=/path/to/image.png&output_dir=/path/to/output&text_prompt=person.car.
    """
    logger.info("Received GET preprocess request")
    
    image_path = request.args.get('image_path')
    output_dir = request.args.get('output_dir')
    text_prompt = request.args.get('text_prompt', 'person . car .')
    
    logger.info(f"Parameters: image_path={image_path}, output_dir={output_dir}, text_prompt={text_prompt}")
    
    if not image_path:
        return jsonify({"error": "Missing required parameter: image_path"}), 400
    if not output_dir:
        return jsonify({"error": "Missing required parameter: output_dir"}), 400
    
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 400
    
    data = {
        'image_path': image_path,
        'output_dir': output_dir,
        'text_prompt': text_prompt,
        'model_version': request.args.get('model_version', 'v2'),
        'use_fp16': request.args.get('use_fp16', 'true').lower() == 'true',
        'resolution_level': int(request.args.get('resolution_level', '9')),
        'box_threshold': float(request.args.get('box_threshold', '0.4')),
        'min_area_ratio': float(request.args.get('min_area_ratio', '0.003')),
        'max_area_ratio': float(request.args.get('max_area_ratio', '0.2')),
    }
    
    task = create_task("preprocess")
    logger.info(f"Created task: {task.id}")
    
    thread = threading.Thread(target=run_preprocess, args=(task.id, data))
    thread.start()
    
    return jsonify({"task_id": task.id, "status": "started", "message": "Preprocessing started. Check /api/status/<task_id> for progress."})


@app.route('/api/render_get', methods=['GET'])
def api_render_get():
    """
    Run 4D control map rendering via GET request.
    
    Usage (simple mode):
    /api/render_get?preprocess_dir=...&trajectory_dir=...&video_length=49
    
    Usage (full mode):
    /api/render_get?image_path=...&depth_npz_path=...&masks_dir=...&camera_trajectory_path=...&gaussian_trajectory_path=...&output_dir=...
    """
    logger.info("Received GET render request")
    
    # Try simple mode first
    preprocess_dir = request.args.get('preprocess_dir')
    trajectory_dir = request.args.get('trajectory_dir')
    video_length = int(request.args.get('video_length', '49'))
    
    if preprocess_dir and trajectory_dir:
        # Simple mode: derive paths from directories
        # Find input image
        image_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            candidates = [
                os.path.join(preprocess_dir, f"0001.{ext}"),
                os.path.join(preprocess_dir, f"input.{ext}"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    image_path = c
                    break
            if image_path:
                break
        
        if not image_path:
            # List files in preprocess_dir
            for f in os.listdir(preprocess_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.'):
                    image_path = os.path.join(preprocess_dir, f)
                    break
        
        if not image_path:
            return jsonify({"error": "Cannot find input image in preprocess_dir"}), 400
        
        data = {
            'image_path': image_path,
            'depth_npz_path': os.path.join(preprocess_dir, 'estimated_depth', 'depth_intrinsics.npz'),
            'masks_dir': os.path.join(preprocess_dir, 'object_mask', 'masks'),
            'camera_trajectory_path': os.path.join(trajectory_dir, 'custom_camera_trajectory.npz'),
            'gaussian_trajectory_path': os.path.join(trajectory_dir, 'custom_3D_gaussian_trajectory.json'),
            'output_dir': os.path.join(trajectory_dir, 'rendered_4D_control_maps'),
            'video_length': video_length,
            'fps': int(request.args.get('fps', '10'))
        }
        
        logger.info(f"Render data (simple mode): {data}")
    else:
        # Full mode: require all parameters
        required_params = ['image_path', 'depth_npz_path', 'masks_dir', 
                           'camera_trajectory_path', 'gaussian_trajectory_path', 'output_dir']
        
        data = {}
        for param in required_params:
            value = request.args.get(param)
            if not value:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400
            data[param] = value
        
        data['fps'] = int(request.args.get('fps', '10'))
        data['video_length'] = video_length
    
    task = create_task("render")
    logger.info(f"Created render task: {task.id}")
    
    thread = threading.Thread(target=run_render, args=(task.id, data))
    thread.start()
    
    return jsonify({"id": task.id, "status": "started"})


@app.route('/api/generate_get', methods=['GET'])
def api_generate_get():
    """
    Run video generation via GET request.
    
    Usage (simple mode):
    /api/generate_get?preprocess_dir=...&control_map_dir=...&video_prompt=...&video_length=49&output_dir=...
    
    Usage (full mode):
    /api/generate_get?prompt=...&image_path=...&output_dir=...&rendering_maps_path=...
    """
    logger.info("Received GET generate request")
    
    # Try simple mode first
    preprocess_dir = request.args.get('preprocess_dir')
    control_map_dir = request.args.get('control_map_dir')
    video_prompt = request.args.get('video_prompt')
    video_length = int(request.args.get('video_length', '49'))
    output_dir = request.args.get('output_dir')
    
    if preprocess_dir and control_map_dir:
        # Simple mode: derive paths from directories
        # Find input image
        image_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            candidates = [
                os.path.join(preprocess_dir, f"0001.{ext}"),
                os.path.join(preprocess_dir, f"input.{ext}"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    image_path = c
                    break
            if image_path:
                break
        
        if not image_path:
            for f in os.listdir(preprocess_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.'):
                    image_path = os.path.join(preprocess_dir, f)
                    break
        
        if not image_path:
            return jsonify({"error": "Cannot find input image in preprocess_dir"}), 400
        
        if not output_dir:
            output_dir = os.path.join(preprocess_dir, 'generated_videos')
        
        if not video_prompt:
            video_prompt = "A video of the scene with smooth motion."
        
        data = {
            'prompt': video_prompt,
            'image_path': image_path,
            'output_dir': output_dir,
            'rendering_maps_path': control_map_dir,
            'video_length': video_length,
            'model_path': request.args.get('model_path', 'model/VerseCrafter'),
            'num_inference_steps': int(request.args.get('num_inference_steps', '50')),
            'guidance_scale': float(request.args.get('guidance_scale', '5.0')),
            'seed': int(request.args.get('seed', '2025')),
            'fps': int(request.args.get('fps', '16')),
            'sample_size': request.args.get('sample_size', '720,1280'),
            'num_gpus': int(request.args.get('num_gpus', '8')),
            'ulysses_degree': int(request.args.get('ulysses_degree', '2')),
            'ring_degree': int(request.args.get('ring_degree', '4'))
        }
        
        logger.info(f"Generate data (simple mode): {data}")
    else:
        # Full mode: require specific parameters
        required_params = ['prompt', 'image_path', 'output_dir', 'rendering_maps_path']
        
        data = {}
        for param in required_params:
            value = request.args.get(param)
            if not value:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400
            data[param] = value
        
        # Optional parameters
        data['model_path'] = request.args.get('model_path', 'model/VerseCrafter')
        data['num_inference_steps'] = int(request.args.get('num_inference_steps', '50'))
        data['sample_size'] = request.args.get('sample_size', '720,1280')
        data['num_gpus'] = int(request.args.get('num_gpus', '8'))
        data['ulysses_degree'] = int(request.args.get('ulysses_degree', '2'))
        data['ring_degree'] = int(request.args.get('ring_degree', '4'))
        data['video_length'] = video_length
    
    task = create_task("generate")
    logger.info(f"Created generate task: {task.id}")
    
    thread = threading.Thread(target=run_generate, args=(task.id, data))
    thread.start()
    
    return jsonify({"id": task.id, "status": "started"})


@app.route('/api/test', methods=['POST'])
def api_test():
    """Simple test endpoint to verify POST requests work."""
    logger.info("Test endpoint called")
    try:
        data = request.get_json()
        logger.info(f"Test data: {data}")
        return jsonify({"received": data, "status": "ok"})
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/preprocess_sync', methods=['POST'])
def api_preprocess_sync():
    """
    Synchronous preprocess endpoint - runs Steps 1-3 and returns when complete.
    Use this for simpler integration.
    """
    logger.info("Received sync preprocess request")
    
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400
    
    required = ['image_path', 'output_dir']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate image path exists
    if not os.path.exists(data['image_path']):
        return jsonify({"error": f"Image not found: {data['image_path']}"}), 400
    
    # Run synchronously
    task = create_task("preprocess_sync")
    
    try:
        run_preprocess(task.id, data)
        
        with task_lock:
            result_task = tasks[task.id]
        
        if result_task.status == TaskStatus.COMPLETED:
            return jsonify({
                "status": "completed",
                "result": result_task.result
            })
        else:
            return jsonify({
                "status": "failed",
                "error": result_task.error
            }), 500
            
    except Exception as e:
        import traceback
        logger.error(f"Sync preprocess error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VerseCrafter API Server')
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8188, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Model configuration
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use (1 for single GPU, 2/4/8 for multi-GPU)')
    parser.add_argument('--model_path', type=str, default='model/VerseCrafter',
                        help='Path to VerseCrafter model weights')
    parser.add_argument('--base_model_path', type=str, default='model/Wan2.1-T2V-14B',
                        help='Path to base Wan model')
    parser.add_argument('--config_path', type=str, default='config/wan2.1/wan_civitai.yaml',
                        help='Path to model config file')
    parser.add_argument('--gpu_memory_mode', type=str, default='model_full_load',
                        choices=['model_full_load', 'model_full_load_and_qfloat8', 
                                'model_cpu_offload', 'model_cpu_offload_and_qfloat8',
                                'sequential_cpu_offload'],
                        help='GPU memory mode for model loading')
    
    args = parser.parse_args()
    
    # Store startup config for async loading (used after preprocessing completes)
    _startup_config['num_gpus'] = args.num_gpus
    _startup_config['model_path'] = args.model_path
    _startup_config['base_model_path'] = args.base_model_path
    _startup_config['config_path'] = args.config_path
    _startup_config['gpu_memory_mode'] = args.gpu_memory_mode
    
    print("=" * 60)
    print("VerseCrafter API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model config:")
    print(f"  num_gpus: {args.num_gpus}")
    print(f"  model_path: {args.model_path}")
    print(f"  gpu_memory_mode: {args.gpu_memory_mode}")
    print("=" * 60)
    
    # Model loading strategy:
    # - Multi-GPU (num_gpus > 1): Start model server on GPUs 1-N immediately at startup
    # - Single GPU: Load model AFTER render completes, unload after generation
    # - GPU 0 is always reserved for preprocessing and rendering (multi-GPU mode)
    
    if args.num_gpus == 1:
        # Single GPU mode: DO NOT load model at startup
        # Model will be loaded after render completes, then unloaded after generation
        print("\n[Single GPU Mode]")
        print("  GPU Allocation:")
        print("    - GPU 0: Shared for all tasks (sequential)")
        print("  Workflow:")
        print("    - Step 1-3: Preprocessing uses GPU 0")
        print("    - Step 4: User edits trajectories")
        print("    - Step 5: Render uses GPU 0")
        print("    - After Step 5: Model loads (GPU freed by render)")
        print("    - Step 6: Generate video")
        print("    - After Step 6: Model unloads (GPU freed for next task)")
    else:
        # Multi-GPU mode: Start model server immediately at startup
        model_gpus = args.num_gpus - 1
        print(f"\n[Multi-GPU Mode ({args.num_gpus} GPUs)]")
        print(f"  GPU Allocation:")
        print(f"    - GPU 0: Reserved for preprocessing & rendering")
        print(f"    - GPUs 1-{args.num_gpus-1}: Model server ({model_gpus} GPUs)")
        print(f"  Workflow:")
        print("    - Server startup: Model server starts loading immediately (background)")
        print("    - Step 1-3: Preprocessing uses GPU 0 (parallel with model loading)")
        print("    - Step 4: User edits trajectories (model loading continues)")
        print("    - Step 5: Render uses GPU 0 (model should be ready by now)")
        print("    - Step 6: Model server generates video (fast, no reload)")
        
        # Start model loading immediately at startup (non-blocking)
        print("\n[Background Loading] Starting model loading now...")
        start_model_loading_async()
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

