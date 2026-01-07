"""
VerseCrafter Model Server

A standalone server that loads the VerseCrafter model via torchrun (supports multi-GPU).
The model stays in memory and provides an HTTP API for video generation.

Usage:
    # Single GPU
    python model_server.py --port 8189
    
    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 model_server.py --port 8189

The server exposes:
    POST /generate - Generate video with the loaded model
    GET /health - Check if model is loaded and ready
"""

import os
import sys
import json
import uuid
import time
import threading
import queue
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project paths BEFORE importing anything else
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(PROJECT_ROOT, 'inference')
VIDEOX_FUN_PATH = os.path.join(PROJECT_ROOT, 'third_party/VideoX-Fun')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, INFERENCE_DIR)
if VIDEOX_FUN_PATH not in sys.path:
    sys.path.insert(0, VIDEOX_FUN_PATH)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we're running under torchrun (distributed mode)
IS_DISTRIBUTED = 'RANK' in os.environ
RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

# Only rank 0 runs the Flask server
if RANK == 0:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    status: TaskStatus
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Global state
tasks: Dict[str, Task] = {}
task_lock = threading.Lock()

# Work queue for distributing tasks to all ranks
work_queue = queue.Queue()
result_dict: Dict[str, Any] = {}
result_lock = threading.Lock()

# Lock for broadcast operations (prevent heartbeat and generation from conflicting)
broadcast_lock = threading.Lock()

_pipeline = None
_vae = None
_device = None
_weight_dtype = None
_model_config = {}
_model_loaded = False


def create_task() -> Task:
    task = Task(
        id=str(uuid.uuid4()),
        status=TaskStatus.PENDING,
        progress=0.0,
        message="Task created"
    )
    with task_lock:
        tasks[task.id] = task
    return task


def update_task(task_id: str, **kwargs):
    with task_lock:
        if task_id in tasks:
            task = tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)


def load_model(args):
    """Load the VerseCrafter model."""
    global _pipeline, _vae, _device, _weight_dtype, _model_config, _model_loaded
    
    try:
        logger.info("=" * 60)
        logger.info("Loading VerseCrafter model...")
        logger.info(f"  Distributed: {IS_DISTRIBUTED}, Rank: {RANK}, World Size: {WORLD_SIZE}")
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
        
        # Calculate ulysses_degree and ring_degree
        # IMPORTANT: ulysses_degree × ring_degree MUST equal num_gpus
        num_gpus = WORLD_SIZE
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
            # Try to find factors closest to sqrt(num_gpus)
            sqrt_n = int(math.sqrt(num_gpus))
            for i in range(sqrt_n, 0, -1):
                if num_gpus % i == 0:
                    ulysses_degree = i
                    ring_degree = num_gpus // i
                    break
            else:
                # Fallback for prime numbers
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
        enable_teacache = True
        teacache_threshold = 0.10
        num_skip_start_steps = 5
        
        # Set up device
        _device = set_multi_gpus_devices(ulysses_degree, ring_degree)
        logger.info(f"Using device: {_device}")
        
        # Load config
        config = OmegaConf.load(args.config_path)
        
        transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        if geoada_in_dim is not None:
            transformer_additional_kwargs['geoada_in_dim'] = geoada_in_dim
        
        # Load transformer
        logger.info(f"Loading transformer from: {args.model_path}")
        if os.path.isdir(args.model_path):
            transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
                args.model_path,
                transformer_additional_kwargs=transformer_additional_kwargs,
                low_cpu_mem_usage=True,
                torch_dtype=_weight_dtype,
            )
        else:
            transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
                os.path.join(args.base_model_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                transformer_additional_kwargs=transformer_additional_kwargs,
                low_cpu_mem_usage=True,
                torch_dtype=_weight_dtype,
            )
            if args.model_path:
                logger.info(f"Loading weights from checkpoint: {args.model_path}")
                if args.model_path.endswith("safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(args.model_path)
                else:
                    state_dict = torch.load(args.model_path, map_location="cpu")
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                m, u = transformer.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights: missing={len(m)}, unexpected={len(u)}")
        
        # Load VAE
        logger.info("Loading VAE...")
        _vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.base_model_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(_weight_dtype)
        
        # Load Tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.base_model_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        
        # Load Text encoder
        logger.info("Loading text encoder...")
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.base_model_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
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
        
        # GPU memory mode
        gpu_memory_mode = args.gpu_memory_mode
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
        
        # Store config
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


def run_generation_all_ranks(params: Dict[str, Any], task_id: str = None):
    """Run video generation on ALL ranks simultaneously.
    
    This function must be called by all ranks at the same time for distributed inference.
    """
    global _pipeline, _vae, _device, _weight_dtype, _model_config
    
    import torch
    import numpy as np
    from PIL import Image
    from videox_fun.utils.utils import (
        get_image_latent,
        get_video_to_video_latent,
        save_videos_grid
    )
    
    try:
        if RANK == 0 and task_id:
            update_task(task_id, status=TaskStatus.RUNNING, progress=0.05, message="Preparing generation...")
        
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
        
        if RANK == 0:
            logger.info(f"[Task {task_id}] Starting generation on all {WORLD_SIZE} ranks...")
            logger.info(f"  Prompt: {prompt[:100]}...")
            logger.info(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # Setup TeaCache
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
        
        if RANK == 0 and task_id:
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
                        if RANK == 0:
                            logger.warning(f"Control video not found: {control_video_path_full}")
                        if len(control_videos) > 0:
                            control_videos.append(torch.zeros_like(control_videos[0]))
                    
                    if RANK == 0 and task_id:
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
                control_videos[0][:, :, 0] = img_latent.squeeze(2)
                
                control_video = control_videos
            else:
                raise ValueError(f"Rendering maps path not found: {rendering_maps_path}")
            
            if RANK == 0 and task_id:
                update_task(task_id, progress=0.3, message="Running inference...")
            
            # Run pipeline - ALL RANKS execute this together
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
        
        if RANK == 0 and task_id:
            update_task(task_id, progress=0.9, message="Saving video...")
        
        # Save results (only rank 0)
        video_path = None
        if RANK == 0:
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
            
            if task_id:
                update_task(task_id, 
                            status=TaskStatus.COMPLETED, 
                            progress=1.0, 
                            message="Video generation complete!",
                            result={"output_dir": output_dir, "video_path": video_path})
        
        return video_path
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[Rank {RANK}] Generation failed: {error_msg}")
        if RANK == 0 and task_id:
            update_task(task_id, status=TaskStatus.FAILED, error=error_msg)
        return None


def worker_loop():
    """Worker loop for non-rank-0 processes.
    
    Waits for work commands via distributed broadcast and executes them.
    """
    import torch
    import torch.distributed as dist
    
    logger.info(f"Rank {RANK} entering worker loop...")
    
    while True:
        # Wait for command from rank 0
        # Command format: [0] = command type, [1:] = data length
        cmd_tensor = torch.zeros(2, dtype=torch.long, device=_device)
        dist.broadcast(cmd_tensor, src=0)
        
        cmd_type = cmd_tensor[0].item()
        data_len = cmd_tensor[1].item()
        
        if cmd_type == 0:  # No-op / heartbeat
            continue
        elif cmd_type == 1:  # Generate
            # Receive params as JSON string
            if data_len > 0:
                data_tensor = torch.zeros(data_len, dtype=torch.uint8, device=_device)
                dist.broadcast(data_tensor, src=0)
                params_json = bytes(data_tensor.cpu().tolist()).decode('utf-8')
                params = json.loads(params_json)
            else:
                params = {}
            
            # Run generation (all ranks participate)
            run_generation_all_ranks(params, task_id=None)
            
        elif cmd_type == 99:  # Shutdown
            logger.info(f"Rank {RANK} received shutdown command")
            break


def broadcast_and_generate(params: Dict[str, Any], task_id: str):
    """Broadcast params to all ranks and run generation.
    
    Called by rank 0 to coordinate distributed generation.
    """
    import torch
    import torch.distributed as dist
    
    if WORLD_SIZE == 1:
        # Single GPU, just run directly
        run_generation_all_ranks(params, task_id)
        return
    
    # Use lock to prevent conflict with heartbeat broadcast
    with broadcast_lock:
        # Encode params as JSON
        params_json = json.dumps(params)
        params_bytes = params_json.encode('utf-8')
        
        # Broadcast command: type=1 (generate), length=len(params_bytes)
        cmd_tensor = torch.tensor([1, len(params_bytes)], dtype=torch.long, device=_device)
        dist.broadcast(cmd_tensor, src=0)
        
        # Broadcast params data
        data_tensor = torch.tensor(list(params_bytes), dtype=torch.uint8, device=_device)
        dist.broadcast(data_tensor, src=0)
        
        # Rank 0 also runs generation (still inside lock to ensure workers receive params first)
        run_generation_all_ranks(params, task_id)


# Flask routes (only on rank 0)
if RANK == 0:
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "ready" if _model_loaded else "loading",
            "model_loaded": _model_loaded,
            "distributed": IS_DISTRIBUTED,
            "world_size": WORLD_SIZE,
        })
    
    @app.route('/generate', methods=['POST'])
    def generate():
        """Generate video."""
        if not _model_loaded:
            return jsonify({"error": "Model not loaded yet"}), 503
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            task = create_task()
            
            # Run generation in background thread
            # This thread will broadcast to other ranks
            def run_in_thread():
                broadcast_and_generate(data, task.id)
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            
            return jsonify({
                "task_id": task.id,
                "status": "started"
            })
            
        except Exception as e:
            logger.error(f"Generate error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/status/<task_id>', methods=['GET'])
    def status(task_id: str):
        """Get task status."""
        with task_lock:
            if task_id not in tasks:
                return jsonify({"error": "Task not found"}), 404
            
            task = tasks[task_id]
            return jsonify({
                "id": task.id,
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "result": task.result,
                "error": task.error,
            })


def send_heartbeat():
    """Send heartbeat (no-op command) to all worker ranks to prevent NCCL timeout.
    
    NCCL has a default timeout of 10 minutes. If workers are waiting for a broadcast
    and no command is sent within that time, they will timeout and crash.
    This function sends a no-op command every few minutes to keep the connection alive.
    """
    import torch
    import torch.distributed as dist
    
    if WORLD_SIZE == 1:
        return  # No workers to keep alive
    
    # Use lock to prevent conflict with generation broadcast
    with broadcast_lock:
        # Send no-op command (type=0)
        cmd_tensor = torch.tensor([0, 0], dtype=torch.long, device=_device)
        dist.broadcast(cmd_tensor, src=0)
        logger.debug("Sent heartbeat to worker ranks")


def heartbeat_loop():
    """Background thread that sends periodic heartbeats to prevent NCCL timeout."""
    import time
    
    # NCCL default timeout is 10 minutes (600s)
    # Send heartbeat every 5 minutes to be safe
    HEARTBEAT_INTERVAL = 300  # 5 minutes in seconds
    
    logger.info(f"Heartbeat thread started (interval: {HEARTBEAT_INTERVAL}s)")
    
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        try:
            send_heartbeat()
            logger.info("Heartbeat sent to worker ranks")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            break


def main():
    parser = argparse.ArgumentParser(description='VerseCrafter Model Server')
    parser.add_argument('--port', type=int, default=8189, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--model_path', type=str, default='model/VerseCrafter',
                        help='Path to VerseCrafter model')
    parser.add_argument('--base_model_path', type=str, default='model/Wan2.1-T2V-14B',
                        help='Path to base Wan model')
    parser.add_argument('--config_path', type=str, default='config/wan2.1/wan_civitai.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu_memory_mode', type=str, default='model_full_load',
                        choices=['model_full_load', 'model_full_load_and_qfloat8', 
                                'model_cpu_offload', 'model_cpu_offload_and_qfloat8',
                                'sequential_cpu_offload'],
                        help='GPU memory mode')
    
    args = parser.parse_args()
    
    if RANK == 0:
        print("=" * 60)
        print("VerseCrafter Model Server")
        print("=" * 60)
        print(f"Distributed: {IS_DISTRIBUTED}")
        print(f"Rank: {RANK}, World Size: {WORLD_SIZE}")
        print(f"Model path: {args.model_path}")
        print("=" * 60)
    
    # Load model on all ranks
    success = load_model(args)
    
    if not success:
        print(f"ERROR: Rank {RANK} failed to load model!")
        sys.exit(1)
    
    if RANK == 0:
        # Start heartbeat thread to prevent NCCL timeout on worker ranks
        if WORLD_SIZE > 1:
            heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            print("Heartbeat thread started for worker ranks")
        
        # Rank 0 runs the Flask server
        print(f"\nStarting server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True)
    else:
        # Other ranks enter worker loop
        worker_loop()


if __name__ == '__main__':
    main()
