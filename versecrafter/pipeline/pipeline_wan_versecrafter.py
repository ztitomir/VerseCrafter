import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
videox_fun_path = os.path.join(project_root, 'third_party/VideoX-Fun')
if videox_fun_path not in sys.path:
    sys.path.insert(0, videox_fun_path)

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import T5Tokenizer

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel)
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from versecrafter.models import VerseCrafterWanTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    """
    Resize mask to match latent spatial dimensions, with optional first-frame-only processing.
    
    Args:
        mask: Input mask tensor of shape (B, C, T, H, W)
        latent: Reference latent tensor to determine target size
        process_first_frame_only: If True, handle first frame separately from remaining frames
        
    Returns:
        Resized mask tensor matching latent spatial dimensions
    """
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanVerseCrafterPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: VerseCrafterWanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        """
        Initialize the pipeline with all necessary components for text-to-video generation.
        
        Args:
            tokenizer: Tokenizer for encoding text prompts
            text_encoder: T5 encoder for converting tokens to embeddings
            vae: Variational autoencoder for encoding/decoding video latents
            transformer: 3D transformer model for diffusion-based generation
            scheduler: Noise scheduler for the diffusion process
        """
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # Initialize video/image processors with VAE compression ratios
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.config.spatial_compression_ratio,
            do_normalize=False,
            do_binarize=False,
            do_convert_grayscale=True,
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 text embeddings from prompts.
        
        Args:
            prompt: Text prompt(s) to encode
            num_videos_per_prompt: Number of videos per prompt for batch generation
            max_sequence_length: Maximum token sequence length
            device: Device for tensors
            dtype: Data type for embeddings
            
        Returns:
            List of prompt embeddings, one per batch item, trimmed to actual sequence length
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Tokenize prompts with fixed length
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # Warn if prompt is truncated
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # Get actual sequence lengths from attention mask
        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        
        # Encode tokens to embeddings using T5 encoder
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # Return trimmed embeddings based on actual sequence lengths
        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, 
        latents=None, num_length_latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1
            if num_length_latents is None
            else num_length_latents
            ),
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def geoada_encode_multi_frames(self, multi_frames, ref_images, vae=None):
        """
        Encode multiple control video frames to latent space for GeoAda (geometric adaptation).
        
        Args:
            multi_frames: List of control video tensors to encode
            ref_images: List of reference images for spatial adaptation
            vae: VAE model for encoding (uses self.vae if None)
            
        Returns:
            List of encoded latent tensors with optional reference image padding
        """
        vae = self.vae if vae is None else vae
        weight_dtype = multi_frames[0].dtype
        
        if ref_images is None:
            ref_images = [None] * len(multi_frames[0])
        else:
            assert len(ref_images) == multi_frames[0].shape[0]

        # Encode each control video to latent space
        encoded_latents = []
        for frames in multi_frames:
            latents = vae.encode(frames)[0].mode()
            encoded_latents.append(latents)
        
        # Concatenate latents from different control videos along the batch dimension
        latents = []
        for items in zip(*encoded_latents):
            latents.append(torch.cat(items, dim=0))

        # Add reference image latents if provided, padded with zeros for other frames
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                ref_latent = vae.encode(refs)[0].mode()
                # Pad reference latent with zeros for non-reference control frames
                ref_latent = [torch.cat([u] + [torch.zeros_like(u)] * (len(multi_frames)-1), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def geoada_encode_masks(self, masks, ref_images=None, vae_stride=[4, 8, 8]):
        """
        Encode masks to latent space with VAE stride-aware reshaping.
        
        Args:
            masks: Input masks of shape (C, T, H, W)
            ref_images: Optional reference images for spatial adaptation
            vae_stride: VAE stride values [temporal_stride, height_stride, width_stride]
            
        Returns:
            List of encoded masks in latent space
        """
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            # Calculate new spatial dimensions after VAE stride compression
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # Reshape mask to separate patch dimensions (depth, height, 8, width, 8)
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            # Permute to group patch dimensions: (8, 8, depth, height, width)
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            # Flatten patch dimensions: (8*8, depth, height, width)
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # Interpolate to target latent dimensions
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            # Add reference image padding if provided
            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def geoada_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        """
        Encode control video and control image to latent space for guidance.
        
        Converts pixel-space control inputs to latent space to match the shape of the 
        latent tensors that will be processed by the diffusion model. Encoding is done
        in small batches to manage memory usage.
        
        Args:
            control: Control video tensor (optional)
            control_image: Control image tensor (optional)
            batch_size: Batch size (for future use)
            height, width: Output dimensions
            dtype: Data type for tensors
            device: Device to place tensors on
            generator: Random number generator
            do_classifier_free_guidance: Whether classifier-free guidance is used
            
        Returns:
            Tuple of (control_latents, control_image_latents)
        """
        # Encode control inputs to latent space. We process in small batches
        # before converting to target dtype to avoid issues with CPU offloading
        # and mixed precision operations.

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            # Process control in small batches
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                # Encode to latent space using VAE
                control_bs = self.vae.encode(control_bs)[0]
                # Use mode of the latent distribution
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            # Process control image in small batches
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                # Encode to latent space using VAE
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                # Use mode of the latent distribution
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from 
    # diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.
    # StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            invalid_inputs = [
                k for k in callback_on_step_end_tensor_inputs 
                if k not in self._callback_tensor_inputs
            ]
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {invalid_inputs}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        control_video: Union[torch.FloatTensor, List[torch.FloatTensor]] = None,
        subject_ref_images: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        geoada_context_scale: float = 1.0
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt to text embeddings
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # Concatenate negative and positive embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, mu=1
            )
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        latent_channels = self.vae.config.latent_channels

        if comfyui_progressbar:
            pbar.update(1)

        # Prepare mask latent variables - preprocess mask video to match resolution
        if mask_video is not None:
            bs, _, video_length, height, width = mask_video.size()
            mask_condition = self.mask_processor.preprocess(
                rearrange(mask_video, "b c f h w -> (b f) c h w"),
                height=height,
                width=width
            )
            mask_condition = mask_condition.to(dtype=torch.float32)
            mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)
            mask_condition = torch.tile(mask_condition, [1, 3, 1, 1, 1]).to(dtype=weight_dtype, device=device)
        
        # Process control videos for spatial guidance
        if control_video is not None:
            video_length = control_video[0].shape[2]
            input_videos = []
            for cv in control_video:
                # Preprocess and convert to latent dtype
                cv_processed = self.image_processor.preprocess(
                    rearrange(cv, "b c f h w -> (b f) c h w"),
                    height=height,
                    width=width
                )
                cv_processed = cv_processed.to(dtype=torch.float32)
                cv_processed = rearrange(cv_processed, "(b f) c h w -> b c f h w", f=video_length)
                cv_processed = cv_processed.to(dtype=weight_dtype, device=device)
                input_videos.append(cv_processed)

        # Process reference video with mask guidance
        elif video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(
                rearrange(video, "b c f h w -> (b f) c h w"),
                height=height,
                width=width
            )
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(
                init_video, "(b f) c h w -> b c f h w", f=video_length
            ).to(dtype=weight_dtype, device=device)

            # Apply mask to zero out regions that should be generated
            input_video = init_video * (mask_condition < 0.5)
            input_video = input_video.to(dtype=weight_dtype, device=device)

        # Process subject reference images for GeoAda spatial control
        if subject_ref_images is not None:
            video_length = subject_ref_images.shape[2]
            subject_ref_images = self.image_processor.preprocess(
                rearrange(subject_ref_images, "b c f h w -> (b f) c h w"),
                height=height,
                width=width
            )
            subject_ref_images = subject_ref_images.to(dtype=torch.float32)
            subject_ref_images = rearrange(subject_ref_images, "(b f) c h w -> b c f h w", f=video_length)
            subject_ref_images = subject_ref_images.to(dtype=weight_dtype, device=device)

            # Convert to list of individual frames for per-frame processing
            bs, c, f, h, w = subject_ref_images.size()
            new_subject_ref_images = []
            for i in range(bs):
                new_subject_ref_images.append([])
                for j in range(f):
                    new_subject_ref_images[i].append(subject_ref_images[i, :, j:j+1])
            subject_ref_images = new_subject_ref_images
        
        # Encode control videos and masks to latent space for GeoAda
        geoada_latents = self.geoada_encode_multi_frames(input_videos, subject_ref_images)

        mask_latents = self.geoada_encode_masks(mask_condition, subject_ref_images)
        geoada_context = self.geoada_latent(geoada_latents, mask_latents)

        # 5. Prepare latents - initialize with noise for diffusion process
        # Noise shape is adjusted based on temporal compression from GeoAda context
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
            num_length_latents=geoada_latents[0].size(1)
        )

        # 6. Prepare extra step kwargs for scheduler (e.g., eta for DDIM, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (
            self.vae.latent_channels,
            geoada_latents[0].size(1),
            geoada_latents[0].size(2),
            geoada_latents[0].size(3),
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )
        
        # 7. Denoising loop - iteratively denoise latents from noise to clean samples
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                # Duplicate latents for classifier-free guidance (unconditioned + conditioned)
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Duplicate GeoAda context for classifier-free guidance
                geoada_context_input = (
                    torch.stack(geoada_context * 2)
                    if do_classifier_free_guidance
                    else geoada_context
                )

                # Broadcast timestep to batch dimension in ONNX/Core ML compatible way
                timestep = t.expand(latent_model_input.shape[0])
                
                # Predict noise model output using transformer with mixed precision
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        geoada_context=geoada_context_input,
                        seq_len=seq_len,
                        geoada_context_scale=geoada_context_scale
                    )

                # Apply classifier-free guidance to noise prediction
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Scheduler step: compute x_t-1 from x_t and predicted noise
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Execute callback if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        # Remove subject reference frames that were prepended (for GeoAda spatial control)
        if subject_ref_images is not None:
            len_subject_ref_images = len(subject_ref_images[0])
            latents = latents[:, :, len_subject_ref_images:, :, :]

        # 8. Decode latents to pixel space
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            # Return latents directly without decoding
            video = latents

        # Offload all models to CPU for memory efficiency
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
