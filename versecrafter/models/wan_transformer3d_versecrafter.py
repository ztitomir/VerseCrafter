# Modified from https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
videox_fun_path = os.path.join(project_root, 'third_party/VideoX-Fun')
if videox_fun_path not in sys.path:
    sys.path.insert(0, videox_fun_path)

from typing import Any, Dict

import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from versecrafter.models.wan_transformer3d import (WanAttentionBlock, WanTransformer3DModel,
                                sinusoidal_embedding_1d, _prepare_sequence_tensors,
                                _process_teacache_skip_logic)
from videox_fun.utils import cfg_skip


def _process_blocks_forward(blocks, x, kwargs, hints, geoada_context_scale,
                            gradient_checkpointing, e0, seq_lens, grid_sizes, freqs,
                            context, context_lens, dtype, t):
    """
    Helper function to process blocks forward pass with gradient checkpointing support.
    Extracts the common logic for iterating through blocks with gradient checkpointing.
    Specific to VerseCrafterWanTransformer3DModel for handling GeoAda blocks.
    
    Args:
        blocks (nn.ModuleList): Transformer blocks to iterate through
        x (Tensor): Input tensor to process through blocks
        kwargs (Dict): Forward pass keyword arguments
        hints (Tensor): GeoAda hints for spatial control
        geoada_context_scale (float): GeoAda context scale
        gradient_checkpointing (bool): Whether to use gradient checkpointing
        e0 (Tensor): Time embeddings
        seq_lens (Tensor): Sequence lengths
        grid_sizes (Tensor): Grid sizes for unpatchification
        freqs (Tensor): Frequency embeddings for RoPE
        context (Tensor): Text context embeddings
        context_lens (Tensor): Context lengths
        dtype (torch.dtype): Data type for computation
        t (Tensor): Timesteps
    
    Returns:
        Tensor: Processed output tensor
    """
    # Prepare extra kwargs for gradient checkpointing
    extra_kwargs = {
        'e': e0,
        'seq_lens': seq_lens,
        'grid_sizes': grid_sizes,
        'freqs': freqs,
        'context': context,
        'context_lens': context_lens,
        'dtype': dtype,
        't': t,
    }
    
    for block in blocks:
        if torch.is_grad_enabled() and gradient_checkpointing:
            def create_custom_forward(module, **static_kwargs):
                def custom_forward(*inputs):
                    return module(*inputs, **static_kwargs)
                return custom_forward
            
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **extra_kwargs),
                    x,
                    hints,
                    geoada_context_scale,
                    **ckpt_kwargs,
                )
        else:
            x = block(x, **kwargs)
    
    return x


class VerseCrafterWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)

        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class VerseCrafterWanTransformer3DModel(WanTransformer3DModel):
    @register_to_config
    def __init__(self,
                 geoada_layers=None,
                 geoada_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.geoada_layers = [i for i in range(0, self.num_layers, 2)] if geoada_layers is None else geoada_layers
        self.geoada_in_dim = self.in_dim if geoada_in_dim is None else geoada_in_dim

        assert 0 in self.geoada_layers
        self.geoada_layers_mapping = {i: n for n, i in enumerate(self.geoada_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, 
                                    self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.geoada_layers_mapping[i] if i in self.geoada_layers else None)
            for i in range(self.num_layers)
        ])

        # geoada blocks
        self.geoada_blocks = nn.ModuleList([
            VerseCrafterWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, 
                                        self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.geoada_layers
        ])

        # geoada patch embeddings
        self.geoada_patch_embedding = nn.Conv3d(
            self.geoada_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        """
        Override the parent's from_pretrained to handle geoada_in_dim changes.
        When geoada_in_dim differs from pretrained weights, reinitialize related parameters.
        """
        # Check if geoada_in_dim is being changed by reading the config
        import json
        if subfolder is not None:
            config_path = os.path.join(pretrained_model_path, subfolder, 'config.json')
        else:
            config_path = os.path.join(pretrained_model_path, 'config.json')
        
        geoada_in_dim_changed = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            pretrained_geoada_in_dim = config.get('geoada_in_dim', config.get('in_dim', 16))
            requested_geoada_in_dim = transformer_additional_kwargs.get('geoada_in_dim', pretrained_geoada_in_dim)
            
            if requested_geoada_in_dim != pretrained_geoada_in_dim:
                geoada_in_dim_changed = True
                print(f"Detected geoada_in_dim change from {pretrained_geoada_in_dim} to {requested_geoada_in_dim}")
                print(f"geoada_patch_embedding will be reinitialized after loading pretrained weights...")
        
        # Call the parent's from_pretrained to load the model
        model = super().from_pretrained(
            pretrained_model_path, 
            subfolder=subfolder, 
            transformer_additional_kwargs=transformer_additional_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage, 
            torch_dtype=torch_dtype
        )
        
        # If geoada_in_dim changed, reinitialize geoada_patch_embedding
        # (Other parameters like before_proj/after_proj are already loaded correctly)
        if geoada_in_dim_changed:
            print(f"Reinitializing geoada_patch_embedding with Xavier initialization...")
            # The geoada_patch_embedding was already created with new geoada_in_dim in from_config,
            # but it kept the random initialization. Now we apply proper Xavier initialization.
            nn.init.xavier_uniform_(model.geoada_patch_embedding.weight.flatten(1))
            if model.geoada_patch_embedding.bias is not None:
                nn.init.zeros_(model.geoada_patch_embedding.bias)
            
            print(f"Successfully reinitialized geoada_patch_embedding for geoada_in_dim={requested_geoada_in_dim}")
        
        return model

    def forward_geoada(
        self,
        x,
        geoada_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.geoada_patch_embedding(u.unsqueeze(0)) for u in geoada_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        
        for block in self.geoada_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, **new_kwargs),
                        c,
                        **ckpt_kwargs,
                    )
            else:
                c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        geoada_context,
        context,
        seq_len,
        geoada_context_scale=1.0,
        clip_fea=None,
        y=None,
        cond_flag=True
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        dtype = x.dtype
        device = self.patch_embedding.weight.device
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x, seq_lens = _prepare_sequence_tensors(x, seq_len, self.sp_world_size)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t)
        hints = self.forward_geoada(x, geoada_context, seq_len, kwargs)

        kwargs['hints'] = hints
        kwargs['context_scale'] = geoada_context_scale

        # TeaCache
        if self.teacache is not None:
            self.should_calc = _process_teacache_skip_logic(self.teacache, e0, t, cond_flag)
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = (
                    self.teacache.previous_residual_cond 
                    if cond_flag 
                    else self.teacache.previous_residual_uncond
                )
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()
                
                # Process blocks through gradient checkpointing
                x = _process_blocks_forward(
                    self.blocks, x, kwargs, hints, geoada_context_scale,
                    self.gradient_checkpointing, e0, seq_lens, grid_sizes, self.freqs,
                    context, context_lens, dtype, t
                )
                
                # Update TeaCache residual after block processing
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            x = _process_blocks_forward(
                self.blocks, x, kwargs, hints, geoada_context_scale,
                self.gradient_checkpointing, e0, seq_lens, grid_sizes, self.freqs,
                context, context_lens, dtype, t
            )

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
