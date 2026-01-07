"""
VerseCrafter Blender Addon Properties

Defines addon properties and preferences.
"""

import bpy
from bpy.types import PropertyGroup, AddonPreferences
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
)


def update_video_length(self, context):
    """Update Blender timeline when video_length changes."""
    scene = context.scene
    scene.frame_start = 1
    scene.frame_end = self.video_length
    print(f"[VerseCrafter] Timeline updated: frame 1 - {self.video_length}")


class VerseCrafterProperties(PropertyGroup):
    """Properties for VerseCrafter workflow state."""
    
    # Server settings
    server_url: StringProperty(
        name="Server URL",
        description="VerseCrafter API server URL",
        default="http://localhost:8188"
    )
    
    # Input settings
    input_image_path: StringProperty(
        name="Input Image",
        description="Path to the input image (local path)",
        default="",
        subtype='FILE_PATH'
    )
    
    proxy_cookie: StringProperty(
        name="Proxy Cookie",
        description="Cookie for proxy authentication (copy from browser)",
        default=""
    )
    
    text_prompt: StringProperty(
        name="Object Prompt",
        description="Text prompt for object detection (e.g., 'person . car .')",
        default="person . car ."
    )
    
    generation_prompt: StringProperty(
        name="Generation Prompt",
        description="Text prompt for video generation",
        default="A beautiful scene with natural motion."
    )
    
    video_prompt: StringProperty(
        name="Video Prompt",
        description="Text description of the video to generate",
        default="A dynamic scene with smooth camera movement."
    )
    
    # Workflow directory
    workflow_dir: StringProperty(
        name="Workflow Directory",
        description="Directory for intermediate files",
        default="",
        subtype='DIR_PATH'
    )
    
    # Output settings
    output_name: StringProperty(
        name="Output Name",
        description="Name prefix for output files",
        default="camera_object_0"
    )
    
    # Processing settings
    depth_model_version: EnumProperty(
        name="Depth Model",
        description="MoGE depth estimation model version",
        items=[
            ('v2', 'MoGE-V2', 'Latest model with normal estimation'),
            ('v1', 'MoGE-V1', 'Original model'),
        ],
        default='v2'
    )
    
    use_fp16: BoolProperty(
        name="Use FP16",
        description="Use half precision for faster inference",
        default=True
    )
    
    resolution_level: IntProperty(
        name="Resolution Level",
        description="Depth estimation resolution (0-9)",
        default=9,
        min=0,
        max=9
    )
    
    box_threshold: FloatProperty(
        name="Detection Threshold",
        description="Confidence threshold for object detection",
        default=0.4,
        min=0.0,
        max=1.0
    )
    
    min_area_ratio: FloatProperty(
        name="Min Area Ratio",
        description="Minimum object area ratio",
        default=0.003,
        min=0.0,
        max=1.0,
        precision=4
    )
    
    max_area_ratio: FloatProperty(
        name="Max Area Ratio",
        description="Maximum object area ratio",
        default=0.2,
        min=0.0,
        max=1.0,
        precision=4
    )
    
    # Video generation settings
    num_inference_steps: IntProperty(
        name="Inference Steps",
        description="Number of diffusion steps",
        default=30,
        min=1,
        max=200
    )
    
    guidance_scale: FloatProperty(
        name="Guidance Scale",
        description="Classifier-free guidance scale",
        default=5.0,
        min=1.0,
        max=20.0
    )
    
    video_length: IntProperty(
        name="Video Length",
        description="Number of frames to generate",
        default=81,
        min=1,
        max=241,
        update=update_video_length
    )
    
    fps: IntProperty(
        name="FPS",
        description="Frames per second",
        default=16,
        min=1,
        max=60
    )
    
    seed: IntProperty(
        name="Seed",
        description="Random seed for generation",
        default=2025,
        min=0
    )
    
    # Status tracking
    step1_complete: BoolProperty(name="Step 1 Complete", default=False)
    step2_complete: BoolProperty(name="Step 2 Complete", default=False)
    step3_complete: BoolProperty(name="Step 3 Complete", default=False)
    step4_complete: BoolProperty(name="Step 4 Complete", default=False)
    step5_complete: BoolProperty(name="Step 5 Complete", default=False)
    step6_complete: BoolProperty(name="Step 6 Complete", default=False)
    
    # Processing state
    is_processing: BoolProperty(name="Is Processing", default=False)
    status_message: StringProperty(name="Status Message", default="")


class VerseCrafterPreferences(AddonPreferences):
    """Addon preferences for VerseCrafter."""
    
    bl_idname = __package__
    
    default_server_url: StringProperty(
        name="Default Server URL",
        description="Default API server URL",
        default="http://localhost:8188"
    )
    
    model_path: StringProperty(
        name="VerseCrafter Model Path",
        description="Path to VerseCrafter model on server",
        default="model/VerseCrafter"
    )
    
    base_model_path: StringProperty(
        name="Base Model Path",
        description="Path to Wan2.1 base model on server",
        default="model/Wan2.1-T2V-14B"
    )
    
    auto_save_blend: BoolProperty(
        name="Auto-save .blend",
        description="Automatically save .blend file before processing",
        default=True
    )
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Server Settings:")
        layout.prop(self, "default_server_url")
        
        layout.separator()
        layout.label(text="Model Paths (on Server):")
        layout.prop(self, "model_path")
        layout.prop(self, "base_model_path")
        
        layout.separator()
        layout.prop(self, "auto_save_blend")

