"""
VerseCrafter Blender Addon

This addon provides a UI for the VerseCrafter video generation pipeline,
enabling seamless workflow between Blender and the GPU server.

Installation:
1. In Blender, go to Edit > Preferences > Add-ons
2. Click "Install..." and select this folder
3. Enable "VerseCrafter Workflow" addon

Usage:
1. Set the API server URL in the addon settings
2. Use the VerseCrafter panel in the 3D View sidebar (N key)
3. Follow the workflow steps in order
"""

bl_info = {
    "name": "VerseCrafter Workflow",
    "author": "VerseCrafter Team",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > VerseCrafter",
    "description": "Integrate VerseCrafter video generation with Blender",
    "category": "Animation",
}

import bpy
import os
import sys

# Add addon directory to path for imports
addon_dir = os.path.dirname(os.path.realpath(__file__))
if addon_dir not in sys.path:
    sys.path.append(addon_dir)

from .operators import (
    VERSECRAFTER_OT_preprocess,
    VERSECRAFTER_OT_load_scene,
    VERSECRAFTER_OT_export_trajectories,
    VERSECRAFTER_OT_postprocess,
    VERSECRAFTER_OT_test_connection,
    VERSECRAFTER_OT_dummy_progress,
    VERSECRAFTER_OT_open_folder,
)

from .panels import (
    VERSECRAFTER_PT_main_panel,
    VERSECRAFTER_PT_settings_panel,
    VERSECRAFTER_PT_preprocess_panel,
    VERSECRAFTER_PT_trajectory_panel,
    VERSECRAFTER_PT_generate_panel,
)

from .properties import VerseCrafterProperties, VerseCrafterPreferences


classes = [
    # Properties
    VerseCrafterProperties,
    VerseCrafterPreferences,
    # Operators
    VERSECRAFTER_OT_preprocess,
    VERSECRAFTER_OT_load_scene,
    VERSECRAFTER_OT_export_trajectories,
    VERSECRAFTER_OT_postprocess,
    VERSECRAFTER_OT_test_connection,
    VERSECRAFTER_OT_dummy_progress,
    VERSECRAFTER_OT_open_folder,
    # Panels (order matters for UI display)
    VERSECRAFTER_PT_main_panel,
    VERSECRAFTER_PT_settings_panel,
    VERSECRAFTER_PT_preprocess_panel,
    VERSECRAFTER_PT_trajectory_panel,
    VERSECRAFTER_PT_generate_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.versecrafter = bpy.props.PointerProperty(type=VerseCrafterProperties)
    
    print("VerseCrafter addon registered")


def unregister():
    del bpy.types.Scene.versecrafter
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    print("VerseCrafter addon unregistered")


if __name__ == "__main__":
    register()

