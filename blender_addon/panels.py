"""
VerseCrafter Blender Addon Panels

Defines UI panels for the addon.

Workflow:
- Settings: Server configuration
- Step 1: Preprocessing (depth, segmentation, gaussian fitting)
- Step 2: Edit Trajectories in Blender (camera & object animation)
- Step 3: Generate Video (export trajectories, render control maps, generate video)
"""

import bpy
from bpy.types import Panel


class VERSECRAFTER_PT_main_panel(Panel):
    """Main VerseCrafter panel in the 3D View sidebar."""
    bl_label = "VerseCrafter"
    bl_idname = "VERSECRAFTER_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VerseCrafter'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.versecrafter
        
        # Status section with progress bar style
        box = layout.box()
        
        # Step 1: Preprocessing (combines old steps 1-3)
        step1_complete = props.step1_complete and props.step2_complete and props.step3_complete
        # Step 2: Edit Trajectories (old step 4 - but just editing, not export)
        step2_complete = props.step4_complete
        # Step 3: Generate Video (old steps 4-6 - export + render + generate)
        step3_complete = props.step6_complete
        
        # Calculate overall progress
        completed_steps = sum([step1_complete, step2_complete, step3_complete])
        progress_percent = completed_steps / 3.0
        
        # Progress bar header
        header_row = box.row()
        header_row.label(text=f"Progress: {completed_steps}/3 Steps", icon='SORTTIME')
        
        # Visual progress bar using split columns
        progress_row = box.row(align=True)
        progress_row.scale_y = 0.5
        
        steps_info = [
            (step1_complete, "1: Preprocess"),
            (step2_complete, "2: Trajectories"),
            (step3_complete, "3: Video"),
        ]
        
        for complete, name in steps_info:
            col = progress_row.column(align=True)
            col.scale_x = 1.0
            if complete:
                col.alert = False
                col.operator("versecrafter.dummy_progress", text="✓", emboss=True, depress=True)
            else:
                col.alert = True
                col.operator("versecrafter.dummy_progress", text="○", emboss=True, depress=False)
        
        # Step labels row
        label_row = box.row(align=True)
        label_row.scale_y = 0.6
        for complete, name in steps_info:
            col = label_row.column(align=True)
            col.label(text=name)
        
        # Status message
        if props.status_message:
            box.separator()
            msg_row = box.row()
            msg_row.label(text=props.status_message)
        
        # Processing indicator with animated dots effect
        if props.is_processing:
            proc_row = box.row()
            proc_row.alert = True
            proc_row.label(text="⏳ Processing...", icon='TIME')


class VERSECRAFTER_PT_settings_panel(Panel):
    """Settings panel - Server configuration."""
    bl_label = "Server Settings"
    bl_idname = "VERSECRAFTER_PT_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VerseCrafter'
    bl_parent_id = "VERSECRAFTER_PT_main_panel"
    bl_options = set()  # Expanded by default
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.versecrafter
        
        # Server URL
        layout.prop(props, "server_url")
        
        # Proxy cookie (for authenticated proxies)
        box = layout.box()
        box.label(text="Proxy Authentication (if needed):", icon='LOCKED')
        box.prop(props, "proxy_cookie", text="Cookie")
        box.label(text="Copy from browser: F12 → Network → 8188/ → Request Header → Cookie", icon='INFO')
        
        # Test connection button
        row = layout.row()
        row.operator("versecrafter.test_connection", icon='URL', text="Test Connection")


class VERSECRAFTER_PT_preprocess_panel(Panel):
    """Step 1: Preprocessing panel."""
    bl_label = "Step 1: Preprocessing"
    bl_idname = "VERSECRAFTER_PT_preprocess_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VerseCrafter'
    bl_parent_id = "VERSECRAFTER_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.versecrafter
        
        # Input image (local)
        layout.prop(props, "input_image_path")
        
        # Workflow directory
        layout.prop(props, "workflow_dir")
        
        # Object detection prompt
        layout.prop(props, "text_prompt")
        
        # Area ratio settings
        box = layout.box()
        box.label(text="Object Detection Area Ratio:", icon='FILTER')
        col = box.column(align=True)
        col.prop(props, "min_area_ratio")
        col.prop(props, "max_area_ratio")
        
        # Run button
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.enabled = not props.is_processing
        row.operator("versecrafter.preprocess", icon='PLAY', text="Run Preprocessing")


class VERSECRAFTER_PT_trajectory_panel(Panel):
    """Step 2: Trajectory editing panel."""
    bl_label = "Step 2: Customize trajectory"
    bl_idname = "VERSECRAFTER_PT_trajectory_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VerseCrafter'
    bl_parent_id = "VERSECRAFTER_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.versecrafter
        
        # Check if preprocessing is complete
        step1_complete = props.step1_complete and props.step2_complete and props.step3_complete
        if not step1_complete:
            layout.label(text="Run Step 1 first", icon='ERROR')
            return
        
        # Instructions
        box = layout.box()
        box.label(text="Operating in Blender Layout:", icon='OUTLINER_OB_CAMERA')
        col = box.column(align=True)
        col.label(text="1. Set video length")
        col.label(text="2. Customize camera trajectory")
        col.label(text="3. Customize 3D Gaussian trajectory")
        
        # Output settings
        layout.separator()
        layout.prop(props, "output_name")
        layout.prop(props, "video_length")
        
        # Export trajectories button
        layout.separator()
        row = layout.row()
        row.scale_y = 1.3
        row.enabled = not props.is_processing
        row.operator("versecrafter.export_trajectories", icon='EXPORT', text="Export Trajectories")
        
        # Open folder button (moved from Step 3)
        if props.workflow_dir:
            op = layout.operator("versecrafter.open_folder", icon='FILE_FOLDER', text="Open Workflow Folder")
            op.folder_type = "workflow"


class VERSECRAFTER_PT_generate_panel(Panel):
    """Step 3: Generate Video panel (export + render + generate)."""
    bl_label = "Step 3: Generate Video"
    bl_idname = "VERSECRAFTER_PT_generate_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VerseCrafter'
    bl_parent_id = "VERSECRAFTER_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.versecrafter
        
        # Check if preprocessing is complete
        step1_complete = props.step1_complete and props.step2_complete and props.step3_complete
        if not step1_complete:
            layout.label(text="Run Step 1 first", icon='ERROR')
            return
        
        # Video prompt
        layout.prop(props, "video_prompt")
        
        # Generation settings
        box = layout.box()
        box.label(text="Generation Settings", icon='PREFERENCES')
        
        col = box.column(align=True)
        col.prop(props, "num_inference_steps")
        col.prop(props, "guidance_scale")
        col.prop(props, "fps")
        col.prop(props, "seed")
        
        # Run button - this will render control maps and generate video
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.enabled = not props.is_processing and props.step4_complete
        row.operator("versecrafter.postprocess", icon='RENDER_ANIMATION', text="Generate Video")
        
        # Hint if trajectories not exported
        if not props.step4_complete:
            layout.label(text="⚠️ Export trajectories in Step 2 first", icon='INFO')
        
        # Open output folder
        if props.step6_complete:
            layout.separator()
            layout.label(text="🎉 Video generated!", icon='CHECKMARK')
            op = layout.operator("versecrafter.open_folder", icon='FILE_MOVIE', text="Open Video Folder")
            op.folder_type = "video"
