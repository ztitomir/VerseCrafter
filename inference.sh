# example 0
INPUT_IMAGE=demo_data/LXKcD2zSPMc_0351466_0353266_0001469_0001550/0001.png
OUTPUT_DIR=demo_data/LXKcD2zSPMc_0351466_0353266_0001469_0001550
MODEL_PATH="model/VerseCrafter"
PROMPT='A sun-drenched street in Valletta, Malta, showcasing towering honey-colored limestone buildings adorned with traditional wrought-iron balconies and arched doorways. On the left-hand sidewalk, a man in a bright orange T-shirt and a woman in a beige summer dress walk side-by-side. Several cars are parked in the distance. The vibrant Mediterranean sunlight casts soft shadows, illuminating the weathered textures of the ancient architecture, which stretches towards distant city fortifications under a clear, pale blue sky.'

# example 1
INPUT_IMAGE=demo_data/dc58debb-191d-50cb-8ba1-a5afdeec1808_0000091_0000172/0001.png
OUTPUT_DIR=demo_data/dc58debb-191d-50cb-8ba1-a5afdeec1808_0000091_0000172
MODEL_PATH="model/VerseCrafter"
PROMPT='In the video, a couple is walking hand in hand through a snowy outdoor market at night. The man is wearing a black puffer jacket and blue jeans, while the woman is dressed in a black puffer jacket and beige pants. They appear to be enjoying their time together as they stroll past various stalls.  The first stall they pass by has a display of items such as mugs, ornaments, and other merchandise arranged on shelves. The stall is well-lit with string lights, creating a warm and inviting atmosphere. Next to it, there is another stall with a yellow interior, decorated with green plants and more string lights. This stall seems to be selling food or drinks, as there are bottles and containers visible on the counter.  In the background, there are more stalls and people sitting at tables under umbrellas, suggesting that the market is bustling with activity despite the cold weather. The overall scene is festive and cozy, with the snow adding to the charm of the evening.'

# 1. Depth Estimation with MoGE-V2
python inference/moge-v2_infer.py  -i $INPUT_IMAGE -o $OUTPUT_DIR/estimated_depth --maps


# 2. Segmentation with Grounded-SAM-2
python inference/grounded_sam2_infer.py \
    --image_path "$INPUT_IMAGE" \
    --text_prompt "person . car ." \
    --output_dir "$OUTPUT_DIR/object_mask" \
    --min_area_ratio 0.003 \
    --max_area_ratio 0.2

# 3. Fit 3D Gaussian to segmented objects
python inference/fit_3D_gaussian.py \
    --image_path $INPUT_IMAGE \
    --npz_path $OUTPUT_DIR/estimated_depth/depth_intrinsics.npz \
    --masks_dir $OUTPUT_DIR/object_mask/masks \
    --output_dir $OUTPUT_DIR/fitted_3D_gaussian


# 4. Customize Trajectory (Manual Operation in Blender)
# ============================================================
# NOTE: The following scripts CANNOT be run directly from the command line.
#       You must run them inside Blender's Scripting environment:
#
#   1. Open Blender and go to the "Scripting" tab
#   2. Update ROOT_DIR in both scripts to your absolute input directory path
#   3. Run build_4d_control_scene.py to load the scene (point cloud, camera, objects)
#   4. Switch to "Layout" tab to customize camera and object trajectories
#   5. Return to "Scripting" tab and run export_blender_custom_trajectories.py
#      to export custom_camera_trajectory.npz and custom_3D_gaussian_trajectory.json
#
# See README.md Step 4 for detailed instructions.
# ============================================================
# inference/blender_script/build_4d_control_scene.py
# inference/blender_script/export_blender_custom_trajectories.py


# 5. Render 4D control maps
python inference/rendering_4D_control_maps.py \
    --png_path $INPUT_IMAGE \
    --npz_path $OUTPUT_DIR/estimated_depth/depth_intrinsics.npz \
    --mask_dir $OUTPUT_DIR/object_mask/masks \
    --trajectory_npz $OUTPUT_DIR/camera_object_0/custom_camera_trajectory.npz \
    --ellipsoid_json $OUTPUT_DIR/camera_object_0/custom_3D_gaussian_trajectory.json \
    --output_dir $OUTPUT_DIR/camera_object_0/rendering_4D_maps


# 6. VerseCrafter Inference 
torchrun --nproc-per-node=8 inference/versecrafter_inference.py \
  --transformer_path $MODEL_PATH \
  --num_inference_steps 30 \
  --sample_size "720,1280" \
  --ulysses_degree 2 \
  --ring_degree 4 \
  --prompt "$PROMPT" \
  --input_image_path $INPUT_IMAGE \
  --save_path $OUTPUT_DIR/camera_object_0 \
  --rendering_maps_path $OUTPUT_DIR/camera_object_0/rendering_4D_maps
