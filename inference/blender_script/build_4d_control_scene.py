"""
File Name: build_4d_control_scene.py
Description: Initializes a scene environment for 4D Geometric Control within Blender.

This script aims to construct an interactively editable 4D world state based on 
input RGB-D data and 3D Gaussian parameters.

Key Components:
1.  Static Background Point Cloud: Reconstructs the background point cloud ($P^{bg}$) 
    from the input RGB image and Depth map, providing 3D spatial context.
2.  Editable Object Trajectories: Converts per-frame/per-object 3D Gaussian parameters 
    ($\mathcal{G}_{o}^{t}$) into editable ellipsoid meshes, serving as geometric proxies 
    for the objects' motion and shape.
3.  Camera Pose and Path: Sets up the render camera based on extrinsic matrices and 
    visualizes the camera frustum and trajectory path (CamPath), facilitating user 
    adjustment of camera motion.

Workflow:
    - Run this script to import the initial scene geometry and trajectory estimates.
    - Users can interactively specify or refine the 4D trajectories of objects and 
      the camera by keyframing and dragging the 'Ellipsoid_*' objects and the 
      'CamPath' curve in the Blender viewport.
    - All coordinate systems in the scene are unified and transformed to the Blender 
      world coordinate system (Z-up).
"""
import bpy
import numpy as np
import cv2
import mathutils
import json
import math
from pathlib import Path
import matplotlib
import matplotlib.cm as cm

# ================== Configuration ==================
ROOT_DIR = "/absolute/path/to/demo_data/your_folder"
PNG_PATH = f"{ROOT_DIR}/0001.png"  
NPZ_PATH = f"{ROOT_DIR}/estimated_depth/depth_intrinsics.npz"
JSON_PATH = f"{ROOT_DIR}/fitted_3D_gaussian/gaussian_params.json"

PIXEL_STEP = 1
SENSOR_WIDTH = 36.0

# Ellipsoid parameters
ELLIPSOID_SCALE_FACTOR = 2.0
ELLIPSOID_SEGMENTS = 128
ELLIPSOID_RINGS = 64

# Static snapshot parameters
NUM_SNAPSHOT_FRAMES = 1
SNAPSHOT_OPACITY = 0.6
OBJECT_SNAPSHOT_OPACITY = 0.8
SHOW_CAMERA_FRUSTUM = False
CAMERA_FRUSTUM_SIZE = 0.5
SHOW_OBJECT_SNAPSHOTS = False
SHOW_CAMERA_FRAMES = False
CAMERA_FRAME_OPACITY = 0.9

# Path visualization parameters
SHOW_CAMERA_PATH = False
SHOW_ELLIPSOID_PATHS = False
CAMERA_PATH_RADIUS = 0.002
ELLIPSOID_PATH_RADIUS = 0.003


# ================== Cleanup Functions ==================
def remove_objects_by_prefix(prefixes):
    """Remove Blender objects matching given prefixes."""
    for obj in list(bpy.data.objects):
        if any(obj.name.startswith(p) for p in prefixes) or obj.name in prefixes:
            bpy.data.objects.remove(obj, do_unlink=True)


def remove_meshes_by_prefix(prefixes):
    """Remove Blender meshes matching given prefixes."""
    for mesh in list(bpy.data.meshes):
        if any(mesh.name.startswith(p) for p in prefixes):
            bpy.data.meshes.remove(mesh, do_unlink=True)


def remove_curves_by_name(names):
    """Remove Blender curves matching given names."""
    for curve in list(bpy.data.curves):
        if curve.name in names:
            bpy.data.curves.remove(curve, do_unlink=True)


def remove_materials_by_prefix(prefixes):
    """Remove Blender materials matching given prefixes."""
    for mat in list(bpy.data.materials):
        if any(mat.name.startswith(p) for p in prefixes):
            bpy.data.materials.remove(mat, do_unlink=True)


# Cleanup old objects
remove_objects_by_prefix([
    "Camera_", "PointCloud", "CamPath", "Ellipsoid_", 
    "EllipsoidPath_", "ObjectSnapshot_", "CameraFrame_"
])
remove_meshes_by_prefix([
    "PointCloudMesh", "EllipsoidMesh_", "CameraFrustum_", 
    "ObjectSnapshotMesh_", "CameraFrame_"
])
remove_curves_by_name(["CamPath"])
remove_materials_by_prefix(["PointCloudMat", "EllipsoidMat_", "CameraMat_", "ObjectSnapshotMat_", "CameraFrameMat_"])

# Cleanup old images
for img in list(bpy.data.images):
    if img.name.startswith("CameraFrameImage_"):
        bpy.data.images.remove(img)

# ================== Load NPZ Data ==================
data = np.load(NPZ_PATH)
depths = data["depth"]
intrinsics = data["intrinsic"]

num_frames = depths.shape[0] if depths.ndim == 3 else 1
if depths.ndim == 2:
    depths = depths[np.newaxis, ...]
    num_frames = 1

height_d, width_d = depths.shape[1], depths.shape[2]

# Denormalize intrinsics if needed
if intrinsics.ndim == 2:
    if intrinsics[0, 0] < 10 or intrinsics[1, 1] < 10:
        print(f"[INFO] Detected normalized intrinsics, converting to pixel coordinates...")
        intrinsics_denorm = intrinsics.copy()
        intrinsics_denorm[0, 0] = intrinsics[0, 0] * width_d
        intrinsics_denorm[1, 1] = intrinsics[1, 1] * height_d
        intrinsics_denorm[0, 2] = intrinsics[0, 2] * width_d
        intrinsics_denorm[1, 2] = intrinsics[1, 2] * height_d
        intrinsics = intrinsics_denorm
        print(f"[INFO] Denormalized K: fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}")
elif intrinsics.ndim == 3:
    if intrinsics[0, 0, 0] < 10 or intrinsics[0, 1, 1] < 10:
        print(f"[INFO] Detected normalized intrinsics, converting to pixel coordinates...")
        intrinsics_denorm = intrinsics.copy()
        intrinsics_denorm[:, 0, 0] = intrinsics[:, 0, 0] * width_d
        intrinsics_denorm[:, 1, 1] = intrinsics[:, 1, 1] * height_d
        intrinsics_denorm[:, 0, 2] = intrinsics[:, 0, 2] * width_d
        intrinsics_denorm[:, 1, 2] = intrinsics[:, 1, 2] * height_d
        intrinsics = intrinsics_denorm
        print(f"[INFO] Denormalized K[0]: fx={intrinsics[0,0,0]:.2f}, fy={intrinsics[0,1,1]:.2f}")

# ================== Coordinate Transform Matrix ==================
# OpenCV: X=right, Y=down, Z=forward
# Blender (Z-up): X=right, Y=forward, Z=up
COORD_TRANSFORM_CV2BLENDER = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float32)

# Create camera extrinsics (identity: camera at origin, no rotation)
extrinsics_opencv = np.zeros((num_frames, 4, 4), dtype=np.float32)
for i in range(num_frames):
    extrinsics_opencv[i] = np.eye(4, dtype=np.float32)

# Convert camera extrinsics to Blender coordinate system
extrinsics = np.zeros((num_frames, 4, 4), dtype=np.float32)
transform_4x4 = np.eye(4, dtype=np.float32)
transform_4x4[:3, :3] = COORD_TRANSFORM_CV2BLENDER
transform_inv = transform_4x4.copy()
transform_inv[:3, :3] = COORD_TRANSFORM_CV2BLENDER.T

for i in range(num_frames):
    extrinsics[i] = transform_4x4 @ extrinsics_opencv[i] @ transform_inv

print(f"[INFO] Camera extrinsics: Converted from OpenCV to Blender coordinate system")
print(f"[INFO] Depth range: min={depths.min():.3f}, max={depths.max():.3f}, mean={depths.mean():.3f}")

# ================== Load PNG Image ==================
rgb_image = cv2.imread(PNG_PATH)
if rgb_image is None:
    raise RuntimeError("Cannot open PNG image: " + PNG_PATH)

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
height_v, width_v, _ = rgb_image.shape
rgb_frames = np.array([rgb_image] * num_frames)

print(f"[INFO] Loaded PNG image: {height_v}x{width_v}")

# ================== Resolution Alignment ==================
if (height_d, width_d) != (height_v, width_v):
    print("[WARN] Depth vs RGB size mismatch, resizing depth to RGB size.")
    new_depths = []
    for i in range(num_frames):
        d_resized = cv2.resize(depths[i], (width_v, height_v), interpolation=cv2.INTER_NEAREST)
        new_depths.append(d_resized)
    depths = np.stack(new_depths, axis=0)
    height_d, width_d = height_v, width_v

# ================== Setup Scene ==================
scene = bpy.context.scene

# Calculate snapshot frame indices
if NUM_SNAPSHOT_FRAMES >= num_frames:
    snapshot_indices = list(range(num_frames))
else:
    snapshot_indices = np.linspace(0, num_frames - 1, NUM_SNAPSHOT_FRAMES, dtype=int).tolist()

print(f"[INFO] Sampling {len(snapshot_indices)} frames: {snapshot_indices}")

# ================== Create Background Point Cloud ==================
print("[INFO] Creating point cloud from first frame...")
pc_frame = 0
depth = depths[pc_frame].astype(np.float32)
rgb = rgb_frames[pc_frame]

if intrinsics.ndim == 3:
    K = intrinsics[pc_frame].astype(np.float32)
else:
    K = intrinsics.astype(np.float32)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

E = extrinsics[pc_frame].astype(np.float32)
R = E[0:3, 0:3]
t = E[0:3, 3]

background_verts = []
background_colors = []

height, width = depth.shape
for v in range(0, height, PIXEL_STEP):
    for u in range(0, width, PIXEL_STEP):
        d_val = depth[v, u]
        if d_val <= 0 or not np.isfinite(d_val):
            continue

        # Back-project to camera coordinates (OpenCV convention)
        x_cv = (u - cx) / fx * d_val
        y_cv = (v - cy) / fy * d_val
        z_cv = d_val
        P_cv_opencv = np.array([x_cv, y_cv, z_cv], dtype=np.float32)
        
        # Convert camera coordinates from OpenCV to Blender
        P_cv_blender = COORD_TRANSFORM_CV2BLENDER @ P_cv_opencv
        
        # Transform to world coordinates
        P_world = R.T @ (P_cv_blender - t)
        
        background_verts.append((float(P_world[0]), float(P_world[1]), float(P_world[2])))
        color = rgb[v, u]
        background_colors.append((
            float(color[0]) / 255.0,
            float(color[1]) / 255.0,
            float(color[2]) / 255.0,
            1.0
        ))

print(f"[INFO] Point count: {len(background_verts)}")

# ================== Create Point Cloud Mesh ==================
zs = np.array([p[2] for p in background_verts], dtype=np.float32)
zs_abs = np.abs(zs[np.isfinite(zs)])
median_z = float(np.median(zs_abs)) if zs_abs.size > 0 else 1.0
print(f"[INFO] Median depth: {median_z:.3f}")

uniform_point_size = median_z * 0.01
print(f"[INFO] Uniform point size: {uniform_point_size:.6f}")

verts = []
faces = []
vcolors = []

for idx, (x, y, z) in enumerate(background_verts):
    r, g, b, a = background_colors[idx]
    size = uniform_point_size

    base = len(verts)
    verts.extend([
        (x - size, y, z - size),
        (x + size, y, z - size),
        (x + size, y, z + size),
        (x - size, y, z + size),
    ])
    faces.append((base, base + 1, base + 2, base + 3))
    vcolors.extend([(r, g, b, a)] * 4)

mesh = bpy.data.meshes.new("PointCloudMesh")
mesh.from_pydata(verts, [], faces)
mesh.update()

pc_obj = bpy.data.objects.new("PointCloud_Main", mesh)
scene.collection.objects.link(pc_obj)

if len(vcolors) == len(mesh.loops):
    color_layer = mesh.color_attributes.new(name="Col", domain='CORNER', type='FLOAT_COLOR')
    for i, c in enumerate(vcolors):
        color_layer.data[i].color = c

# Point cloud material
mat = bpy.data.materials.new("PointCloudMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

for n in list(nodes):
    nodes.remove(n)

out_node = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
em_node = nodes.new("ShaderNodeEmission")
attr_node = nodes.new("ShaderNodeAttribute")
attr_node.attribute_name = "Col"

links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
links.new(attr_node.outputs["Color"], em_node.inputs["Color"])
em_node.inputs["Strength"].default_value = 0.05

add_node = nodes.new("ShaderNodeAddShader")
links.new(bsdf.outputs["BSDF"], add_node.inputs[0])
links.new(em_node.outputs["Emission"], add_node.inputs[1])
links.new(add_node.outputs[0], out_node.inputs["Surface"])

pc_obj.data.materials.append(mat)


# ================== Utility Functions ==================
def get_color_from_colormap(color_idx: int):
    """Get color from tab20 colormap."""
    cmap = matplotlib.colormaps['tab20']
    return cmap(color_idx % 20)[:3]


def create_camera_frustum(frame_idx: int, snapshot_idx: int, K: np.ndarray, E: np.ndarray):
    """Create camera frustum visualization."""
    fx, fy = K[0, 0], K[1, 1]
    
    R_blender = E[0:3, 0:3]
    t_blender = E[0:3, 3]
    C = -R_blender.T @ t_blender
    
    R_cam_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    R_bcam2world = R_blender.T @ R_cam_rotation
    
    near = CAMERA_FRUSTUM_SIZE
    aspect = width_v / height_v
    fov_y = 2 * np.arctan(height_v / (2 * fy))
    h = near * np.tan(fov_y / 2)
    w = h * aspect
    
    frustum_corners_local = np.array([
        [0, 0, 0],
        [-w, near, h],
        [w, near, h],
        [w, near, -h],
        [-w, near, -h],
    ], dtype=np.float32)
    
    frustum_corners_world = []
    for corner in frustum_corners_local:
        if np.allclose(corner, [0, 0, 0]):
            frustum_corners_world.append(C)
        else:
            corner_world = R_bcam2world @ corner + C
            frustum_corners_world.append(corner_world)
    
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    
    mesh = bpy.data.meshes.new(f"CameraFrustum_{snapshot_idx}")
    mesh.from_pydata(frustum_corners_world, edges, [])
    mesh.update()
    
    obj = bpy.data.objects.new(f"Camera_{snapshot_idx}", mesh)
    scene.collection.objects.link(obj)
    
    mat = bpy.data.materials.new(name=f"CameraMat_{snapshot_idx}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for n in list(nodes):
        nodes.remove(n)
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.inputs["Color"].default_value = (1.0, 1.0, 0.0, 1.0)
    emission_node.inputs["Strength"].default_value = 2.0
    
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    obj.data.materials.append(mat)
    
    if SHOW_CAMERA_FRAMES:
        create_camera_frame_plane(
            frame_idx, snapshot_idx, frustum_corners_world[1:5], 
            R_bcam2world, C
        )
    
    return obj


def create_camera_frame_plane(frame_idx: int, snapshot_idx: int, corners: list, 
    R_bcam2world: np.ndarray, C: np.ndarray):
    """Create video frame plane at camera frustum near plane."""
    rgb_frame = rgb_frames[frame_idx]
    
    verts = corners
    faces_list = [(0, 1, 2, 3)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    mesh = bpy.data.meshes.new(f"CameraFrame_{snapshot_idx}")
    mesh.from_pydata(verts, [], faces_list)
    mesh.update()
    
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers[0]
    
    for i, loop in enumerate(mesh.loops):
        uv_layer.data[loop.index].uv = uvs[i]
    
    plane_obj = bpy.data.objects.new(f"CameraFrame_{snapshot_idx}", mesh)
    scene.collection.objects.link(plane_obj)
    
    img_name = f"CameraFrameImage_{snapshot_idx}"
    
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    
    img = bpy.data.images.new(img_name, width=width_v, height=height_v, alpha=True)
    
    img_data = rgb_frame.astype(np.float32) / 255.0
    img_data = np.flipud(img_data)
    
    img_data_rgba = np.zeros((height_v, width_v, 4), dtype=np.float32)
    img_data_rgba[:, :, :3] = img_data
    img_data_rgba[:, :, 3] = CAMERA_FRAME_OPACITY
    
    img.pixels = img_data_rgba.flatten()
    img.pack()
    
    mat = bpy.data.materials.new(name=f"CameraFrameMat_{snapshot_idx}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for n in list(nodes):
        nodes.remove(n)
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.image = img
    
    principled.inputs["Alpha"].default_value = CAMERA_FRAME_OPACITY
    principled.inputs["Emission Strength"].default_value = 1.0
    
    links.new(tex_image.outputs["Color"], principled.inputs["Base Color"])
    links.new(tex_image.outputs["Color"], principled.inputs["Emission Color"])
    links.new(tex_image.outputs["Alpha"], principled.inputs["Alpha"])
    links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
    
    mat.blend_method = 'BLEND'
    mat.show_transparent_back = True
    
    plane_obj.data.materials.append(mat)
    
    return plane_obj


def create_ellipsoid_snapshot(obj_id: int, snapshot_idx: int, mean: list, cov: list, color_idx: int, opacity: float):
    """Create single ellipsoid snapshot."""
    mean_np = np.array(mean, dtype=np.float32)
    cov_np = np.array(cov, dtype=np.float32)
    
    # Transform Gaussian parameters from OpenCV to Blender coordinate system
    mean_blender = COORD_TRANSFORM_CV2BLENDER @ mean_np
    cov_blender = COORD_TRANSFORM_CV2BLENDER @ cov_np @ COORD_TRANSFORM_CV2BLENDER.T
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_blender)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    axes_lengths = ELLIPSOID_SCALE_FACTOR * np.sqrt(eigenvalues)
    
    # Ensure eigenvectors form a proper rotation matrix (det = 1)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] = -eigenvectors[:, 0]
    
    # Create sphere at origin first
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=ELLIPSOID_SEGMENTS,
        ring_count=ELLIPSOID_RINGS,
        radius=1.0,
        location=(0, 0, 0)
    )
    ellipsoid_obj = bpy.context.active_object
    ellipsoid_obj.name = f"Ellipsoid_{obj_id}"
    
    mesh = ellipsoid_obj.data
    mesh.name = f"EllipsoidMesh_{obj_id}"
    
    # Compute transform matrix: rotation * scale
    scale_matrix = np.diag(axes_lengths)
    transform_3x3 = eigenvectors @ scale_matrix
    
    # Apply transform directly to mesh vertices (bake the shape)
    for v in mesh.vertices:
        pos = np.array(v.co, dtype=np.float32)
        new_pos = transform_3x3 @ pos
        v.co = mathutils.Vector(new_pos.tolist())
    
    mesh.update()
    
    # Set object location (position can be edited by user)
    ellipsoid_obj.location = mathutils.Vector(mean_blender.tolist())
    ellipsoid_obj.rotation_mode = 'QUATERNION'
    ellipsoid_obj.rotation_quaternion = mathutils.Quaternion()  # Identity rotation
    ellipsoid_obj.scale = (1.0, 1.0, 1.0)  # Unit scale
    
    # Store original covariance in Blender coordinate system as custom property
    # This allows export script to correctly reconstruct covariance after user edits
    ellipsoid_obj["cov_blender"] = cov_blender.flatten().tolist()
    ellipsoid_obj["eigenvectors"] = eigenvectors.flatten().tolist()
    ellipsoid_obj["eigenvalues"] = eigenvalues.tolist()
    
    bpy.context.view_layer.update()
    
    # Create material
    mat = bpy.data.materials.new(name=f"EllipsoidMat_{obj_id}_snap_{snapshot_idx}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for n in list(nodes):
        nodes.remove(n)
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    
    color_rgb = get_color_from_colormap(color_idx)
    principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)
    principled.inputs["Alpha"].default_value = opacity
    principled.inputs["Roughness"].default_value = 0.3
    
    fresnel_node = nodes.new("ShaderNodeFresnel")
    fresnel_node.inputs["IOR"].default_value = 1.45
    
    colorramp = nodes.new("ShaderNodeValToRGB")
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (*color_rgb, 0.3 * opacity)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (*color_rgb, 1.0 * opacity)
    
    links.new(fresnel_node.outputs["Fac"], colorramp.inputs["Fac"])
    links.new(colorramp.outputs["Color"], principled.inputs["Alpha"])
    links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
    
    mat.blend_method = 'BLEND'
    ellipsoid_obj.data.materials.append(mat)
    
    return ellipsoid_obj


def create_ellipsoid_path(obj_id: int, mean_positions: list, color_idx: int):
    """Create ellipsoid trajectory path."""
    if len(mean_positions) < 2:
        return None
    
    curve_data = bpy.data.curves.new(name=f"EllipsoidPath_{obj_id}", type='CURVE')
    curve_data.dimensions = '3D'
    
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(mean_positions) - 1)
    
    for i, pos in enumerate(mean_positions):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        spline.points[i].co = (x, y, z, 1.0)
    
    path_depth = ELLIPSOID_PATH_RADIUS if ELLIPSOID_PATH_RADIUS is not None else (median_z * 0.0003)
    curve_data.bevel_depth = path_depth
    curve_data.bevel_resolution = 2
    
    path_obj = bpy.data.objects.new(f"EllipsoidPath_{obj_id}", curve_data)
    scene.collection.objects.link(path_obj)
    
    color_rgb = get_color_from_colormap(color_idx)
    mat = bpy.data.materials.new(name=f"EllipsoidPathMat_{obj_id}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for n in list(nodes):
        nodes.remove(n)
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.inputs["Color"].default_value = (*color_rgb, 1.0)
    emission_node.inputs["Strength"].default_value = 1.5
    
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    path_obj.data.materials.append(mat)
    
    return path_obj


def setup_blender_camera(K, E, image_width, image_height):
    """Create and setup a Blender camera."""
    cam_data = bpy.data.cameras.new(name='RenderCamera')
    
    fx, fy = K[0, 0], K[1, 1]
    
    cam_data.lens_unit = 'FOV'
    cam_data.angle = 2 * math.atan(image_width / (2 * fx))

    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height
    
    cam_obj = bpy.data.objects.new('RenderCameraObj', cam_data)
    
    R_blender = E[0:3, 0:3]
    t_blender = E[0:3, 3]
    C_world = -R_blender.T @ t_blender
    
    R_cam_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    R_world_cam = R_blender.T @ R_cam_rotation
    
    cam_obj.location = mathutils.Vector(C_world)
    cam_obj.rotation_euler = mathutils.Matrix(R_world_cam).to_euler('XYZ')
    
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    print("[INFO] Created and set up Blender camera.")
    return cam_obj


# ================== Create Camera Snapshots ==================
cam_centers = []

for snapshot_idx, frame_idx in enumerate(snapshot_indices):
    if intrinsics.ndim == 3:
        K = intrinsics[frame_idx].astype(np.float32)
    else:
        K = intrinsics.astype(np.float32)
    
    E = extrinsics[frame_idx].astype(np.float32)
    R = E[0:3, 0:3]
    t = E[0:3, 3]
    C = -R.T @ t
    cam_centers.append(C.copy())
    
    if SHOW_CAMERA_FRUSTUM:
        create_camera_frustum(frame_idx, snapshot_idx, K, E)
        print(f"[INFO] Created camera frustum {snapshot_idx} at frame {frame_idx}")

# Create camera path
if SHOW_CAMERA_PATH and len(cam_centers) >= 2:
    curve_data = bpy.data.curves.new(name="CamPath", type='CURVE')
    curve_data.dimensions = '3D'
    
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(cam_centers) - 1)
    
    for i, C in enumerate(cam_centers):
        x, y, z = float(C[0]), float(C[1]), float(C[2])
        spline.points[i].co = (x, y, z, 1.0)
    
    cam_path_depth = CAMERA_PATH_RADIUS if CAMERA_PATH_RADIUS is not None else (median_z * 0.0002)
    curve_data.bevel_depth = cam_path_depth
    curve_data.bevel_resolution = 2
    
    curve_obj = bpy.data.objects.new("CamPath", curve_data)
    scene.collection.objects.link(curve_obj)
    print(f"[INFO] Camera path created with {len(cam_centers)} points")

# Object snapshots disabled (no mask available)
if SHOW_OBJECT_SNAPSHOTS:
    print("[INFO] Object snapshots disabled (no mask available)")

# ================== Setup Main Camera ==================
if num_frames > 0:
    if intrinsics.ndim == 3:
        K_cam = intrinsics[0]
    else:
        K_cam = intrinsics
    E_cam = extrinsics[0]
    setup_blender_camera(K_cam, E_cam, width_v, height_v)

# ================== Load JSON and Create Ellipsoids ==================
if Path(JSON_PATH).exists():
    print("[INFO] Loading Gaussian parameters from JSON...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    if "gaussian_params" in json_data:
        # Single-frame format with multiple objects
        print("[INFO] Detected single-frame gaussian_params format")
        gaussian_params = json_data["gaussian_params"]
        num_objects = json_data.get("num_objects", len(gaussian_params))
        
        print(f"[INFO] Found {num_objects} objects in single frame")
        
        if num_objects > 0:
            for obj_idx, (obj_id_str, obj_data) in enumerate(gaussian_params.items()):
                obj_id = int(obj_id_str)
                mean = obj_data["mean"]
                cov = obj_data["cov"]
                label = obj_data.get("label", "unknown")
                
                snapshot_idx = 0
                color_idx = obj_id - 1
                
                print(f"[INFO] Creating ellipsoid for object {obj_id} ({label})")
                create_ellipsoid_snapshot(obj_id, snapshot_idx, mean, cov, color_idx, SNAPSHOT_OPACITY)
                print(f"[INFO] Created ellipsoid for object {obj_id} ({label})")
            
            print(f"[INFO] Created {num_objects} ellipsoids")
    
    elif "frames" in json_data:
        # Multi-frame format
        print("[INFO] Detected multi-frame format")
        metadata = json_data.get("metadata", {})
        frames = json_data.get("frames", [])
        
        num_objects = metadata.get("num_objects", 0)
        num_json_frames = metadata.get("num_frames", 0)
        
        print(f"[INFO] Found {num_objects} objects across {num_json_frames} frames")
        
        if num_objects > 0 and num_json_frames > 0:
            object_params = {}
            object_color_indices = {}
            
            for frame in frames:
                frame_idx = frame["frame_index"]
                for obj in frame["objects"]:
                    obj_id = obj["object_id"]
                    color_idx = obj["color_index"]
                    mean = obj["gaussian_3d"]["mean"]
                    cov = obj["gaussian_3d"]["covariance"]
                    
                    if obj_id not in object_params:
                        object_params[obj_id] = []
                        object_color_indices[obj_id] = color_idx
                    
                    object_params[obj_id].append((frame_idx, mean, cov))
            
            for obj_id, params_list in object_params.items():
                if len(params_list) == 0:
                    continue
                
                color_idx = object_color_indices[obj_id]
                
                all_mean_positions = []
                for snapshot_idx, target_frame_idx in enumerate(snapshot_indices):
                    closest_idx = min(range(len(params_list)), 
                                    key=lambda i: abs(params_list[i][0] - target_frame_idx))
                    frame_idx, mean, cov = params_list[closest_idx]
                    all_mean_positions.append(mean)
                    
                    create_ellipsoid_snapshot(obj_id, snapshot_idx, mean, cov, color_idx, SNAPSHOT_OPACITY)
                    print(f"[INFO] Created ellipsoid snapshot {snapshot_idx} for object {obj_id}")
                
                if SHOW_ELLIPSOID_PATHS and len(all_mean_positions) >= 2:
                    create_ellipsoid_path(obj_id, all_mean_positions, color_idx)
                    print(f"[INFO] Created path for object {obj_id}")
            
            print(f"[INFO] Created {len(object_params)} objects with {NUM_SNAPSHOT_FRAMES} snapshots each")
    else:
        print("[ERROR] Unknown JSON format - missing 'gaussian_params' or 'frames'")
else:
    print(f"[WARN] JSON file not found: {JSON_PATH}")

print("[INFO] ===============================================")
print("[INFO] Visualization completed!")
print(f"[INFO] Total points: {len(background_verts)}")
print(f"[INFO] Total cameras: {len(snapshot_indices)}")
print(f"[INFO] Snapshot frames: {snapshot_indices}")
print(f"[INFO] Camera frustum: {'ENABLED' if SHOW_CAMERA_FRUSTUM else 'DISABLED'}")
print(f"[INFO] Camera frames: {'ENABLED' if SHOW_CAMERA_FRAMES else 'DISABLED'}")
print(f"[INFO] Camera path: {'ENABLED' if SHOW_CAMERA_PATH else 'DISABLED'}")
print(f"[INFO] Ellipsoid paths: {'ENABLED' if SHOW_ELLIPSOID_PATHS else 'DISABLED'}")
print("[INFO] ===============================================")
