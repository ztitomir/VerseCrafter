"""
3D Gaussian Fitting Module

This module fits 3D Gaussian distributions to segmented objects in a single image.
It takes depth maps and segmentation masks as input and outputs 3D Gaussian parameters
suitable for 3D scene representation and rendering.

Workflow:
1. Load depth map and camera intrinsics from NPZ file
2. For each segmentation mask, extract corresponding 3D points from depth
3. Fit a 3D Gaussian distribution to the point cloud
4. Project and visualize 3D Gaussians onto 2D image plane
5. Save parameters to JSON for downstream processing
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
from scipy.stats import chi2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_point_cloud_from_depth(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert depth map to 3D point cloud in world coordinates.
    
    Uses camera intrinsics to unproject 2D pixels with depth to 3D camera coordinates,
    then transforms to world space using camera extrinsic matrix.
    
    Args:
        depth (torch.Tensor): A 2D tensor of shape (H, W) representing the depth map.
                              Each value corresponds to the depth of a pixel in the camera's view.
        intrinsic (torch.Tensor): A 3x3 tensor representing the camera's intrinsic matrix.
        extrinsic (torch.Tensor): A 4x4 tensor representing the camera's extrinsic matrix(w2c).
        mask (Optional[torch.Tensor]): A 2D tensor of shape (H, W) representing a binary mask.
                                       If provided, only points corresponding to non-zero mask values are included.
                                       Defaults to None.
    Returns:
        torch.Tensor: A 2D tensor of shape (N, 3), where N is the number of valid points.
                      Each row represents the (x, y, z) coordinates of a point in world space.
                      Points with zero depth or excluded by the mask are omitted.
    """
    h, w = depth.shape
    device = depth.device
    
    # Create pixel coordinate grids for the image
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Unproject: convert 2D pixel coordinates to 3D camera coordinates using depth
    ones = torch.ones_like(x)
    xy_homogeneous = torch.stack([x, y, ones], dim=0).reshape(3, -1)
    
    K_inv = torch.inverse(intrinsic)
    pts3d_cam = K_inv @ xy_homogeneous
    pts3d_cam = pts3d_cam * depth.reshape(-1)
    pts3d_cam = torch.cat([pts3d_cam, torch.ones(1, pts3d_cam.shape[1], device=device)], dim=0)
    
    # Transform from camera to world coordinates
    c2w = torch.inverse(extrinsic)
    pts3d_world_homogeneous = c2w @ pts3d_cam
    pts3d_world = pts3d_world_homogeneous[:3, :].T  # (H*W, 3)
    
    # Filter points: keep only those in the mask or with valid depth
    if mask is not None:
        mask_flat = mask.reshape(-1).bool()
        pts3d_world = pts3d_world[mask_flat]
    else:
        depth_valid = depth.reshape(-1) > 0
        pts3d_world = pts3d_world[depth_valid]
    
    return pts3d_world


def fit_3d_gaussian(
    points: torch.Tensor,
    device: str = 'cuda'
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fit a 3D Gaussian distribution to a point cloud.
    
    Computes mean and covariance matrix of the point cloud.
    Adds small regularization to ensure positive definiteness.
    
    Args:
        points: Point cloud (N, 3)
        device: Computation device
    
    Returns:
        mean, covariance tensors or (None, None) if fitting fails
    """
    if len(points) == 0:
        logger.warning("Empty point cloud, cannot fit Gaussian")
        return None, None
    
    if len(points) < 3:
        logger.warning(f"Too few points ({len(points)}), cannot reliably fit Gaussian")
        return None, None
    
    try:
        # Compute mean of point cloud
        mean = torch.mean(points, dim=0)
        
        # Center points around mean
        points_centered = points - mean.unsqueeze(0)
        
        # Compute covariance matrix
        cov = (points_centered.T @ points_centered) / (len(points) - 1)
        
        # Add small regularization to ensure positive definite
        cov = cov + 1e-6 * torch.eye(3, device=device, dtype=cov.dtype)
        
        return mean, cov
        
    except Exception as e:
        logger.error(f"Gaussian fitting failed: {e}")
        return None, None


def load_mask(mask_path: str, device: str = 'cuda', erode_kernel_size: int = 5) -> torch.Tensor:
    """Load and preprocess segmentation mask.
    
    Converts mask to binary and applies morphological erosion to remove boundary noise.
    """
    try:
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img, dtype=np.uint8)
        
        # Threshold to binary mask
        mask_binary = (mask_np > 127).astype(np.uint8) * 255
        
        # Erode mask to remove boundary noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
        
        mask_tensor = torch.from_numpy(mask_eroded > 127).to(device)
        return mask_tensor
    except Exception as e:
        logger.error(f"Failed to load mask {mask_path}: {e}")
        return None


def get_object_color(obj_id: int, obj_id_to_color_idx: Dict[int, int], device: str = 'cuda') -> torch.Tensor:
    """Get distinct color for object visualization using tab10 colormap."""
    # Use tab20 for high contrast and saturation (20 bright, distinct colors)
    cmap = matplotlib.colormaps['tab20']
    color_idx = obj_id_to_color_idx.get(obj_id, 0)
    color_rgb = cmap(color_idx % 20)[:3]  # 0~1
    color_tensor = torch.tensor(color_rgb, dtype=torch.float32, device=device)
    return color_tensor

def project_gaussian_to_2d(
    mean: torch.Tensor,
    cov: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Project 3D Gaussian to 2D image plane with ROI-based optimization.
    
    Uses bounding box optimization to only compute density within the ROI,
    avoiding full-image rasterization.
    
    Args:
        mean: (3,) 3D Gaussian mean in world coordinates
        cov: (3, 3) 3D covariance matrix in world coordinates
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera extrinsic matrix (world to camera)
        image_size: (width, height)
        device: computation device
    
    Returns:
        density: (height, width) 2D density map
        mahalanobis_dist_sq: (height, width) squared Mahalanobis distance map
        z_depth: depth of Gaussian center in camera coordinates
    """
    width, height = image_size
    
    # Initialize output maps
    density = torch.zeros((height, width), device=device, dtype=torch.float32)
    mahalanobis_dist_sq = torch.zeros((height, width), device=device, dtype=torch.float32) + float('inf')
    
    # 1. Transform from world to camera coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    mean_cam = R @ mean + t
    z_depth = mean_cam[2].item()
    
    # Near plane culling
    if z_depth <= 0.2:
        return density, mahalanobis_dist_sq, z_depth
    
    # 2. Project mean to image plane
    mean_2d_homo = intrinsic @ mean_cam
    mean_2d = mean_2d_homo[:2] / mean_2d_homo[2]
    
    u, v = mean_2d[0].item(), mean_2d[1].item()
    
    # Screen bounds check with margin
    margin = 50
    if u < -margin or u > width + margin or v < -margin or v > height + margin:
        return density, mahalanobis_dist_sq, z_depth
        
    # 3. Project covariance (EWA Splatting)
    cov_cam = R @ cov @ R.T
    
    # Jacobian of perspective projection
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    x, y, z = mean_cam[0], mean_cam[1], mean_cam[2]
    
    J = torch.tensor([
        [fx / z, 0, -(fx * x) / (z * z)],
        [0, fy / z, -(fy * y) / (z * z)]
    ], device=device, dtype=torch.float32)
    
    # Project 3D covariance to 2D image space
    cov_2d = J @ cov_cam @ J.T
    
    # Add regularization for numerical stability
    cov_2d = cov_2d + 1e-4 * torch.eye(2, device=device)
    
    # Compute inverse and determinant for Mahalanobis distance
    try:
        inv_cov_2d = torch.linalg.inv(cov_2d)
        det_cov_2d = torch.linalg.det(cov_2d)
    except RuntimeError:
        return density, mahalanobis_dist_sq, z_depth
        
    if det_cov_2d <= 0:
        return density, mahalanobis_dist_sq, z_depth
        
    # Compute 3-sigma radius for efficient ROI-based rasterization
    radius = 3.0 * torch.sqrt(torch.max(torch.diag(cov_2d)))
    radius_int = int(math.ceil(radius.item()))
    
    # Determine region of interest bounds
    mu_x_int, mu_y_int = int(u), int(v)
    min_x = max(0, mu_x_int - radius_int)
    max_x = min(width, mu_x_int + radius_int + 1)
    min_y = max(0, mu_y_int - radius_int)
    max_y = min(height, mu_y_int + radius_int + 1)
    
    if min_x >= max_x or min_y >= max_y:
        return density, mahalanobis_dist_sq, z_depth
        
    # Generate local pixel grid within ROI
    x_range = torch.arange(min_x, max_x, device=device, dtype=torch.float32)
    y_range = torch.arange(min_y, max_y, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy')
    local_pixel_coords = torch.stack([grid_x, grid_y], dim=-1)
    
    # Compute local difference from Gaussian mean
    diff = local_pixel_coords - mean_2d
    
    # Compute squared Mahalanobis distance: d^T @ inv_cov @ d
    mahal_dist_local = torch.einsum('ijk,kl,ijl->ij', diff, inv_cov_2d, diff)
    
    # Compute 2D Gaussian probability density (PDF)
    coeff = 1.0 / (2 * math.pi * torch.sqrt(det_cov_2d))
    pdf_local = coeff * torch.exp(-0.5 * mahal_dist_local)
    
    # Fill back to full image
    density[min_y:max_y, min_x:max_x] = pdf_local
    mahalanobis_dist_sq[min_y:max_y, min_x:max_x] = mahal_dist_local
    
    return density, mahalanobis_dist_sq, z_depth


def visualize_gaussian_projections(
    gaussian_params: Dict[int, Dict],
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_size: Tuple[int, int],
    output_path: Path,
    probability_threshold: float = 0.97,
    device: str = 'cuda',
    input_image_path: Optional[str] = None
) -> Dict[int, int]:
    """Visualize 3D Gaussian projections on 2D image with alpha blending.
    
    Projects 3D Gaussians to 2D image space and renders them with proper occlusion handling
    using depth sorting. Optionally overlays on input image.
    
    Args:
        gaussian_params: Dictionary mapping object IDs to Gaussian parameters
        intrinsic: Camera intrinsic matrix
        extrinsic: Camera extrinsic matrix (world to camera)
        image_size: (width, height) of output image
        output_path: Directory to save visualization results
        probability_threshold: Probability mass to include in ellipse (default 0.97)
        device: Computation device
        input_image_path: Optional input image for overlay visualization
    
    Returns:
        Dictionary mapping object IDs to color indices
    """
    width, height = image_size
    
    # Convert numpy arrays to tensors if needed
    intrinsic_t = (torch.from_numpy(intrinsic).float().to(device) 
                   if isinstance(intrinsic, np.ndarray) 
                   else intrinsic.to(device))
                   
    extrinsic_t = (torch.from_numpy(extrinsic).float().to(device) 
                   if isinstance(extrinsic, np.ndarray) 
                   else extrinsic.to(device))
    
    # Chi-squared threshold for 2D Gaussian confidence ellipse
    mahalanobis_threshold = chi2.ppf(probability_threshold, df=2)
    logger.info(
        f"Probability threshold: {probability_threshold*100:.1f}% -> "
        f"Mahalanobis threshold: {mahalanobis_threshold:.4f}"
    )
    
    # Project all Gaussians
    gaussian_projections = []
    obj_id_to_color_idx = {}
    next_color_idx = 0
    
    for obj_id, params in sorted(gaussian_params.items()):
        mean = np.array(params['mean'])
        cov = np.array(params['cov'])
        label = params.get('label', f'object_{obj_id}')
        
        mean_t = torch.from_numpy(mean).float().to(device)
        cov_t = torch.from_numpy(cov).float().to(device)
        
        # Project 3D Gaussian to 2D image plane
        density, mahalanobis_dist_sq, z_depth = project_gaussian_to_2d(
            mean_t, cov_t, intrinsic_t, extrinsic_t, image_size, device
        )
        
        if z_depth > 0:
            if obj_id not in obj_id_to_color_idx:
                obj_id_to_color_idx[obj_id] = next_color_idx
                next_color_idx += 1
            
            color_rgb = get_object_color(obj_id, obj_id_to_color_idx, device)
            
            gaussian_projections.append({
                'density': density,
                'mahalanobis_dist_sq': mahalanobis_dist_sq,
                'color_rgb': color_rgb,
                'z_depth': z_depth,
                'obj_id': obj_id,
                'label': label
            })
    
    # Sort by depth (back to front) for proper occlusion handling
    gaussian_projections.sort(key=lambda x: x['z_depth'], reverse=True)
    
    # Render Gaussians with alpha blending
    rgb_frame = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    mask_frame = torch.zeros((height, width), dtype=torch.float32, device=device)
    
    for proj in gaussian_projections:
        density = proj['density']
        mahalanobis_dist_sq = proj['mahalanobis_dist_sq']
        
        # Generate mask based on Mahalanobis distance (confidence ellipse)
        mask_pixels = mahalanobis_dist_sq <= mahalanobis_threshold
        mask_frame = torch.maximum(mask_frame, mask_pixels.float())
        
        # Compute alpha based on normalized density for smooth visualization
        density_max = density.max()
        if density_max > 0:
            alpha_soft = (density / density_max).clamp(0.0, 1.0)
        else:
            alpha_soft = torch.zeros_like(density)
        
        # Alpha blending: composite color with existing frame
        color_rgb = proj['color_rgb'].view(1, 1, 3)
        alpha_3d = alpha_soft.unsqueeze(-1)
        rgb_frame = color_rgb * alpha_3d + rgb_frame * (1 - alpha_3d)
    
    # Save Gaussian projection visualization
    rgb_frame_uint8 = (rgb_frame.clamp(0, 1) * 255).to(torch.uint8)
    gaussian_proj_img = rgb_frame_uint8.cpu().numpy().astype(np.uint8)
    gaussian_proj_pil = Image.fromarray(gaussian_proj_img, mode='RGB')
    gaussian_proj_pil.save(output_path / 'gaussian_projection.png')
    logger.info(f"✓ Saved Gaussian projection to {output_path / 'gaussian_projection.png'}")
    
    # Optionally overlay on input image
    if input_image_path is not None:
        try:
            if isinstance(input_image_path, str):
                input_img_pil = Image.open(input_image_path).convert('RGB')
            else:
                input_img_pil = input_image_path
            
            if input_img_pil.size != (width, height):
                logger.warning(
                    f"Input image size {input_img_pil.size} doesn't match expected "
                    f"{(width, height)}, resizing"
                )
                input_img_pil = input_img_pil.resize((width, height), Image.Resampling.LANCZOS)
            
            input_img_np = np.array(input_img_pil, dtype=np.uint8)
            
            mask_np = mask_frame.cpu().numpy().astype(np.float32)
            rgb_proj_np = gaussian_proj_img.astype(np.float32)
            
            # Blend Gaussian projection with input image
            alpha_proj = np.expand_dims(mask_np, axis=-1)
            blend_factor = 0.7
            gaussian_overlay = (rgb_proj_np * alpha_proj * blend_factor + 
                              input_img_np * (1 - alpha_proj * blend_factor)).astype(np.uint8)
            
            gaussian_overlay_pil = Image.fromarray(gaussian_overlay, mode='RGB')
            gaussian_overlay_pil.save(output_path / 'gaussian_overlay_on_image.png')
            logger.info(f"✓ Saved Gaussian overlay to {output_path / 'gaussian_overlay_on_image.png'}")
            
        except Exception as e:
            logger.warning(f"Failed to generate overlay image: {e}")
    
    return obj_id_to_color_idx


def tensor_to_json_serializable(t):
    """Convert tensor or numpy array to JSON-serializable format."""
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy().tolist()
    elif isinstance(t, np.ndarray):
        return t.tolist()
    else:
        return t


def process_single_image(
    npz_path: str,
    masks_dir: str,
    output_dir: str,
    device: str = 'cuda',
    input_image_path: Optional[str] = None,
    enable_visualization: bool = True
):
    """Process a single image: fit Gaussians for each segmented object.
    
    Main processing pipeline:
    1. Load depth map and camera intrinsics from NPZ
    2. For each segmentation mask, extract 3D points and fit Gaussian
    3. Generate 2D Gaussian projections for visualization
    4. Save all results (parameters and images) to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load NPZ file containing depth and camera intrinsics
    logger.info(f"Loading NPZ file: {npz_path}")
    try:
        data = np.load(npz_path)

        depth_np = data['depth'].astype(np.float32)  # (H, W)
        intrinsic_np = data['intrinsic'].astype(np.float32)  # (3, 3)

        if depth_np.ndim == 3:
            depth_np = depth_np[0]  # Reduce to (H, W)

        if intrinsic_np.ndim == 3:
            intrinsic_np = intrinsic_np[0]  # Reduce to (3, 3)
        
        logger.info(f"✓ Depth map shape: {depth_np.shape}")
        logger.info(f"✓ Camera intrinsics shape: {intrinsic_np.shape}")
    except Exception as e:
        logger.error(f"Failed to load NPZ: {e}")
        return
    
    # Convert to tensors
    depth = torch.from_numpy(depth_np).to(device)
    intrinsic = torch.from_numpy(intrinsic_np).to(device)
    
    # Assume camera at origin with no rotation
    extrinsic = torch.eye(4, device=device, dtype=torch.float32)
    
    h, w = depth.shape
    logger.info(f"Image resolution: {w}x{h}")
    
    # Extract focal lengths and principal point
    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()
    
    logger.info(f"Original intrinsic: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    
    # Normalize intrinsics if they are in normalized format (< 10)
    if abs(fx) < 10 or abs(fy) < 10:
        intrinsic[0, 0] *= w   # fx * width
        intrinsic[1, 1] *= h   # fy * height
        intrinsic[0, 2] *= w   # cx * width
        intrinsic[1, 2] *= h   # cy * height
    
    # Find and load all segmentation masks
    masks_path = Path(masks_dir)
    if not masks_path.exists():
        logger.error(f"Masks directory does not exist: {masks_dir}")
        return
    
    mask_files = sorted([f for f in masks_path.glob("mask_*.png")])
    logger.info(f"Found {len(mask_files)} mask files")
    
    if len(mask_files) == 0:
        logger.error("No mask files found")
        return
    
    gaussian_params = {}  # {object_id: {mean, cov, num_points, ...}}
    
    # Process each segmentation mask
    for mask_file in mask_files:
        try:
            stem = mask_file.stem
            parts = stem.split('_')
            obj_id = int(parts[1])
            obj_label = '_'.join(parts[2:]) if len(parts) > 2 else f"object_{obj_id}"
            
            logger.info(f"\nProcessing: {obj_label} (ID: {obj_id})")
            
            # Load mask
            mask = load_mask(str(mask_file), device=device)
            if mask is None:
                logger.warning(f"Skipping invalid mask: {mask_file.name}")
                continue
            
            num_pixels = mask.sum().item()
            
            # Extract 3D point cloud from depth map using mask
            points = get_point_cloud_from_depth(depth, intrinsic, extrinsic, mask)
            num_points = len(points)
            
            if num_points < 10:
                logger.warning(f"Too few points, skipping this object")
                continue
            
            # Fit 3D Gaussian distribution to points
            mean, cov = fit_3d_gaussian(points, device)
            
            if mean is None or cov is None:
                logger.warning(f"Gaussian fitting failed, skipping this object")
                continue
            
            # Compute Gaussian statistics
            eigvals = torch.linalg.eigvalsh(cov)
            trace = cov.trace()
            
            logger.info(f"  Mean: {mean.cpu().numpy()}")
            logger.info(f"  Covariance trace: {trace.item():.6f}")
            
            # Save parameters to dictionary
            gaussian_params[obj_id] = {
                "label": obj_label,
                "mean": tensor_to_json_serializable(mean),
                "cov": tensor_to_json_serializable(cov),
                "num_points": num_points,
                "num_mask_pixels": num_pixels,
                "eigvals": tensor_to_json_serializable(eigvals),
                "trace": trace.item()
            }
            
        except Exception as e:
            logger.error(f"Failed to process mask {mask_file.name}: {e}")
            continue
    
    # Generate 2D visualizations if enabled
    obj_id_to_color_idx = {}
    if enable_visualization:
        if len(gaussian_params) > 0:
            try:
                obj_id_to_color_idx = visualize_gaussian_projections(
                    gaussian_params=gaussian_params,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    image_size=(w, h),
                    output_path=output_path,
                    probability_threshold=0.97,
                    device=device,
                    input_image_path=input_image_path
                )
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        else:
            logger.warning("No objects detected, skipping visualization")
    else:
        # Assign colors even if visualization is disabled
        next_color_idx = 0
        for obj_id in sorted(gaussian_params.keys()):
            obj_id_to_color_idx[obj_id] = next_color_idx
            next_color_idx += 1
    
    # Save all results to JSON
    output_data = {
        "image_info": {
            "resolution": [int(w), int(h)],
            "depth_shape": depth_np.shape[:2]
        },
        "camera_info": {
            "intrinsic": tensor_to_json_serializable(intrinsic),
            "extrinsic": tensor_to_json_serializable(extrinsic)
        },
        "gaussian_params": gaussian_params,
        "num_objects": len(gaussian_params),
        "obj_id_to_color_idx": obj_id_to_color_idx
    }
    
    output_json = output_path / "gaussian_params.json"
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"✓ Saved parameters to {output_json}")
    
    return output_data


def main():
    """Parse arguments and run the 3D Gaussian fitting pipeline."""
    parser = argparse.ArgumentParser(
        description="Fit 3D Gaussians from single-image NPZ and segmentation masks"
    )
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to NPZ file (containing depth and intrinsic)'
    )
    parser.add_argument(
        '--masks_dir',
        type=str,
        required=True,
        help='Path to segmentation masks directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gaussian_results',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Computation device (cuda/cpu)'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='Input RGB image path (optional) for overlay visualization'
    )
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization (only save JSON parameters)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug-level logging'
    )
    
    args = parser.parse_args()

    # Configure logging level
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input files
    if not Path(args.npz_path).exists():
        logger.error(f"NPZ file does not exist: {args.npz_path}")
        return
    
    if not Path(args.masks_dir).exists():
        logger.error(f"Masks directory does not exist: {args.masks_dir}")
        return
    
    if args.image_path is not None and not Path(args.image_path).exists():
        logger.error(f"Image file does not exist: {args.image_path}")
        return
    
    # Run processing pipeline
    result = process_single_image(
        npz_path=args.npz_path,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        device=args.device,
        input_image_path=args.image_path,
        enable_visualization=not args.no_visualization
    )
    
    logger.info("=" * 80)
    logger.info("✓ Fitting 3D Gaussians complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
