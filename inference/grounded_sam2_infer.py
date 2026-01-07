"""
Image Segmentation with Grounded SAM2
This script provides an implementation for image segmentation using the Grounded SAM2 model. 
It includes functionality for loading models, segmenting images based on text prompts, 
visualizing results, and saving outputs.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add Grounded-SAM-2 directory to path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Grounded-SAM-2'))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
grounded_sam2_path = os.path.join(project_root, 'third_party/Grounded-SAM-2')
if grounded_sam2_path not in sys.path:
    sys.path.insert(0, grounded_sam2_path)

import torch
import numpy as np
import cv2
import supervision as sv
from PIL import Image

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)
 


class ImageSegmenter:
    """Image segmentation using Grounded SAM2 (SAM2 with text prompt guidance).
    
    Combines GroundingDINO for text-guided object detection with SAM2 for precise segmentation.
    Workflow: Text prompt -> GroundingDINO detections -> SAM2 segmentation -> Masks
    """
    
    def __init__(
        self,
        grounding_dino_config: str = None,
        grounding_dino_checkpoint: str = None,
        sam2_checkpoint: str = None,
        sam2_model_cfg: str = None,
        device: str = None
    ):
        """Initialize the ImageSegmenter with GroundingDINO and SAM2 models.
        
        Args:
            grounding_dino_config: Path to GroundingDINO config file
            grounding_dino_checkpoint: Path to GroundingDINO weights
            sam2_checkpoint: Path to SAM2 model weights
            sam2_model_cfg: Path to SAM2 model configuration
            device: Computation device (cuda/cpu). Auto-detects if None.
        """
        # Set default paths relative to Grounded-SAM-2 directory
        # Get the absolute path of the inference directory (where this script is)
        inference_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(inference_dir)
        grounded_sam2_dir = os.path.join(project_root, 'third_party/Grounded-SAM-2')
        
        # Set default checkpoint paths if not provided
        if grounding_dino_config is None:
            grounding_dino_config = os.path.join(
                grounded_sam2_dir, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            )
        if grounding_dino_checkpoint is None:
            grounding_dino_checkpoint = os.path.join(
                grounded_sam2_dir, "gdino_checkpoints/groundingdino_swint_ogc.pth"
            )
        if sam2_checkpoint is None:
            sam2_checkpoint = os.path.join(
                grounded_sam2_dir, "checkpoints/sam2.1_hiera_large.pt"
            )
        if sam2_model_cfg is None:
            sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load GroundingDINO model for text-guided object detection
        logger.info("Loading GroundingDINO model...")
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )
        
        # Load SAM2 model for segmentation
        logger.info("Loading SAM2 model...")
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("✓ Models loaded successfully")
    
    def segment_image(
        self,
        image_path: str,
        text_prompt: str = "person . car . dog .",
        box_threshold: float = 0.4,
        text_threshold: float = 0.25,
        keep_topk: int = 6,
        min_area_ratio: float = 0.0,
        max_area_ratio: float = 1.0
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Segment image using text prompt.
        
        Pipeline:
        1. Load image and run GroundingDINO detection with text prompt
        2. Keep top-K objects by area
        3. Run SAM2 segmentation using detected boxes
        4. Filter masks by area ratio (optional)
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt describing objects to detect (e.g., "person . car .")
            box_threshold: Confidence threshold for object detection
            text_threshold: Text matching threshold for GroundingDINO
            keep_topk: Keep only top-K largest objects
            min_area_ratio: Minimum mask area ratio (0.0-1.0)
            max_area_ratio: Maximum mask area ratio (0.0-1.0)
        
        Returns:
            Tuple of (original_image, detection_dict, mask_array)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        logger.info(f"Image size: {w}x{h}")
        
        # Run GroundingDINO detection with text prompt
        logger.info(f"Running detection with text prompt: {text_prompt}")
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=True
        )
        
        if boxes.shape[0] == 0:
            logger.warning("No objects detected!")
            return image_source, {}, np.array([])
        
        logger.info(f"Detected {boxes.shape[0]} objects")
        
        # Convert box coordinates and filter by area
        boxes = boxes * torch.Tensor([w, h, w, h])
        areas = boxes[:, 2] * boxes[:, 3]  # Width * Height for each box
        topk = min(keep_topk, boxes.shape[0])
        topk_idx = torch.topk(areas, topk).indices
        
        # Keep only top-K objects
        boxes = boxes[topk_idx]
        confidences = confidences[topk_idx]
        labels = [labels[i] for i in topk_idx.tolist()]
        
        logger.info(f"Keeping top-{topk} objects by area")
        
        # Convert box format and prepare for SAM2
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences_np = confidences.numpy().tolist()
        
        # Run SAM2 segmentation
        logger.info("Running SAM2 segmentation...")
        self.image_predictor.set_image(image_source)
        
        # Enable TF32 for faster computation on Ampere+ GPUs
        try:
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except:
            pass
        
        # Predict masks using bounding boxes with mixed precision
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        
        # Squeeze batch dimension if present
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Apply area ratio filtering if enabled
        if min_area_ratio > 0.0 or max_area_ratio < 1.0:
            masks_tensor = torch.from_numpy(masks) if isinstance(masks, np.ndarray) else masks
            if masks_tensor.ndim == 4:
                masks_tensor = masks_tensor.squeeze(1)
            
            # Compute mask area ratios
            image_area = h * w
            mask_areas = masks_tensor.flatten(1).float().sum(dim=1)
            mask_area_ratios = mask_areas / image_area
            logger.info(f"Mask area ratios: {mask_area_ratios.tolist()}")
            
            # Filter masks within area ratio range
            valid_mask = (mask_area_ratios >= min_area_ratio) & (mask_area_ratios <= max_area_ratio)
            
            if not torch.any(valid_mask):
                logger.warning(f"No masks within [{min_area_ratio}, {max_area_ratio}] area ratio range!")
                return image_source, {}, np.array([])
            
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()
            masks = masks[valid_indices]
            input_boxes = input_boxes[valid_indices]
            confidences_np = [confidences_np[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            logger.info(f"Filtered masks: {len(valid_indices)}/{len(mask_area_ratios)} masks passed area ratio filter")
        
        # Convert masks to uint8 format
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy().astype(np.uint8) * 255
        else:
            masks_np = masks.astype(np.uint8) * 255
        
        logger.info(f"✓ Segmentation complete, generated {len(masks_np)} masks")
        
        # Prepare detection results dictionary
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                scores_list = scores.cpu().numpy().tolist()
            else:
                scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores
        else:
            scores_list = None
        
        detections = {
            "labels": labels,
            "boxes": input_boxes,
            "confidences": confidences_np,
            "masks": masks_np,
            "scores": scores_list
        }
        
        return image_source, detections, masks_np
    
    def visualize_results(
        self,
        image_source: np.ndarray,
        detections: Dict
    ) -> np.ndarray:
        """Visualize segmentation results with bounding boxes, labels, and masks.
        
        Overlays detected objects with boxes, text labels, and colored masks on the image.
        
        Args:
            image_source: Original image array
            detections: Detection results dictionary with labels, boxes, masks
        
        Returns:
            Annotated image with visualization overlays
        """
        if not detections or "labels" not in detections:
            logger.warning("No detections to visualize")
            return image_source
        
        labels = detections["labels"]
        masks = detections["masks"]
        boxes = detections["boxes"]
        
        # Convert masks to binary format
        masks_binary = (masks > 127).astype(bool)
        
        # Create detection object for supervision visualization
        sv_detections = sv.Detections(
            xyxy=boxes,  
            mask=masks_binary, 
            class_id=np.arange(len(labels), dtype=np.int32)
        )
        
        annotated_frame = image_source.copy()
        
        # Annotate bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections
        )
        
        # Annotate labels with text
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            labels=[f"{i+1}: {label}" for i, label in enumerate(labels)]
        )
        
        # Annotate segmentation masks
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections
        )
        
        logger.info(f"✓ Visualization complete with {len(labels)} objects")
        
        return annotated_frame
    
    def save_results(
        self,
        output_dir: str,
        image_source: np.ndarray,
        detections: Dict,
        vis_image: np.ndarray,
        image_name: str = "result"
    ) -> Dict:
        """Save segmentation results to disk.
        
        Saves visualization image, individual masks, combined mask, and annotations.
        
        Args:
            output_dir: Output directory path
            image_source: Original image array
            detections: Detection results with masks and labels
            vis_image: Visualized image with overlays
            image_name: Base name for output files
        
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        vis_path = output_path / f"{image_name}_visualization.png"
        Image.fromarray(vis_image).save(vis_path)
        logger.info(f"✓ Saved visualization to {vis_path}")
        
        # Save individual masks
        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        if "masks" in detections:
            for idx, mask in enumerate(detections["masks"]):
                mask_path = masks_dir / f"mask_{idx+1:02d}_{detections['labels'][idx]}.png"
                Image.fromarray(mask).save(mask_path)
            logger.info(f"✓ Saved {len(detections['masks'])} masks to {masks_dir}")
        
        # Save combined mask (one channel per object)
        h, w = image_source.shape[:2]
        mask_combined = np.zeros((h, w), dtype=np.uint8)
        for idx, mask in enumerate(detections["masks"]):
            mask_combined[mask > 127] = idx + 1
        
        combined_mask_path = output_path / f"{image_name}_mask_combined.png"
        Image.fromarray(mask_combined).save(combined_mask_path)
        logger.info(f"✓ Saved combined mask to {combined_mask_path}")
        
        # Save annotations file
        info_path = output_path / f"{image_name}_annotations.txt"
        with open(info_path, 'w') as f:
            f.write(f"Image: {image_source.shape}\n")
            f.write(f"Total Objects: {len(detections['labels'])}\n")
            f.write("\nDetections:\n")
            for idx, (label, conf, box) in enumerate(
                zip(detections["labels"], detections["confidences"], detections["boxes"])
            ):
                f.write(f"  {idx+1}. {label} (conf: {conf:.3f}), box: {box}\n")
        logger.info(f"✓ Saved annotations to {info_path}")
        
        return {
            "visualization": str(vis_path),
            "masks_dir": str(masks_dir),
            "combined_mask": str(combined_mask_path),
            "annotations": str(info_path)
        }


def main():
    """Main entry point: parse arguments and run image segmentation."""
    parser = argparse.ArgumentParser(
        description="Segment image using Grounded SAM2 with visualization"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug-level logging'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--text_prompt',
        type=str,
        default="person . car . dog . cat .",
        help='Text prompt for detection (format: "class1 . class2 . ...")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./segmentation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--box_threshold',
        type=float,
        default=0.4,
        help='Box confidence threshold for GroundingDINO'
    )
    parser.add_argument(
        '--text_threshold',
        type=float,
        default=0.25,
        help='Text matching threshold for GroundingDINO'
    )
    parser.add_argument(
        '--keep_topk',
        type=int,
        default=6,
        help='Keep top-K largest objects'
    )
    parser.add_argument(
        '--min_area_ratio',
        type=float,
        default=0.005,
        help='Minimum mask area ratio (0.0-1.0). Example: 0.005 means 0.5%% of image area'
    )
    parser.add_argument(
        '--max_area_ratio',
        type=float,
        default=0.2,
        help='Maximum mask area ratio (0.0-1.0). Example: 0.2 means 20%% of image area'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input image path
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        return
    
    # Initialize segmenter
    segmenter = ImageSegmenter(device=args.device)
    
    # Run segmentation
    image_source, detections, masks = segmenter.segment_image(
        image_path=args.image_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        keep_topk=args.keep_topk,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio
    )
    
    if not detections or "labels" not in detections:
        logger.error("Segmentation failed or no objects detected")
        return
    
    # Generate visualization
    vis_image = segmenter.visualize_results(image_source, detections)
    
    # Save results
    image_name = Path(args.image_path).stem
    save_paths = segmenter.save_results(
        args.output_dir,
        image_source,
        detections,
        vis_image,
        image_name=image_name
    )
    
    logger.info("=" * 80)
    logger.info("✓ Segmentation complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
