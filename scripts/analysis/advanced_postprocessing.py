"""
Advanced post-processing for YOLO tree segmentation results.
Includes ensemble methods, filtering, and result optimization.
"""

import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from PIL import Image

class TreeSegmentationPostProcessor:
    def __init__(self, model_paths, weights=None):
        """
        Initialize post-processor with multiple models for ensemble
        
        Args:
            model_paths: List of paths to trained models
            weights: List of weights for ensemble averaging
        """
        self.models = [YOLO(path) for path in model_paths]
        self.weights = weights or [1.0] * len(model_paths)
        
    def ensemble_predict(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """Run ensemble prediction using multiple models"""
        
        all_predictions = []
        
        for i, model in enumerate(self.models):
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            
            if results and len(results) > 0 and results[0].masks is not None:
                result = results[0]
                weight = self.weights[i]
                
                for j in range(len(result.boxes)):
                    confidence = float(result.boxes.conf[j]) * weight
                    class_id = int(result.boxes.cls[j])
                    mask = result.masks.xy[j]
                    
                    all_predictions.append({
                        'confidence': confidence,
                        'class_id': class_id,
                        'mask': mask,
                        'model_id': i
                    })
        
        return self.merge_predictions(all_predictions, iou_threshold)
    
    def merge_predictions(self, predictions, iou_threshold):
        """Merge overlapping predictions using Non-Maximum Suppression"""
        
        if not predictions:
            return []
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, pred in enumerate(predictions):
            if i in used:
                continue
                
            # Find overlapping predictions
            overlapping = [i]
            for j in range(i + 1, len(predictions)):
                if j in used:
                    continue
                    
                iou = self.calculate_mask_iou(pred['mask'], predictions[j]['mask'])
                if iou > iou_threshold:
                    overlapping.append(j)
            
            # Merge overlapping predictions
            if len(overlapping) > 1:
                merged_pred = self.merge_overlapping_masks(
                    [predictions[idx] for idx in overlapping]
                )
            else:
                merged_pred = pred
            
            merged.append(merged_pred)
            used.update(overlapping)
        
        return merged
    
    def calculate_mask_iou(self, mask1, mask2):
        """Calculate IoU between two polygon masks"""
        try:
            # Convert to simple bounding box IoU for efficiency
            bbox1 = self.mask_to_bbox(mask1)
            bbox2 = self.mask_to_bbox(mask2)
            return self.bbox_iou(bbox1, bbox2)
        except:
            return 0.0
    
    def mask_to_bbox(self, mask):
        """Convert mask to bounding box"""
        if len(mask) == 0:
            return [0, 0, 0, 0]
        x_coords = mask[::2] if len(mask) % 2 == 0 else mask[:-1:2]
        y_coords = mask[1::2] if len(mask) % 2 == 0 else mask[1::2]
        return [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
    
    def bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def merge_overlapping_masks(self, overlapping_preds):
        """Merge multiple overlapping predictions"""
        # Use the prediction with highest confidence as base
        base_pred = overlapping_preds[0]
        
        # Average confidence scores
        avg_confidence = np.mean([pred['confidence'] for pred in overlapping_preds])
        
        return {
            'confidence': avg_confidence,
            'class_id': base_pred['class_id'],
            'mask': base_pred['mask']
        }
    
    def filter_small_detections(self, predictions, min_area=50):
        """Remove very small detections that are likely noise"""
        filtered = []
        
        for pred in predictions:
            mask = pred['mask']
            if len(mask) >= 6:  # At least 3 points
                # Calculate approximate area
                bbox = self.mask_to_bbox(mask)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                if area >= min_area:
                    filtered.append(pred)
        
        return filtered
    
    def apply_size_based_filtering(self, predictions, image_width, image_height):
        """Filter based on reasonable tree sizes for different resolutions"""
        filtered = []
        
        for pred in predictions:
            mask = pred['mask']
            bbox = self.mask_to_bbox(mask)
            
            # Calculate relative size
            width_ratio = (bbox[2] - bbox[0]) / image_width
            height_ratio = (bbox[3] - bbox[1]) / image_height
            
            # Filter out extremely small or large detections
            if (0.001 < width_ratio < 0.5 and 
                0.001 < height_ratio < 0.5):
                filtered.append(pred)
        
        return filtered

def create_optimized_predictions():
    """Create optimized predictions using post-processing"""
    
    # Define model paths (update these based on your trained models)
    model_paths = [
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/tree_segmentation2/weights/best.pt"
    ]
    
    # If you have multiple models, add them here:
    # model_paths.append("path/to/your/second/model.pt")
    
    processor = TreeSegmentationPostProcessor(model_paths)
    
    evaluation_images_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images"
    output_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/optimized_predictions.json"
    
    image_files = sorted([f for f in Path(evaluation_images_dir).glob("*.tif")])
    
    results_data = {"images": []}
    class_names = {0: 'individual_tree', 1: 'group_of_trees'}
    
    print(f"Processing {len(image_files)} images with advanced post-processing...")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Load image info
        img = Image.open(image_path)
        width, height = img.size
        
        # Extract metadata
        filename = image_path.name
        if filename.startswith('10cm'):
            cm_resolution = 10
            scene_type = "agriculture_plantation"
        elif filename.startswith('20cm'):
            cm_resolution = 20
            scene_type = "mixed_forest"
        elif filename.startswith('40cm'):
            cm_resolution = 40
            scene_type = "industrial_area"
        elif filename.startswith('60cm'):
            cm_resolution = 60
            scene_type = "mixed_forest"
        elif filename.startswith('80cm'):
            cm_resolution = 80
            scene_type = "industrial_area"
        else:
            cm_resolution = 10
            scene_type = "mixed_forest"
        
        # Run ensemble prediction with post-processing
        predictions = processor.ensemble_predict(
            str(image_path),
            conf_threshold=0.2,  # Lower threshold for more detections
            iou_threshold=0.4    # Lower IoU for less aggressive NMS
        )
        
        # Apply additional filtering
        predictions = processor.filter_small_detections(predictions, min_area=25)
        predictions = processor.apply_size_based_filtering(predictions, width, height)
        
        # Convert to required format
        annotations = []
        for pred in predictions:
            class_name = class_names.get(pred['class_id'], 'individual_tree')
            segmentation = pred['mask'].flatten().tolist()
            
            if len(segmentation) >= 6:
                annotation = {
                    "class": class_name,
                    "confidence_score": round(pred['confidence'], 2),
                    "segmentation": [round(coord, 1) for coord in segmentation]
                }
                annotations.append(annotation)
        
        image_result = {
            "file_name": filename,
            "width": width,
            "height": height,
            "cm_resolution": cm_resolution,
            "scene_type": scene_type,
            "annotations": annotations
        }
        
        results_data["images"].append(image_result)
    
    # Save optimized results
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    total_annotations = sum(len(img["annotations"]) for img in results_data["images"])
    print(f"\nOptimized predictions saved to: {output_path}")
    print(f"Total annotations: {total_annotations}")
    
    return output_path

if __name__ == "__main__":
    print("🔧 Advanced Post-Processing for Tree Segmentation")
    print("=================================================")
    
    try:
        output_file = create_optimized_predictions()
        print(f"\n✅ Optimized predictions ready: {output_file}")
        print("\nImprovements applied:")
        print("- Ensemble prediction (if multiple models)")
        print("- Advanced NMS for overlapping detections")
        print("- Size-based filtering")
        print("- Small detection removal")
        print("- Confidence optimization")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your model files exist and are accessible.")