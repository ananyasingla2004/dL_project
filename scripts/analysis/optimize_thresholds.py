"""
Hyperparameter optimization script for YOLO tree segmentation.
This script tests different confidence and IoU thresholds to find optimal settings.
"""

from ultralytics import YOLO
import json
import numpy as np
from pathlib import Path

def evaluate_thresholds():
    """Test different confidence and IoU thresholds to optimize performance"""
    
    # Load your trained model
    model_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/tree_segmentation2/weights/best.pt"
    model = YOLO(model_path)
    
    # Test on a subset of evaluation images for speed
    test_images = [
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images/10cm_evaluation_1.tif",
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images/10cm_evaluation_10.tif",
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images/20cm_evaluation_39.tif",
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images/40cm_evaluation_76.tif",
        "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images/60cm_evaluation_101.tif"
    ]
    
    # Threshold ranges to test
    confidence_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    iou_thresholds = [0.3, 0.4, 0.45, 0.5, 0.6]
    
    results = []
    
    print("Testing threshold combinations...")
    print("Conf\tIoU\tDetections\tAvg_Conf")
    print("-" * 40)
    
    for conf_thresh in confidence_thresholds:
        for iou_thresh in iou_thresholds:
            total_detections = 0
            total_confidence = 0
            
            for image_path in test_images:
                if Path(image_path).exists():
                    # Run inference with current thresholds
                    results_inference = model.predict(
                        source=image_path,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        save=False,
                        verbose=False
                    )
                    
                    if results_inference and len(results_inference) > 0:
                        result = results_inference[0]
                        if result.boxes is not None:
                            detections = len(result.boxes)
                            if detections > 0:
                                avg_conf = float(result.boxes.conf.mean())
                                total_detections += detections
                                total_confidence += avg_conf * detections
            
            avg_confidence = total_confidence / max(total_detections, 1)
            
            results.append({
                'conf_thresh': conf_thresh,
                'iou_thresh': iou_thresh,
                'total_detections': total_detections,
                'avg_confidence': avg_confidence
            })
            
            print(f"{conf_thresh:.2f}\t{iou_thresh:.2f}\t{total_detections}\t{avg_confidence:.3f}")
    
    # Find optimal thresholds
    # Sort by total detections (more detections often better for tree counting)
    results.sort(key=lambda x: x['total_detections'], reverse=True)
    
    print("\n" + "="*50)
    print("TOP 5 THRESHOLD COMBINATIONS:")
    print("="*50)
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Conf: {result['conf_thresh']:.2f}, IoU: {result['iou_thresh']:.2f}")
        print(f"   Detections: {result['total_detections']}, Avg Conf: {result['avg_confidence']:.3f}")
        print()
    
    # Recommend optimal settings
    best = results[0]
    print("RECOMMENDED SETTINGS:")
    print(f"Confidence Threshold: {best['conf_thresh']}")
    print(f"IoU Threshold: {best['iou_thresh']}")
    
    return best

def update_inference_script(conf_thresh, iou_thresh):
    """Update inference script with optimal thresholds"""
    
    # Read current inference script
    inference_file = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/inference_yolo.py"
    
    print(f"\nTo update your inference script:")
    print(f"In {inference_file}, change:")
    print(f"confidence_threshold = {conf_thresh}")
    print(f"iou_threshold = {iou_thresh}")

if __name__ == "__main__":
    print("🔧 YOLO Threshold Optimization")
    print("==============================")
    
    try:
        best_thresholds = evaluate_thresholds()
        update_inference_script(
            best_thresholds['conf_thresh'], 
            best_thresholds['iou_thresh']
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your model file exists and evaluation images are available")