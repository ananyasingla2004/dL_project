"""
YOLO training script for tree segmentation using pre-trained COCO weights.
This script uses YOLOv8 for instance segmentation.
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_yolo_model():
    """Train YOLO model with pre-trained COCO weights"""
    
    # Check if dataset exists
    dataset_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/yolo_dataset"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print("Dataset not found! Please run convert_to_yolo.py first.")
        return
    
    # Initialize YOLO model with pre-trained COCO weights
    # Using YOLOv8n-seg (nano segmentation model) for faster training
    # You can use 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt' for better accuracy
    model = YOLO('yolov8n-seg.pt')  # Load pre-trained COCO model
    
    print("Starting training with pre-trained COCO weights...")
    print("Model: YOLOv8n-seg")
    print(f"Dataset: {yaml_path}")
    
    # Training parameters
    results = model.train(
        data=yaml_path,           # Path to dataset YAML
        epochs=10,                # Number of training epochs (set to 10 for submission)
        imgsz=640,                # Image size
        batch=4,                  # Batch size (reduced for CPU)
        device='mps',             # Use Metal Performance Shaders (MPS) on Mac
        project='runs/segment',   # Project directory
        name='tree_segmentation', # Experiment name
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every 10 epochs
        patience=20,              # Early stopping patience
        
        # Data augmentation
        hsv_h=0.015,              # HSV hue augmentation
        hsv_s=0.7,                # HSV saturation augmentation
        hsv_v=0.4,                # HSV value augmentation
        degrees=10.0,             # Rotation degrees
        translate=0.1,            # Translation
        scale=0.5,                # Scale augmentation
        shear=0.0,                # Shear augmentation
        perspective=0.0,          # Perspective augmentation
        flipud=0.0,               # Vertical flip probability
        fliplr=0.5,               # Horizontal flip probability
        mosaic=1.0,               # Mosaic augmentation probability
        mixup=0.0,                # Mixup augmentation probability
        
        # Optimization
        optimizer='AdamW',        # Optimizer
        lr0=0.01,                 # Initial learning rate
        lrf=0.01,                 # Final learning rate factor
        momentum=0.937,           # Momentum
        weight_decay=0.0005,      # Weight decay
        warmup_epochs=3.0,        # Warmup epochs
        warmup_momentum=0.8,      # Warmup momentum
        warmup_bias_lr=0.1,       # Warmup bias learning rate
        
        # Validation
        val=True,                 # Validate during training
        
        # Other settings
        verbose=True,             # Verbose output
        seed=42,                  # Random seed for reproducibility
        deterministic=True,       # Deterministic training
        single_cls=False,         # Treat as single class
        rect=False,               # Rectangular training
        cos_lr=False,             # Cosine learning rate scheduler
        close_mosaic=10,          # Disable mosaic in last N epochs
        resume=False,             # Resume training from last checkpoint
        amp=True,                 # Automatic Mixed Precision training
        fraction=1.0,             # Dataset fraction to train on
        profile=False,            # Profile ONNX and TensorRT speeds
        
        # Loss weights
        cls=1.0,                  # Classification loss weight
        box=7.5,                  # Box loss weight  
    )
    
    print("Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Last model saved at: {model.trainer.last}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"Validation results: {metrics}")
    
    return model, results

def export_model(model_path):
    """Export trained model to different formats"""
    model = YOLO(model_path)
    
    # Export to ONNX for deployment
    model.export(format='onnx', imgsz=640)
    print("Model exported to ONNX format")
    
    # Export to TensorRT if available
    try:
        model.export(format='engine', imgsz=640)
        print("Model exported to TensorRT format")
    except:
        print("TensorRT export failed (requires TensorRT installation)")

if __name__ == "__main__":
    print("YOLO Tree Segmentation Training")
    print("================================")
    
    # Train the model
    model, results = train_yolo_model()
    
    # Export the best model
    best_model_path = model.trainer.best
    print(f"\nExporting best model: {best_model_path}")
    export_model(best_model_path)
    
    print("\nTraining pipeline completed!")
    print("Next steps:")
    print("1. Check training results in 'runs/segment/tree_segmentation'")
    print("2. Use the best model for inference on evaluation images")
    print("3. Run inference_yolo.py to generate predictions")