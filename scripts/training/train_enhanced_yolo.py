"""
Enhanced YOLO training script with improved data augmentation and hyperparameters.
This version includes more sophisticated augmentation for better generalization.
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_enhanced_yolo_model():
    """Train YOLO model with enhanced settings for better performance"""
    
    # Check if dataset exists
    dataset_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/yolo_dataset"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print("Dataset not found! Please run convert_to_yolo.py first.")
        return
    
    # Use a larger model for better accuracy
    # Options: yolov8n-seg.pt (fastest), yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt (best)
    model = YOLO('yolov8s-seg.pt')  # Small model - good balance of speed/accuracy
    
    print("Starting ENHANCED training with pre-trained COCO weights...")
    print("Model: YOLOv8s-seg (Small)")
    print(f"Dataset: {yaml_path}")
    
    # Enhanced training parameters
    results = model.train(
        data=yaml_path,
        epochs=100,               # More epochs for better convergence
        imgsz=640,                # Image size
        batch=8,                  # Larger batch if you have enough memory
        device='cpu',             # Use 'mps' if you want to try Metal Performance Shaders
        project='runs/segment',
        name='enhanced_tree_segmentation',
        save=True,
        save_period=25,           # Save every 25 epochs
        patience=30,              # More patience for better training
        
        # Enhanced Data Augmentation
        hsv_h=0.02,               # Increased HSV hue variation
        hsv_s=0.8,                # Higher saturation variation
        hsv_v=0.5,                # Higher value variation
        degrees=15.0,             # More rotation (trees can be at different angles)
        translate=0.15,           # More translation
        scale=0.7,                # More scale variation (different tree sizes)
        shear=5.0,                # Add shear transformation
        perspective=0.0002,       # Slight perspective changes
        flipud=0.2,               # Some vertical flips (aerial imagery)
        fliplr=0.5,               # Horizontal flips
        mosaic=1.0,               # Mosaic augmentation
        mixup=0.1,                # Add mixup augmentation
        copy_paste=0.1,           # Copy-paste augmentation for segmentation
        
        # Optimized Learning Parameters
        optimizer='AdamW',
        lr0=0.001,                # Lower initial learning rate for stability
        lrf=0.001,                # Lower final learning rate
        momentum=0.9,             # Higher momentum
        weight_decay=0.0005,
        warmup_epochs=5.0,        # More warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Training Strategy
        val=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,              # Use cosine learning rate scheduler
        close_mosaic=15,          # Disable mosaic in last 15 epochs
        resume=False,
        amp=True,                 # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        
        # Loss weights (tune these based on your data)
        cls=1.0,                  # Classification loss
        box=7.5,                  # Box regression loss
        # Note: segmentation loss is handled automatically
        
        # Additional parameters for better segmentation
        overlap_mask=True,        # Allow overlapping masks
        mask_ratio=4,             # Mask downsampling ratio
        dropout=0.0,              # No dropout (can add if overfitting)
    )
    
    print("Enhanced training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Last model saved at: {model.trainer.last}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"Validation results: {metrics}")
    
    return model, results

def compare_models():
    """Compare different YOLO model sizes on your dataset"""
    
    models_to_test = [
        ('yolov8n-seg.pt', 'Nano - Fastest'),
        ('yolov8s-seg.pt', 'Small - Balanced'),
        ('yolov8m-seg.pt', 'Medium - Better Accuracy'),
    ]
    
    dataset_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/yolo_dataset/data.yaml"
    
    print("Model Comparison (5 epochs each for speed):")
    print("=" * 50)
    
    for model_name, description in models_to_test:
        print(f"\nTesting {model_name} - {description}")
        model = YOLO(model_name)
        
        # Quick training for comparison
        results = model.train(
            data=dataset_path,
            epochs=5,
            imgsz=640,
            batch=4,
            device='cpu',
            project='runs/segment',
            name=f'comparison_{model_name.split(".")[0]}',
            verbose=False,
            save=False
        )
        
        # Get validation metrics
        metrics = model.val(data=dataset_path, verbose=False)
        print(f"mAP50 (Box): {metrics.box.map50:.4f}")
        print(f"mAP50 (Mask): {metrics.seg.map50:.4f}")
        print(f"Parameters: ~{model.model.parameters():,}")

if __name__ == "__main__":
    print("🚀 Enhanced YOLO Tree Segmentation Training")
    print("===========================================")
    
    choice = input("Choose option:\n1. Enhanced Training (100 epochs)\n2. Model Comparison (quick)\nEnter 1 or 2: ")
    
    if choice == "1":
        model, results = train_enhanced_yolo_model()
        print("\nEnhanced training pipeline completed!")
        print("Check results in 'runs/segment/enhanced_tree_segmentation'")
    elif choice == "2":
        compare_models()
    else:
        print("Invalid choice. Run script again with 1 or 2.")