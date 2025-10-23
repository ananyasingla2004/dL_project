"""
Improved YOLO training script with better settings for class imbalance
"""

from ultralytics import YOLO
import os

def main():
    """Train YOLO with improved settings"""
    
    print("="*60)
    print("IMPROVED YOLO TRAINING")
    print("="*60)
    print()
    print("Improvements:")
    print("  - Using YOLOv8m-seg (medium) instead of nano")
    print("  - Training for 50 epochs instead of 10")
    print("  - Enhanced augmentation")
    print("  - Early stopping with patience=15")
    print("="*60)
    print()
    
    # Use MEDIUM model for better capacity
    model = YOLO('yolov8m-seg.pt')
    
    # Train with improved settings
    results = model.train(
        data='yolo_dataset/data.yaml',
        epochs=50,              # Increased from 10
        batch=4,                # Reduced for CPU
        imgsz=640,
        device='cpu',
        patience=15,            # Early stopping
        save=True,
        project='runs/segment',
        name='tree_segmentation_improved',
        
        # Enhanced augmentation
        degrees=15.0,           # Rotation augmentation
        translate=0.1,          # Translation augmentation
        scale=0.5,              # Scaling augmentation
        fliplr=0.5,             # Horizontal flip
        flipud=0.0,             # No vertical flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        
        # Other settings
        optimizer='AdamW',      # Better optimizer
        lr0=0.001,             # Initial learning rate
        warmup_epochs=3,        # Warmup
        
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: runs/segment/tree_segmentation_improved/weights/best.pt")
    print(f"\nTo use this model, update your inference script with:")
    print(f"  model_path = 'runs/segment/tree_segmentation_improved/weights/best.pt'")

if __name__ == "__main__":
    main()
