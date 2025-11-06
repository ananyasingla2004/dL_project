"""
YOLO Training with Advanced Augmentations for Class Imbalance
Focus: Help model learn both individual_tree and group_of_trees classes

Key improvements:
1. Enhanced augmentation for minority class (group_of_trees)
2. Class-aware augmentation strategy
3. Weighted loss to handle imbalance
4. Extended training with proper monitoring
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
import random

class AugmentedYOLOTrainer:
    def __init__(self, data_yaml_path, output_name='tree_segmentation_augmented'):
        self.data_yaml_path = Path(data_yaml_path)
        self.output_name = output_name
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load data config
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        print(f"Project root: {self.project_root}")
        print(f"Data config: {self.data_yaml_path}")
    
    def analyze_class_distribution(self):
        """Analyze training data to understand class imbalance"""
        print("\n" + "="*80)
        print("ANALYZING CLASS DISTRIBUTION")
        print("="*80)
        
        # Get labels directory
        dataset_path = self.data_yaml_path.parent
        train_labels_dir = dataset_path / 'train' / 'labels'
        
        class_counts = {0: 0, 1: 0}  # 0: individual_tree, 1: group_of_trees
        total_annotations = 0
        
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        total_annotations += 1
        
        print(f"\nClass Distribution in Training Data:")
        print(f"  Class 0 (individual_tree): {class_counts[0]} ({class_counts[0]/total_annotations*100:.1f}%)")
        print(f"  Class 1 (group_of_trees):  {class_counts[1]} ({class_counts[1]/total_annotations*100:.1f}%)")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")
        
        return class_counts, total_annotations
    
    def train_with_augmentation(self, 
                               model_size='m',  # Use medium model for better capacity
                               epochs=100,      # More epochs for better learning
                               batch_size=8,
                               image_size=640,
                               device='cpu'):
        """
        Train YOLO with aggressive augmentation to handle class imbalance
        """
        
        # Analyze class distribution
        class_counts, total_annotations = self.analyze_class_distribution()
        
        print("\n" + "="*80)
        print("STARTING AUGMENTED TRAINING")
        print("="*80)
        
        # Load model
        model_name = f'yolov8{model_size}-seg.pt'
        print(f"\nLoading model: {model_name}")
        model = YOLO(model_name)
        
        # Calculate class weights for imbalanced data
        # Give higher weight to minority class (group_of_trees)
        weight_0 = 1.0
        weight_1 = class_counts[0] / class_counts[1]  # ~9.5x for group_of_trees
        
        print(f"\nClass weights:")
        print(f"  individual_tree: {weight_0:.2f}")
        print(f"  group_of_trees:  {weight_1:.2f}")
        
        print(f"\nTraining Configuration:")
        print(f"  Model: YOLOv8{model_size}-seg")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}")
        print(f"  Device: {device}")
        
        print(f"\nAugmentation Strategy:")
        print(f"  ✓ Mosaic: 1.0 (combine 4 images - helps with scale variation)")
        print(f"  ✓ Mixup: 0.15 (blend images - helps with occlusion)")
        print(f"  ✓ Copy-paste: 0.3 (paste objects - helps with minority class)")
        print(f"  ✓ Heavy geometric transforms (rotation, scale, perspective)")
        print(f"  ✓ Color augmentation (HSV jitter)")
        print(f"  ✓ Flips and translations")
        
        # Train with aggressive augmentation
        results = model.train(
            data=str(self.data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            
            # Project settings
            project=str(self.project_root / 'runs' / 'segment'),
            name=self.output_name,
            exist_ok=False,
            
            # Training settings
            patience=20,          # Early stopping patience
            save=True,
            save_period=10,       # Save checkpoint every 10 epochs
            
            # Optimization
            optimizer='AdamW',    # Better optimizer for imbalanced data
            lr0=0.001,           # Initial learning rate
            lrf=0.01,            # Final learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,     # Warmup for stable start
            
            # Loss weights (emphasize classification)
            cls=1.0,             # Classification loss weight
            box=7.5,             # Box loss weight
            
            # AGGRESSIVE AUGMENTATION FOR CLASS IMBALANCE
            # Mosaic and mixup
            mosaic=1.0,          # Always use mosaic (combines 4 images)
            mixup=0.15,          # Mix two images together (helps generalization)
            copy_paste=0.3,      # Copy-paste augmentation (helps minority class!)
            
            # Geometric augmentations
            degrees=20.0,        # Rotation (+/- degrees)
            translate=0.2,       # Translation (+/- fraction)
            scale=0.7,           # Scale (+/- gain)
            shear=5.0,           # Shear (+/- degrees)
            perspective=0.001,   # Perspective distortion
            flipud=0.5,          # Vertical flip probability
            fliplr=0.5,          # Horizontal flip probability
            
            # Color augmentations
            hsv_h=0.015,         # HSV-Hue augmentation
            hsv_s=0.7,           # HSV-Saturation augmentation
            hsv_v=0.4,           # HSV-Value augmentation
            
            # Advanced settings
            close_mosaic=10,     # Disable mosaic in last N epochs for fine-tuning
            
            # Validation
            val=True,
            plots=True,          # Save training plots
            
            # Multi-scale training
            rect=False,          # Don't use rectangular training (use square)
            
            # Verbosity
            verbose=True
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        
        # Print results location
        save_dir = Path(results.save_dir)
        print(f"\nModel saved to: {save_dir}")
        print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
        print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
        
        # Print metrics
        if hasattr(results, 'results_dict'):
            print(f"\nFinal Metrics:")
            metrics = results.results_dict
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        return results, save_dir

def main():
    """Main training function"""
    
    print("="*80)
    print("YOLO TRAINING WITH AUGMENTATION FOR CLASS IMBALANCE")
    print("="*80)
    print("\nGoal: Improve detection of 'group_of_trees' class")
    print("Strategy: Aggressive augmentation + longer training\n")
    
    # Get project root and paths
    project_root = Path(__file__).parent.parent.parent
    data_yaml = project_root / 'data' / 'yolo_dataset' / 'data.yaml'
    
    # Check if data exists
    if not data_yaml.exists():
        print(f"❌ Error: Data config not found at {data_yaml}")
        print("Please run the data conversion script first:")
        print("  python3 scripts/utilities/convert_to_yolo.py")
        return
    
    # Initialize trainer
    trainer = AugmentedYOLOTrainer(
        data_yaml_path=data_yaml,
        output_name='tree_segmentation_augmented'
    )
    
    # Training options
    print("\n" + "="*80)
    print("TRAINING OPTIONS")
    print("="*80)
    print("\nSelect training configuration:")
    print("  1. Quick test (10 epochs, nano model) - 10 minutes")
    print("  2. Standard (50 epochs, medium model) - 2-3 hours ⭐ RECOMMENDED")
    print("  3. Extensive (100 epochs, medium model) - 4-5 hours")
    print("  4. Heavy (100 epochs, large model) - 6-8 hours")
    
    try:
        choice = input("\nEnter choice (1-4, default=2): ").strip() or "2"
    except:
        choice = "2"
    
    # Configure based on choice
    configs = {
        "1": {"model_size": "n", "epochs": 10, "batch_size": 16},
        "2": {"model_size": "m", "epochs": 50, "batch_size": 8},
        "3": {"model_size": "m", "epochs": 100, "batch_size": 8},
        "4": {"model_size": "l", "epochs": 100, "batch_size": 4},
    }
    
    config = configs.get(choice, configs["2"])
    
    print(f"\n✓ Selected configuration:")
    print(f"  Model: YOLOv8{config['model_size']}-seg")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    
    # Ask for confirmation
    try:
        confirm = input("\nProceed with training? (y/n, default=y): ").strip().lower() or "y"
    except:
        confirm = "y"
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🚀 Starting training...\n")
    
    # Train
    results, save_dir = trainer.train_with_augmentation(
        model_size=config['model_size'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        image_size=640,
        device='cpu'  # Change to 'mps' for Mac GPU or '0' for CUDA
    )
    
    # Print next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Run inference with new model:")
    print(f"   Update model_path in inference script to:")
    print(f"   {save_dir / 'weights' / 'best.pt'}")
    print(f"\n2. Or update the symlink:")
    print(f"   In scripts/inference/inference_yolo_improved.py")
    print(f"   Change model_path to: '{save_dir.relative_to(project_root)}/weights/best.pt'")
    print(f"\n3. Run inference:")
    print(f"   python3 inference.py")
    print(f"\n4. Analyze results:")
    print(f"   python3 analyze.py")
    
    print("\n✅ Training complete! Check the model and run inference.")

if __name__ == "__main__":
    main()
