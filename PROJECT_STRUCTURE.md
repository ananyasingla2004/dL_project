# Project Structure

## 📁 Organized Folder Layout

```
DL_Project/
├── 📂 scripts/                      # All Python scripts
│   ├── 📂 training/                 # Model training scripts
│   │   ├── train_yolo.py           # Original YOLO training (10 epochs, nano)
│   │   ├── train_yolo_improved.py  # Improved training (50 epochs, medium)
│   │   ├── train_enhanced_yolo.py  # Enhanced with augmentation
│   │   └── train_detr.py           # DETR model training
│   │
│   ├── 📂 inference/                # Model inference scripts
│   │   ├── inference_yolo.py       # Original inference (conf=0.1, iou=0.6)
│   │   ├── inference_yolo_improved.py  # Improved (conf=0.25, iou=0.7)
│   │   ├── test_inference.py       # Quick test on 3 images
│   │   └── detr_model.py           # DETR inference
│   │
│   ├── 📂 analysis/                 # Analysis and visualization
│   │   ├── optimize_thresholds.py  # Find optimal confidence/IoU
│   │   ├── visualize_yolo.py       # Generate visualizations
│   │   ├── analyze_for_improvement.py  # Detailed analysis
│   │   └── advanced_postprocessing.py  # Ensemble methods
│   │
│   └── 📂 utilities/                # Helper scripts
│       ├── convert_to_yolo.py      # Convert JSON to YOLO format
│       ├── test_yolo_setup.py      # Quick 2-epoch test
│       ├── setup_project.py        # Automated pipeline
│       └── quick_improve.py        # Setup improvement scripts
│
├── 📂 data/                         # Data files and annotations
│   ├── train_annotations.json      # Ground truth for training
│   ├── sample_answer.json          # Expected output format
│   ├── train_images/               # Training images (150)
│   ├── evaluation_images/          # Test images (150)
│   └── yolo_dataset/               # YOLO-formatted data
│       ├── data.yaml               # Dataset configuration
│       ├── train/                  # Training split (120 images)
│       │   ├── images/
│       │   └── labels/
│       └── val/                    # Validation split (30 images)
│           ├── images/
│           └── labels/
│
├── 📂 models/                       # (Empty - for future model checkpoints)
│   └── (You can move trained models here if needed)
│
├── 📂 results/                      # Prediction outputs
│   ├── predictions.json            # Original results (14,481 detections)
│   ├── optimized_predictions.json  # Optimized (31,947 detections)
│   ├── detr_predictions.json       # DETR results
│   └── improved_predictions.json   # Latest improved results
│
├── 📂 documentation/                # Documentation and reports
│   ├── IMPROVEMENT_GUIDE.md        # Comprehensive improvement guide
│   ├── README.md                   # (If exists)
│   └── yolo_visualizations/        # Analysis visualizations
│       ├── statistics/             # Charts and stats
│       │   ├── detections_histogram.png
│       │   ├── category_distribution.png
│       │   └── summary_statistics.txt
│       ├── sample_predictions/     # Sample image predictions
│       └── improvement_report.txt  # Detailed analysis report
│
├── 📂 runs/                         # YOLO training outputs
│   └── segment/                    # Segmentation runs
│       ├── tree_segmentation/      # First run
│       ├── tree_segmentation2/     # Best model (10 epochs)
│       ├── tree_segmentation3/
│       ├── tree_segmentation4/
│       └── test_run/
│
├── 🔗 train.py -> scripts/training/train_yolo_improved.py
├── 🔗 inference.py -> scripts/inference/inference_yolo_improved.py
├── 🔗 analyze.py -> scripts/analysis/analyze_for_improvement.py
│
└── 📄 PROJECT_STRUCTURE.md         # This file

```

## 🚀 Quick Start Commands

### Training
```bash
# Quick way (using symlink)
python3 train.py

# Or full path
python3 scripts/training/train_yolo_improved.py
```

### Inference
```bash
# Quick way (using symlink)
python3 inference.py

# Or full path
python3 scripts/inference/inference_yolo_improved.py
```

### Analysis
```bash
# Quick way (using symlink)
python3 analyze.py

# Or full path
python3 scripts/analysis/analyze_for_improvement.py
```

### Visualization
```bash
python3 scripts/analysis/visualize_yolo.py
```

### Threshold Optimization
```bash
python3 scripts/analysis/optimize_thresholds.py
```

## 📊 Important Paths in Scripts

If you need to update paths in scripts, here are the key ones:

### Training Scripts
```python
data_yaml = 'data/yolo_dataset/data.yaml'  # Updated
```

### Inference Scripts
```python
evaluation_images_dir = 'data/evaluation_images/'  # Updated
output_path = 'results/predictions.json'  # Updated
model_path = 'runs/segment/tree_segmentation2/weights/best.pt'
```

### Analysis Scripts
```python
predictions_path = 'results/optimized_predictions.json'  # Updated
ground_truth_path = 'data/train_annotations.json'  # Updated
evaluation_images_dir = 'data/evaluation_images/'  # Updated
```

## 🔧 Path Updates Needed

**Note**: Some scripts may still have old paths. You'll need to update them:

1. **In training scripts**: Change `yolo_dataset/data.yaml` → `data/yolo_dataset/data.yaml`
2. **In inference scripts**: Change `evaluation_images/` → `data/evaluation_images/`
3. **In analysis scripts**: Change JSON paths to `results/` and `data/` folders

## 📝 Benefits of This Structure

✅ **Clear separation** of concerns (training, inference, analysis)
✅ **Easy to find** specific scripts by function
✅ **Data and code separated** for better management
✅ **Results in one place** for easy comparison
✅ **Documentation centralized** with visualizations
✅ **Quick access** via symlinks (train.py, inference.py, analyze.py)
✅ **Scalable** - easy to add new scripts in appropriate folders

## 🎯 Next Steps

1. Update script paths if you get file not found errors
2. Use the symlinks (`train.py`, `inference.py`, `analyze.py`) for quick access
3. Keep adding new results to `results/` folder
4. Save new documentation to `documentation/` folder
