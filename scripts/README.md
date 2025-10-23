# Scripts Directory

## 📂 Folder Structure

### training/
Model training scripts - use these to train new models
- `train_yolo_improved.py` - **Recommended**: 50 epochs, medium model
- `train_yolo.py` - Original: 10 epochs, nano model
- `train_enhanced_yolo.py` - With enhanced augmentation
- `train_detr.py` - DETR model training

### inference/
Run predictions on evaluation images
- `inference_yolo_improved.py` - **Recommended**: Better thresholds (0.25/0.7)
- `inference_yolo.py` - Original: Lower thresholds (0.1/0.6)
- `test_inference.py` - Quick test on 3 images
- `detr_model.py` - DETR inference

### analysis/
Analyze and optimize your model
- `analyze_for_improvement.py` - **Start here**: Detailed analysis
- `visualize_yolo.py` - Generate visualizations
- `optimize_thresholds.py` - Find best confidence/IoU thresholds
- `advanced_postprocessing.py` - Ensemble methods

### utilities/
Helper scripts for setup and conversion
- `convert_to_yolo.py` - Convert JSON to YOLO format
- `quick_improve.py` - Setup improvement scripts
- `setup_project.py` - Automated pipeline
- `test_yolo_setup.py` - Quick 2-epoch test

## 🚀 Quick Commands

From project root:

```bash
# Training (recommended)
python3 scripts/training/train_yolo_improved.py

# Inference (recommended)
python3 scripts/inference/inference_yolo_improved.py

# Analysis
python3 scripts/analysis/analyze_for_improvement.py

# Visualization
python3 scripts/analysis/visualize_yolo.py
```

Or use the symlinks:
```bash
python3 train.py      # -> scripts/training/train_yolo_improved.py
python3 inference.py  # -> scripts/inference/inference_yolo_improved.py
python3 analyze.py    # -> scripts/analysis/analyze_for_improvement.py
```
