# Tree Segmentation Project - YOLO with Augmentation

## 🎯 Goal
Train YOLO model to detect and segment trees in aerial imagery, solving class imbalance issue.

## 📊 Problem
- **Training data**: 90.5% individual_tree, 9.5% group_of_trees
- **Previous results**: 100% individual_tree detected, 0% group_of_trees
- **Score**: Stuck at 0.2

## 🚀 Solution
Augmented training with copy-paste, mosaic, and mixup to address class imbalance.

---

## 📁 Project Structure

```
DL_Project/
├── Augmented_Data/           # Your augmented dataset (DO NOT TOUCH)
├── data/
│   ├── train_images/         # Original training images (150)
│   ├── evaluation_images/    # Test images (150) 
│   ├── yolo_dataset/         # YOLO format data
│   └── train_annotations.json
├── scripts/
│   ├── training/
│   │   └── train_yolo_augmented.py
│   └── inference/
│       └── inference_yolo_improved.py
├── runs/                     # Training outputs
├── train_augmented.py        # Quick access to training
├── inference.py              # Quick access to inference
└── README.md                 # This file
```

---

## 🔧 Setup

### Requirements
```bash
pip install ultralytics opencv-python pillow numpy
```

### Data Preparation
Data is already prepared in `data/yolo_dataset/` with:
- Class 0: individual_tree (23,469 instances)
- Class 1: group_of_trees (2,476 instances)

---

## 🏋️ Training

### Quick Start
```bash
python3 train_augmented.py
```

### Training Options
When prompted, select:
- **Option 1**: Quick test (10 epochs, nano) - 10 min
- **Option 2**: Standard (50 epochs, medium) - 2-3 hrs ⭐ **RECOMMENDED**
- **Option 3**: Extensive (100 epochs, medium) - 4-5 hrs
- **Option 4**: Heavy (100 epochs, large) - 6-8 hrs

### What Happens
The script uses aggressive augmentation to fix class imbalance:
- **Copy-Paste (0.3)**: Creates synthetic minority class examples
- **Mosaic (1.0)**: Combines 4 images for multi-scale learning
- **Mixup (0.15)**: Blends images for generalization
- **Heavy transforms**: Rotation, scale, shear, perspective
- **Color jitter**: HSV augmentation for robustness

### Training Output
Model saved to: `runs/segment/tree_segmentation_augmented/weights/best.pt`

---

## 🔮 Inference

### Run Predictions
```bash
python3 inference.py
```

This will:
1. Load the trained model
2. Process all images in `data/evaluation_images/`
3. Generate predictions with improved thresholds (conf=0.25, iou=0.7)
4. Save results to `improved_predictions.json`

### Update Model Path
After training, update the model path in `scripts/inference/inference_yolo_improved.py`:
```python
model_path = project_root / "runs" / "segment" / "tree_segmentation_augmented" / "weights" / "best.pt"
```

---

## 📈 Expected Results

### Before Training
- individual_tree: 31,947 (100%)
- group_of_trees: 0 (0%)
- Score: 0.2

### After Augmented Training
- individual_tree: ~27,000 (85-90%)
- group_of_trees: ~3,000-5,000 (10-15%)
- Score: **0.4-0.6** ⬆️ (+0.2 to +0.4 improvement)

---

## 🎓 Key Augmentation Details

### Copy-Paste Augmentation (0.3)
- **What**: Copies objects from one image and pastes into another
- **Why**: Creates more examples of minority class (group_of_trees)
- **Impact**: Directly addresses 9.5:1 class imbalance

### Mosaic (1.0)
- **What**: Combines 4 images into one training sample
- **Why**: Multi-scale learning and better context
- **Impact**: Handles different resolutions (10cm, 20cm, 40cm, 80cm)

### Mixup (0.15)
- **What**: Blends two images with transparency
- **Why**: Forces model to handle overlapping objects
- **Impact**: Better at detecting overlapping tree canopies

### Geometric Transforms
- Rotation: ±20°
- Scale: 0.7x variation
- Shear: ±5°
- Perspective: 0.001
- Flips: Horizontal + Vertical

---

## 📝 Workflow

### 1. Train Model
```bash
python3 train_augmented.py
# Select option 2 (Standard)
# Wait 2-3 hours
```

### 2. Update Inference Script
Edit `scripts/inference/inference_yolo_improved.py` line ~170:
```python
model_path = project_root / "runs" / "segment" / "tree_segmentation_augmented" / "weights" / "best.pt"
```

### 3. Run Inference
```bash
python3 inference.py
```

### 4. Check Results
```bash
python3 -c "
import json
data = json.load(open('improved_predictions.json'))
classes = {}
for img in data['images']:
    for ann in img['annotations']:
        classes[ann['class']] = classes.get(ann['class'], 0) + 1
total = sum(classes.values())
print('Results:')
for c, count in sorted(classes.items()):
    print(f'  {c}: {count} ({count/total*100:.1f}%)')
print(f'Total detections: {total}')
"
```

---

## ✅ Success Criteria

Your training is successful if:
1. ✅ group_of_trees detected > 0% (target: 8-12%)
2. ✅ Score improves from 0.2 to 0.4+
3. ✅ Total detections around 28,000-32,000
4. ✅ Both classes present in results

---

## 🔧 Troubleshooting

### Still 0% group_of_trees after training?
1. **Train longer**: Use option 3 (100 epochs)
2. **Check validation set**: Ensure it has group_of_trees examples
3. **Try larger model**: Use option 4 (large model)

### Training too slow?
- Using CPU on Mac is slow (2-3 hours for 50 epochs)
- For faster training, use a machine with GPU

### Out of memory?
- Reduce batch size in script (line ~205)
- Use smaller model (option 1 or 2)

---

## 📊 Monitor Training

Watch terminal output for:
- ✅ mAP increasing
- ✅ Both classes in validation
- ✅ Loss decreasing smoothly
- ⚠️ Check for overfitting (train/val loss gap)

Training saves checkpoints every 10 epochs to `runs/segment/tree_segmentation_augmented/`.

---

## 🎯 Quick Reference

```bash
# Train with augmentation
python3 train_augmented.py

# Run inference
python3 inference.py

# Check class distribution in results
python3 -c "
import json
data = json.load(open('improved_predictions.json'))
classes = {}
for img in data['images']:
    for ann in img['annotations']:
        classes[ann['class']] = classes.get(ann['class'], 0) + 1
for c, count in classes.items():
    print(f'{c}: {count}')
"
```

---

## 📌 Important Notes

1. **Do not modify** `Augmented_Data/` folder
2. Training outputs go to `runs/segment/`
3. Inference outputs to `improved_predictions.json`
4. Model uses copy-paste augmentation to fix class imbalance
5. Expected training time: 2-3 hours (option 2 on CPU)

---

## 🚀 Getting Started

Ready to train? Simply run:
```bash
python3 train_augmented.py
```

Select option 2, confirm, and wait for training to complete. Then run inference and check your improved results!

---

**Project Goal**: Improve tree segmentation score from 0.2 to 0.4+ by fixing class imbalance through aggressive augmentation.
