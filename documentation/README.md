# YOLO Tree Segmentation Project

This project uses a pre-trained YOLO model (trained on COCO dataset) for tree segmentation. The model is fine-tuned to detect and segment two classes:
- `individual_tree`
- `group_of_trees`

## Project Structure

```
├── train_annotations.json      # Training annotations
├── sample_answer.json         # Sample output format
├── train_images/              # Training images (extract from zip)
├── evaluation_images/         # Evaluation images
├── convert_to_yolo.py         # Convert annotations to YOLO format
├── train_yolo.py              # Train YOLO model
├── inference_yolo.py          # Run inference on evaluation images
├── setup_project.py           # Setup and run complete pipeline
└── README.md                  # This file
```

## Setup Instructions

### 1. Extract Image Data
First, extract the training images:
```bash
# If you have train_images.zip
unzip train_images.zip
```

### 2. Run the Complete Pipeline
Use the setup script to run everything automatically:
```bash
python setup_project.py
```

Or run each step manually:

### 3. Manual Steps

#### Step 1: Convert Annotations to YOLO Format
```bash
python convert_to_yolo.py
```
This creates a `yolo_dataset/` directory with:
- `train/` and `val/` splits
- Images and labels in YOLO format
- `data.yaml` configuration file

#### Step 2: Train the YOLO Model
```bash
python train_yolo.py
```
This will:
- Download pre-trained YOLOv8n-seg weights (COCO dataset)
- Fine-tune on your tree segmentation data
- Save the best model in `runs/segment/tree_segmentation/weights/`

#### Step 3: Run Inference
```bash
python inference_yolo.py
```
This will:
- Load the trained model
- Process all evaluation images
- Generate `predictions.json` in the required format

## Model Configuration

### YOLO Model Variants
- `yolov8n-seg.pt`: Nano (fastest, least accurate)
- `yolov8s-seg.pt`: Small
- `yolov8m-seg.pt`: Medium  
- `yolov8l-seg.pt`: Large
- `yolov8x-seg.pt`: Extra Large (slowest, most accurate)

You can change the model variant in `train_yolo.py` by modifying:
```python
model = YOLO('yolov8n-seg.pt')  # Change this line
```

### Training Parameters
Key parameters you can adjust in `train_yolo.py`:
- `epochs`: Number of training epochs (default: 100)
- `batch`: Batch size (default: 16)
- `imgsz`: Image size (default: 640)
- `lr0`: Initial learning rate (default: 0.01)

### Inference Parameters
Adjust confidence and IoU thresholds in `inference_yolo.py`:
```python
confidence_threshold = 0.25  # Minimum confidence score
iou_threshold = 0.45         # IoU threshold for NMS
```

## Expected Results

After training, you should see:
1. Training metrics in `runs/segment/tree_segmentation/`
2. Best model weights saved as `.pt` file
3. Validation metrics showing model performance
4. Predictions file matching the `sample_answer.json` format

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The model will automatically use GPU if available, otherwise CPU
2. **Memory Issues**: Reduce batch size in training parameters
3. **Poor Performance**: Try a larger model variant or increase training epochs
4. **Missing Images**: Make sure to extract train_images.zip and evaluation_images.zip

### Performance Tuning

1. **Increase Accuracy**:
   - Use larger model (yolov8l-seg or yolov8x-seg)
   - Increase training epochs
   - Adjust data augmentation parameters

2. **Increase Speed**:
   - Use smaller model (yolov8n-seg)
   - Reduce image size
   - Optimize inference parameters

## File Formats

### Input Format (train_annotations.json)
```json
{
  "images": [
    {
      "file_name": "10cm_train_1.tif",
      "width": 1024,
      "height": 1024,
      "cm_resolution": 10,
      "scene_type": "agriculture_plantation",
      "annotations": [
        {
          "class": "individual_tree",
          "confidence_score": 1.0,
          "segmentation": [x1, y1, x2, y2, ...]
        }
      ]
    }
  ]
}
```

### Output Format (predictions.json)
Same format as input but with predicted annotations and confidence scores.

## Dependencies

- ultralytics
- opencv-python
- numpy
- matplotlib
- pillow
- tqdm
- scipy

All dependencies are automatically installed when you run the setup.