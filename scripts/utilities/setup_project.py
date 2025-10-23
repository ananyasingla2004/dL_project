"""
Complete setup and pipeline runner for YOLO tree segmentation project.
This script runs the entire pipeline from data preparation to inference.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} not found: {filepath}")
        return False

def main():
    print("🌳 YOLO Tree Segmentation Pipeline")
    print("==================================")
    
    # Get the Python executable path
    python_executable = '"/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/.venv/bin/python"'
    
    # Check prerequisites
    print("\n📋 Checking Prerequisites...")
    
    required_files = [
        ("/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/train_annotations.json", "Training annotations"),
        ("/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/sample_answer.json", "Sample answer format"),
        ("/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images", "Evaluation images directory")
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    # Check for training images
    train_images_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/train_images"
    if not os.path.exists(train_images_dir):
        print(f"❌ Training images directory not found: {train_images_dir}")
        print("📦 Please extract train_images.zip first:")
        print("   unzip train_images.zip")
        all_files_exist = False
    else:
        print(f"✅ Training images directory: {train_images_dir}")
    
    if not all_files_exist:
        print("\n❌ Missing required files. Please ensure all files are available before running.")
        return False
    
    # Step 1: Convert annotations to YOLO format
    if not run_command(f'{python_executable} convert_to_yolo.py', 
                      "Converting annotations to YOLO format"):
        return False
    
    # Check if YOLO dataset was created
    yolo_dataset_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/yolo_dataset"
    if not check_file_exists(yolo_dataset_path, "YOLO dataset directory"):
        return False
    
    # Step 2: Train YOLO model
    print(f"\n🚀 Starting YOLO training...")
    print("This may take a while depending on your hardware...")
    
    if not run_command(f'{python_executable} train_yolo.py', 
                      "Training YOLO model with pre-trained COCO weights"):
        print("❌ Training failed. Check the error messages above.")
        return False
    
    # Check if model was trained
    model_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/tree_segmentation/weights/best.pt"
    if not check_file_exists(model_path, "Trained model"):
        return False
    
    # Step 3: Run inference
    if not run_command(f'{python_executable} inference_yolo.py', 
                      "Running inference on evaluation images"):
        return False
    
    # Check if predictions were generated
    predictions_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/predictions.json"
    if not check_file_exists(predictions_path, "Predictions file"):
        return False
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"📊 Training results: runs/segment/tree_segmentation/")
    print(f"🤖 Best model: {model_path}")
    print(f"📋 Predictions: {predictions_path}")
    print("\n📈 Next steps:")
    print("1. Review training metrics in the results directory")
    print("2. Validate predictions against sample_answer.json format")
    print("3. Tune hyperparameters if needed")
    print("4. Consider using a larger model variant for better accuracy")
    
    return True

def quick_setup():
    """Quick setup without full pipeline - just data conversion"""
    print("🔧 Quick Setup Mode")
    print("==================")
    
    python_executable = '"/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/.venv/bin/python"'
    
    # Just convert data to YOLO format
    if run_command(f'{python_executable} convert_to_yolo.py', 
                  "Converting annotations to YOLO format"):
        print("\n✅ Data conversion completed!")
        print("📋 Next steps:")
        print(f"1. Run training: {python_executable} train_yolo.py")
        print(f"2. Run inference: {python_executable} inference_yolo.py")
        print("3. Or run full pipeline: python setup_project.py")
        return True
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_setup()
    else:
        success = main()
    
    if not success:
        print("\n❌ Pipeline failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All done!")
        sys.exit(0)