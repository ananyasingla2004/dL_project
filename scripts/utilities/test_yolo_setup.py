"""
Quick test script to verify YOLO setup is working correctly.
This script will run a few epochs to validate the setup.
"""

from ultralytics import YOLO
import os

def test_yolo_setup():
    """Test YOLO setup with minimal training"""
    
    # Check if dataset exists
    dataset_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/yolo_dataset"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print("❌ Dataset not found! Please run convert_to_yolo.py first.")
        return False
    
    print("✅ Dataset found!")
    
    # Load model
    try:
        model = YOLO('yolov8n-seg.pt')
        print("✅ YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return False
    
    # Test training with minimal epochs
    print("🧪 Testing training with 2 epochs...")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=2,              # Very few epochs for testing
            imgsz=640,
            batch=2,               # Small batch size
            device='cpu',
            project='runs/segment',
            name='test_run',
            verbose=True,
            patience=100,          # Disable early stopping for test
        )
        
        print("✅ Training test completed successfully!")
        print(f"✅ Model saved to: runs/segment/test_run/weights/best.pt")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 YOLO Setup Test")
    print("==================")
    
    success = test_yolo_setup()
    
    if success:
        print("\n🎉 YOLO setup is working correctly!")
        print("📋 Next steps:")
        print("1. Run full training: python train_yolo.py")
        print("2. Or run longer test: modify epochs in this script")
        print("3. Monitor training progress in runs/segment/ directory")
    else:
        print("\n❌ YOLO setup test failed. Please check the errors above.")