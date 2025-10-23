"""
Convert tree segmentation annotations to YOLO format for training.
This script converts the custom JSON format to YOLO segmentation format.
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

class AnnotationConverter:
    def __init__(self, train_annotations_path, train_images_dir, output_dir):
        self.train_annotations_path = train_annotations_path
        self.train_images_dir = train_images_dir
        self.output_dir = Path(output_dir)
        
        # Create YOLO directory structure
        self.yolo_dir = self.output_dir / "yolo_dataset"
        self.train_dir = self.yolo_dir / "train"
        self.val_dir = self.yolo_dir / "val"
        
        # Create directories
        for split in ['train', 'val']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        self.class_map = {
            'individual_tree': 0,
            'group_of_trees': 1
        }
        
    def normalize_polygon(self, segmentation, width, height):
        """Convert polygon coordinates to normalized YOLO format"""
        # Reshape segmentation to pairs of coordinates
        coords = np.array(segmentation).reshape(-1, 2)
        
        # Normalize coordinates to [0, 1]
        coords[:, 0] = coords[:, 0] / width  # x coordinates
        coords[:, 1] = coords[:, 1] / height  # y coordinates
        
        # Flatten back to single list
        return coords.flatten().tolist()
    
    def convert_annotations(self, split_ratio=0.8):
        """Convert annotations to YOLO format"""
        with open(self.train_annotations_path, 'r') as f:
            data = json.load(f)
        
        images = data['images']
        total_images = len(images)
        train_count = int(total_images * split_ratio)
        
        print(f"Total images: {total_images}")
        print(f"Train images: {train_count}")
        print(f"Validation images: {total_images - train_count}")
        
        for idx, image_data in enumerate(images):
            # Determine split
            split = 'train' if idx < train_count else 'val'
            
            file_name = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            
            # Copy image to appropriate directory
            src_image_path = Path(self.train_images_dir) / file_name
            dst_image_path = self.yolo_dir / split / 'images' / file_name
            
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image not found: {src_image_path}")
                continue
            
            # Create YOLO annotation file
            txt_file = file_name.replace('.tif', '.txt')
            label_path = self.yolo_dir / split / 'labels' / txt_file
            
            with open(label_path, 'w') as f:
                for annotation in image_data['annotations']:
                    class_name = annotation['class']
                    class_id = self.class_map[class_name]
                    segmentation = annotation['segmentation']
                    
                    # Normalize polygon coordinates
                    normalized_coords = self.normalize_polygon(segmentation, width, height)
                    
                    # Write YOLO format: class_id x1 y1 x2 y2 ... xn yn
                    coords_str = ' '.join(map(str, normalized_coords))
                    f.write(f"{class_id} {coords_str}\n")
        
        # Create dataset YAML file
        self.create_yaml_file()
        
        print(f"Conversion completed! Dataset saved to: {self.yolo_dir}")
    
    def create_yaml_file(self):
        """Create YAML configuration file for YOLO training"""
        yaml_content = f"""# Tree segmentation dataset configuration
path: {str(self.yolo_dir.absolute())}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: individual_tree
  1: group_of_trees

nc: 2  # number of classes
"""
        
        yaml_path = self.yolo_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"YAML config created: {yaml_path}")

def main():
    # Paths
    train_annotations = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/train_annotations.json"
    train_images = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/train_images"
    output_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project"
    
    # Check if train_images directory exists
    if not os.path.exists(train_images):
        print(f"Warning: {train_images} not found. Please extract train_images.zip first.")
        print("You can extract it using: unzip train_images.zip")
        return
    
    # Convert annotations
    converter = AnnotationConverter(train_annotations, train_images, output_dir)
    converter.convert_annotations()

if __name__ == "__main__":
    main()