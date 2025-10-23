"""
YOLO Model Visualization Script
Helps identify issues and areas for improvement in tree detection
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from collections import defaultdict

def load_predictions(json_path):
    """Load predictions from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_ground_truth(json_path):
    """Load ground truth annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to image-based dictionary
    gt_dict = {}
    for img in data.get('images', []):
        gt_dict[img['file_name']] = img.get('annotations', [])
    return gt_dict

def visualize_predictions(image_path, annotations, save_path, title="Predictions"):
    """Visualize predictions on an image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(img_rgb)
    ax.set_title(f"{title}\n{len(annotations)} detections", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Color map for categories
    colors = {'individual_tree': 'lime', 'group_of_trees': 'yellow'}
    
    # Draw annotations
    for ann in annotations:
        segmentation = ann.get('segmentation', [[]])
        if not segmentation or not segmentation[0]:
            continue
        
        # Get polygon points
        try:
            points = np.array(segmentation[0]).reshape(-1, 2)
            if len(points) < 3:  # Need at least 3 points for a polygon
                continue
        except:
            continue
        
        # Draw polygon
        category = ann.get('category_name', 'individual_tree')
        color = colors.get(category, 'red')
        
        polygon = MPLPolygon(points, fill=False, edgecolor=color, 
                            linewidth=2, alpha=0.8)
        ax.add_patch(polygon)
        
        # Add confidence score if available
        if 'score' in ann:
            # Get centroid
            cx, cy = points.mean(axis=0)
            ax.text(cx, cy, f"{ann['score']:.2f}", 
                   color='white', fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lime', edgecolor='lime', label='Individual Tree'),
        Patch(facecolor='yellow', edgecolor='yellow', label='Group of Trees')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def compare_predictions_gt(image_path, predictions, ground_truth, save_path):
    """Compare predictions with ground truth side by side"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ground Truth
    ax1.imshow(img_rgb)
    ax1.set_title(f"Ground Truth\n{len(ground_truth)} annotations", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    colors = {'individual_tree': 'lime', 'group_of_trees': 'yellow'}
    
    for ann in ground_truth:
        segmentation = ann.get('segmentation', [[]])
        if not segmentation or not segmentation[0]:
            continue
        try:
            points = np.array(segmentation[0]).reshape(-1, 2)
            if len(points) < 3:
                continue
        except:
            continue
        category = ann.get('category_name', 'individual_tree')
        color = colors.get(category, 'red')
        polygon = MPLPolygon(points, fill=False, edgecolor=color, 
                            linewidth=2, alpha=0.8)
        ax1.add_patch(polygon)
    
    # Predictions
    ax2.imshow(img_rgb)
    ax2.set_title(f"Predictions\n{len(predictions)} detections", 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    for ann in predictions:
        segmentation = ann.get('segmentation', [[]])
        if not segmentation or not segmentation[0]:
            continue
        try:
            points = np.array(segmentation[0]).reshape(-1, 2)
            if len(points) < 3:
                continue
        except:
            continue
        category = ann.get('category_name', 'individual_tree')
        color = colors.get(category, 'red')
        polygon = MPLPolygon(points, fill=False, edgecolor=color, 
                            linewidth=2, alpha=0.8)
        ax2.add_patch(polygon)
        
        if 'score' in ann:
            cx, cy = points.mean(axis=0)
            ax2.text(cx, cy, f"{ann['score']:.2f}", 
                    color='white', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")

def analyze_predictions(predictions_data):
    """Analyze prediction statistics"""
    stats = {
        'total_images': len(predictions_data.get('images', [])),
        'total_detections': 0,
        'detections_per_image': [],
        'confidence_scores': [],
        'categories': defaultdict(int),
        'bbox_areas': [],
        'images_with_no_detections': [],
        'images_with_many_detections': []
    }
    
    for img in predictions_data.get('images', []):
        annotations = img.get('annotations', [])
        num_detections = len(annotations)
        stats['total_detections'] += num_detections
        stats['detections_per_image'].append(num_detections)
        
        if num_detections == 0:
            stats['images_with_no_detections'].append(img['file_name'])
        elif num_detections > 50:
            stats['images_with_many_detections'].append((img['file_name'], num_detections))
        
        for ann in annotations:
            if 'score' in ann:
                stats['confidence_scores'].append(ann['score'])
            
            category = ann.get('category_name', 'individual_tree')
            stats['categories'][category] += 1
            
            # Calculate bbox area
            if 'bbox' in ann:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]
                stats['bbox_areas'].append(area)
    
    return stats

def plot_statistics(stats, save_dir):
    """Plot various statistics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Detections per image histogram
    plt.figure(figsize=(12, 6))
    plt.hist(stats['detections_per_image'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Detections', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title(f'Distribution of Detections per Image\nTotal: {stats["total_detections"]} detections across {stats["total_images"]} images', 
              fontsize=14, fontweight='bold')
    plt.axvline(np.mean(stats['detections_per_image']), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["detections_per_image"]):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'detections_histogram.png', dpi=150)
    plt.close()
    print(f"Saved detections histogram")
    
    # 2. Confidence score distribution
    if stats['confidence_scores']:
        plt.figure(figsize=(12, 6))
        plt.hist(stats['confidence_scores'], bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.title(f'Confidence Score Distribution\nMean: {np.mean(stats["confidence_scores"]):.3f}, Median: {np.median(stats["confidence_scores"]):.3f}', 
                  fontsize=14, fontweight='bold')
        plt.axvline(np.mean(stats['confidence_scores']), color='red', 
                    linestyle='--', linewidth=2, label='Mean')
        plt.axvline(np.median(stats['confidence_scores']), color='blue', 
                    linestyle='--', linewidth=2, label='Median')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_distribution.png', dpi=150)
        plt.close()
        print(f"Saved confidence distribution")
    
    # 3. Category distribution
    plt.figure(figsize=(10, 6))
    categories = list(stats['categories'].keys())
    counts = list(stats['categories'].values())
    colors_cat = ['lime', 'yellow'][:len(categories)]
    bars = plt.bar(categories, counts, color=colors_cat, edgecolor='black', linewidth=2)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Detections', fontsize=12)
    plt.title('Detections by Category', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'category_distribution.png', dpi=150)
    plt.close()
    print(f"Saved category distribution")
    
    # 4. Bbox area distribution
    if stats['bbox_areas']:
        plt.figure(figsize=(12, 6))
        plt.hist(stats['bbox_areas'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Bounding Box Area (pixels²)', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.title('Detection Size Distribution', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'bbox_area_distribution.png', dpi=150)
        plt.close()
        print(f"Saved bbox area distribution")
    
    # 5. Summary statistics text
    summary_text = f"""
YOLO MODEL PERFORMANCE SUMMARY
{'='*60}

OVERALL STATISTICS:
- Total Images: {stats['total_images']}
- Total Detections: {stats['total_detections']}
- Average Detections per Image: {np.mean(stats['detections_per_image']):.2f}
- Median Detections per Image: {np.median(stats['detections_per_image']):.0f}
- Std Dev Detections per Image: {np.std(stats['detections_per_image']):.2f}

DETECTION RANGE:
- Min Detections in an Image: {min(stats['detections_per_image'])}
- Max Detections in an Image: {max(stats['detections_per_image'])}
- Images with 0 Detections: {len(stats['images_with_no_detections'])}
- Images with >50 Detections: {len(stats['images_with_many_detections'])}

CONFIDENCE SCORES:"""
    
    if stats['confidence_scores']:
        summary_text += f"""
- Mean Confidence: {np.mean(stats['confidence_scores']):.4f}
- Median Confidence: {np.median(stats['confidence_scores']):.4f}
- Min Confidence: {min(stats['confidence_scores']):.4f}
- Max Confidence: {max(stats['confidence_scores']):.4f}"""
    else:
        summary_text += """
- No confidence scores available"""
    
    summary_text += """

CATEGORY BREAKDOWN:
"""
    for cat, count in stats['categories'].items():
        percentage = (count / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
        summary_text += f"- {cat}: {count} ({percentage:.1f}%)\n"
    
    summary_text += f"\nBOX SIZE STATISTICS:\n"
    if stats['bbox_areas']:
        summary_text += f"- Mean Area: {np.mean(stats['bbox_areas']):.0f} px²\n"
        summary_text += f"- Median Area: {np.median(stats['bbox_areas']):.0f} px²\n"
        summary_text += f"- Min Area: {min(stats['bbox_areas']):.0f} px²\n"
        summary_text += f"- Max Area: {max(stats['bbox_areas']):.0f} px²\n"
    
    if stats['images_with_no_detections']:
        summary_text += f"\nIMAGES WITH NO DETECTIONS ({len(stats['images_with_no_detections'])}):\n"
        for img_name in stats['images_with_no_detections'][:10]:
            summary_text += f"- {img_name}\n"
        if len(stats['images_with_no_detections']) > 10:
            summary_text += f"... and {len(stats['images_with_no_detections']) - 10} more\n"
    
    if stats['images_with_many_detections']:
        summary_text += f"\nIMAGES WITH MANY DETECTIONS (>50):\n"
        for img_name, count in sorted(stats['images_with_many_detections'], key=lambda x: x[1], reverse=True)[:10]:
            summary_text += f"- {img_name}: {count} detections\n"
    
    with open(save_dir / 'summary_statistics.txt', 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    print(f"\nSaved summary statistics to {save_dir / 'summary_statistics.txt'}")

def main():
    # Paths
    predictions_path = Path('optimized_predictions.json')
    ground_truth_path = Path('train_annotations.json')
    evaluation_images_dir = Path('evaluation_images')
    output_dir = Path('yolo_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("YOLO MODEL VISUALIZATION AND ANALYSIS")
    print("="*60)
    
    # Load predictions
    print("\n1. Loading predictions...")
    predictions_data = load_predictions(predictions_path)
    
    # Load ground truth (if available)
    print("2. Loading ground truth...")
    gt_dict = {}
    if ground_truth_path.exists():
        gt_dict = load_ground_truth(ground_truth_path)
    
    # Analyze predictions
    print("\n3. Analyzing predictions...")
    stats = analyze_predictions(predictions_data)
    
    # Plot statistics
    print("\n4. Generating statistics plots...")
    plot_statistics(stats, output_dir / 'statistics')
    
    # Visualize sample images
    print("\n5. Visualizing sample predictions...")
    
    # Sort images by number of detections
    images_sorted = sorted(predictions_data.get('images', []), 
                          key=lambda x: len(x.get('annotations', [])))
    
    # Select diverse samples
    samples = []
    
    # Images with no detections (first 3)
    samples.extend([(img, "No Detections") for img in images_sorted[:3] if len(img.get('annotations', [])) == 0])
    
    # Images with few detections (3 samples around 25th percentile)
    idx_25 = len(images_sorted) // 4
    samples.extend([(img, "Few Detections") for img in images_sorted[max(0, idx_25-1):idx_25+2]])
    
    # Images with median detections (3 samples around median)
    idx_50 = len(images_sorted) // 2
    samples.extend([(img, "Median Detections") for img in images_sorted[max(0, idx_50-1):idx_50+2]])
    
    # Images with many detections (last 3)
    samples.extend([(img, "Many Detections") for img in images_sorted[-3:]])
    
    # Create visualizations
    sample_dir = output_dir / 'sample_predictions'
    sample_dir.mkdir(exist_ok=True)
    
    for i, (img_data, label) in enumerate(samples[:15]):
        img_name = img_data['file_name']
        img_path = evaluation_images_dir / img_name
        
        if not img_path.exists():
            continue
        
        annotations = img_data.get('annotations', [])
        save_path = sample_dir / f"{i+1:02d}_{label.replace(' ', '_')}_{img_name.replace('.tif', '.png')}"
        
        # If we have ground truth, create comparison
        if img_name in gt_dict:
            compare_predictions_gt(img_path, annotations, gt_dict[img_name], save_path)
        else:
            visualize_predictions(img_path, annotations, save_path, 
                                 title=f"{label}: {img_name}")
    
    print(f"\n6. All visualizations saved to: {output_dir}")
    print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
    print("-" * 60)
    
    # Provide recommendations based on statistics
    avg_detections = np.mean(stats['detections_per_image'])
    
    if avg_detections < 100:
        print("⚠️  LOW DETECTION COUNT: Your model is detecting fewer objects than expected.")
        print("   Consider:")
        print("   - Lowering confidence threshold (currently seems conservative)")
        print("   - Training for more epochs")
        print("   - Using a larger YOLO model (e.g., yolov8m-seg or yolov8l-seg)")
    
    if len(stats['images_with_no_detections']) > stats['total_images'] * 0.1:
        print(f"\n⚠️  MISSING DETECTIONS: {len(stats['images_with_no_detections'])} images have no detections")
        print("   This suggests the model might be missing small or difficult trees")
        print("   Consider:")
        print("   - Augmenting training data with more diverse examples")
        print("   - Adjusting image preprocessing")
    
    if stats['confidence_scores']:
        avg_conf = np.mean(stats['confidence_scores'])
        if avg_conf < 0.3:
            print(f"\n⚠️  LOW CONFIDENCE: Average confidence is {avg_conf:.3f}")
            print("   Consider:")
            print("   - Training longer to improve model confidence")
            print("   - Checking if the model is well-suited for this task")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the visualizations folder for detailed insights.")
    print("="*60)

if __name__ == "__main__":
    main()
