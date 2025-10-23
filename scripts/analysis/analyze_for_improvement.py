"""
Deep analysis script to understand why YOLO is stuck at 0.2 score
and provide actionable recommendations for improvement
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def polygon_area(points):
    """Calculate area of polygon using shoelace formula"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_iou(poly1, poly2):
    """Approximate IoU between two polygons using bounding boxes"""
    # Get bounding boxes
    x1_min, y1_min = poly1.min(axis=0)
    x1_max, y1_max = poly1.max(axis=0)
    x2_min, y2_min = poly2.min(axis=0)
    x2_max, y2_max = poly2.max(axis=0)
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_predictions_vs_ground_truth():
    """Compare predictions with training annotations to understand the gap"""
    
    print("="*80)
    print("DETAILED ANALYSIS: WHY IS YOUR MODEL STUCK AT 0.2?")
    print("="*80)
    
    # Load data
    pred_path = Path('optimized_predictions.json')
    gt_path = Path('train_annotations.json')
    
    if not pred_path.exists():
        print("❌ optimized_predictions.json not found!")
        return
    
    if not gt_path.exists():
        print("⚠️  train_annotations.json not found - skipping GT comparison")
        gt_data = None
    else:
        gt_data = load_json(gt_path)
    
    pred_data = load_json(pred_path)
    
    # Analyze predictions structure
    print("\n" + "="*80)
    print("1. PREDICTION STRUCTURE ANALYSIS")
    print("="*80)
    
    pred_images = pred_data.get('images', [])
    print(f"Total prediction images: {len(pred_images)}")
    
    # Check a sample prediction
    if pred_images:
        sample_img = pred_images[0]
        print(f"\nSample image: {sample_img.get('file_name', 'N/A')}")
        sample_anns = sample_img.get('annotations', [])
        print(f"Number of annotations: {len(sample_anns)}")
        
        if sample_anns:
            sample_ann = sample_anns[0]
            print("\nSample annotation structure:")
            for key, value in sample_ann.items():
                if key == 'segmentation':
                    try:
                        coord_len = len(value[0]) if value and value[0] else 0
                        print(f"  - {key}: [list with {coord_len} coordinates]")
                    except:
                        print(f"  - {key}: {type(value)}")
                else:
                    print(f"  - {key}: {value}")
    
    # Analyze ground truth
    if gt_data:
        print("\n" + "="*80)
        print("2. GROUND TRUTH ANALYSIS")
        print("="*80)
        
        gt_images = gt_data.get('images', [])
        print(f"Total GT images: {len(gt_images)}")
        
        total_gt_anns = sum(len(img.get('annotations', [])) for img in gt_images)
        print(f"Total GT annotations: {total_gt_anns}")
        print(f"Average GT annotations per image: {total_gt_anns/len(gt_images):.1f}")
        
        # Category breakdown
        gt_categories = defaultdict(int)
        for img in gt_images:
            for ann in img.get('annotations', []):
                cat = ann.get('category_name', 'unknown')
                gt_categories[cat] += 1
        
        print("\nGT Category breakdown:")
        for cat, count in gt_categories.items():
            print(f"  - {cat}: {count} ({count/total_gt_anns*100:.1f}%)")
    
    # Detailed prediction analysis
    print("\n" + "="*80)
    print("3. PREDICTION QUALITY ANALYSIS")
    print("="*80)
    
    total_pred_anns = sum(len(img.get('annotations', [])) for img in pred_images)
    print(f"Total predictions: {total_pred_anns}")
    print(f"Average predictions per image: {total_pred_anns/len(pred_images):.1f}")
    
    # Analyze polygon sizes
    all_areas = []
    tiny_objects = 0
    huge_objects = 0
    
    for img in pred_images:
        for ann in img.get('annotations', []):
            seg = ann.get('segmentation', [[]])
            if seg and seg[0]:
                try:
                    points = np.array(seg[0]).reshape(-1, 2)
                    area = polygon_area(points)
                    all_areas.append(area)
                    
                    if area < 100:  # Very small
                        tiny_objects += 1
                    elif area > 50000:  # Very large
                        huge_objects += 1
                except:
                    pass
    
    if all_areas:
        print(f"\nPolygon area statistics:")
        print(f"  - Mean area: {np.mean(all_areas):.0f} px²")
        print(f"  - Median area: {np.median(all_areas):.0f} px²")
        print(f"  - Min area: {min(all_areas):.0f} px²")
        print(f"  - Max area: {max(all_areas):.0f} px²")
        print(f"  - Tiny objects (<100px²): {tiny_objects} ({tiny_objects/len(all_areas)*100:.1f}%)")
        print(f"  - Huge objects (>50000px²): {huge_objects} ({huge_objects/len(all_areas)*100:.1f}%)")
    
    # Check for potential issues
    print("\n" + "="*80)
    print("4. POTENTIAL ISSUES DETECTED")
    print("="*80)
    
    issues = []
    
    # Issue 1: Too many predictions
    if total_pred_anns > 30000:
        issues.append({
            'severity': 'HIGH',
            'issue': 'Excessive predictions',
            'details': f'You have {total_pred_anns} predictions. This suggests the model is over-detecting.',
            'solutions': [
                'Increase confidence threshold (try 0.25-0.35 instead of 0.1)',
                'Increase IoU threshold for NMS (try 0.7-0.8 instead of 0.6)',
                'The model might be creating overlapping detections'
            ]
        })
    
    # Issue 2: No confidence scores
    has_scores = any('score' in ann 
                     for img in pred_images 
                     for ann in img.get('annotations', []))
    if not has_scores:
        issues.append({
            'severity': 'MEDIUM',
            'issue': 'Missing confidence scores',
            'details': 'Your predictions don\'t include confidence scores',
            'solutions': [
                'Ensure you\'re saving the confidence scores from YOLO predictions',
                'Confidence scores help understand model certainty',
                'Add: ann["score"] = float(det.conf) in your inference code'
            ]
        })
    
    # Issue 3: Wrong category distribution
    pred_categories = defaultdict(int)
    for img in pred_images:
        for ann in img.get('annotations', []):
            cat = ann.get('category_name', 'unknown')
            pred_categories[cat] += 1
    
    if len(pred_categories) == 1 and 'individual_tree' in pred_categories:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Only detecting "individual_tree" class',
            'details': 'All predictions are "individual_tree", no "group_of_trees" detected',
            'solutions': [
                'Your model is not learning to distinguish between the two classes',
                'Check if training data has both classes properly labeled',
                'Consider using YOLOv8 classification head properly',
                'May need to retrain with better class balance or different architecture'
            ]
        })
    
    # Issue 4: Evaluation vs Training mismatch
    if gt_data:
        avg_gt = total_gt_anns / len(gt_images)
        avg_pred = total_pred_anns / len(pred_images)
        
        if abs(avg_gt - avg_pred) / avg_gt > 0.5:  # 50% difference
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Prediction count mismatch',
                'details': f'GT avg: {avg_gt:.1f} vs Pred avg: {avg_pred:.1f} per image',
                'solutions': [
                    'Significant mismatch suggests model is not well-calibrated',
                    'If predicting more: increase thresholds or improve NMS',
                    'If predicting less: lower thresholds or train longer'
                ]
            })
    
    # Issue 5: Max detections cap
    max_dets_per_img = max(len(img.get('annotations', [])) for img in pred_images)
    imgs_at_max = sum(1 for img in pred_images if len(img.get('annotations', [])) == 300)
    
    if imgs_at_max > 5:
        issues.append({
            'severity': 'HIGH',
            'issue': 'Hitting max detections limit',
            'details': f'{imgs_at_max} images have exactly 300 detections (likely a cap)',
            'solutions': [
                'YOLO has a max_det parameter (default 300)',
                'Increase it: model.predict(max_det=1000)',
                'Or better: improve NMS to reduce overlapping detections'
            ]
        })
    
    # Print issues
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n⚠️  ISSUE #{i} [{issue['severity']}]: {issue['issue']}")
            print(f"    Details: {issue['details']}")
            print(f"    Solutions:")
            for sol in issue['solutions']:
                print(f"      • {sol}")
    else:
        print("\n✅ No major issues detected in prediction structure")
    
    # Recommendations
    print("\n" + "="*80)
    print("5. SPECIFIC RECOMMENDATIONS TO IMPROVE FROM 0.2")
    print("="*80)
    
    recommendations = []
    
    recommendations.append({
        'priority': 1,
        'action': 'Fix class detection',
        'why': 'You\'re only detecting one class',
        'how': [
            'Check your training data has both classes',
            'Verify YOLO is using both classes (check data.yaml)',
            'Retrain with proper multi-class configuration',
            'Use YOLOv8 with cls parameter for classification'
        ]
    })
    
    recommendations.append({
        'priority': 2,
        'action': 'Reduce false positives',
        'why': '31,947 detections seems very high',
        'how': [
            'Increase confidence threshold to 0.25-0.35',
            'Increase IoU threshold to 0.7',
            'Run: python3 optimize_thresholds.py with wider range',
            'Add better NMS post-processing'
        ]
    })
    
    recommendations.append({
        'priority': 3,
        'action': 'Add confidence scores',
        'why': 'Helps evaluation metrics',
        'how': [
            'Modify inference script to save det.conf',
            'Format: {"score": float(det.conf)}',
            'This affects precision-recall calculations'
        ]
    })
    
    recommendations.append({
        'priority': 4,
        'action': 'Train longer / better',
        'why': 'Only 10 epochs might be insufficient',
        'how': [
            'Train for 50-100 epochs',
            'Use early stopping (patience=10)',
            'Try larger model: yolov8m-seg or yolov8l-seg',
            'Add more augmentation: mosaic, mixup'
        ]
    })
    
    recommendations.append({
        'priority': 5,
        'action': 'Validate on training set',
        'why': 'Check if model learned the task',
        'how': [
            'Run inference on training images',
            'Compare with ground truth visually',
            'Calculate metrics on training set',
            'Should get >0.8 score on training data'
        ]
    })
    
    print("\n📋 ACTION PLAN (ordered by priority):\n")
    for rec in recommendations:
        print(f"{rec['priority']}. {rec['action'].upper()}")
        print(f"   Why: {rec['why']}")
        print(f"   How:")
        for step in rec['how']:
            print(f"     • {step}")
        print()
    
    # Generate quick fix script
    print("="*80)
    print("6. QUICK FIX SCRIPT")
    print("="*80)
    
    quick_fix = """
# Quick fix to try immediately:

1. Adjust thresholds (run this):
   python3 inference_yolo.py --conf 0.3 --iou 0.7

2. If you're using the inference script, modify it:
   
   In inference_yolo.py, change:
   - self.conf_threshold = 0.3  # instead of 0.1
   - self.iou_threshold = 0.7   # instead of 0.6
   - Add: max_det=1000 in predict call
   - Save scores: ann['score'] = float(det.conf)

3. Retrain for more epochs:
   
   In train_yolo.py, change:
   - epochs = 50  # instead of 10
   - model = YOLO('yolov8m-seg.pt')  # larger model

4. Verify your data.yaml has both classes:
   
   names:
     0: individual_tree
     1: group_of_trees
"""
    
    print(quick_fix)
    
    # Save report
    report_path = Path('yolo_visualizations/improvement_report.txt')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("YOLO IMPROVEMENT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("ISSUES DETECTED:\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"\n{i}. [{issue['severity']}] {issue['issue']}\n")
            f.write(f"   {issue['details']}\n")
            f.write(f"   Solutions:\n")
            for sol in issue['solutions']:
                f.write(f"     • {sol}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        for rec in recommendations:
            f.write(f"\n{rec['priority']}. {rec['action']}\n")
            f.write(f"   Why: {rec['why']}\n")
            f.write(f"   How:\n")
            for step in rec['how']:
                f.write(f"     • {step}\n")
        
        f.write("\n" + quick_fix)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("SUMMARY: The main issue is likely class detection and threshold tuning")
    print("="*80)

if __name__ == "__main__":
    analyze_predictions_vs_ground_truth()
