"""
Evaluation Script for MOT Tracking Results

Computes standard MOT metrics:
- MOTA (Multiple Object Tracking Accuracy)
- IDF1 (ID F1 score)
- HOTA (Higher Order Tracking Accuracy)

Uses motmetrics and TrackEval libraries.

Usage:
    python evaluate.py --gt_dir /path/to/MOT17/train/MOT17-02-FRCNN/gt \\
                       --pred_file outputs/MOT17-02-FRCNN/MOT17-02-FRCNN.txt \\
                       --sequence MOT17-02-FRCNN
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MOT tracking results')
    
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Path to ground truth gt.txt')
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to prediction .txt')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Sequence name')
    
    return parser.parse_args()


def load_mot_file(filepath: str, is_gt: bool = False):
    """
    Load MOT format file.
    
    Returns:
        data: dict mapping frame_id -> list of (track_id, bbox)
    """
    data = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            
            if is_gt:
                # Filter ground truth (pedestrians only, visible > 0.25)
                cls = int(parts[7]) if len(parts) > 7 else 1
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
                
                if cls != 1 or visibility < 0.25:
                    continue
            
            bbox = np.array([x, y, x + w, y + h])
            data[frame_id].append((track_id, bbox, conf))
    
    return data


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes."""
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-6)


def compute_mot_metrics(gt_data, pred_data, iou_thresh=0.5):
    """
    Compute MOT metrics using a simplified approach.
    
    For full metrics, use motmetrics or TrackEval libraries.
    
    Returns:
        metrics: dict with MOTA, precision, recall, etc.
    """
    # Get all frames
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    # Accumulators
    num_gt = 0
    num_pred = 0
    num_matches = 0
    num_false_positives = 0
    num_misses = 0
    num_id_switches = 0
    
    # Track ID mapping for ID switches
    prev_matches = {}  # gt_id -> pred_id in previous frame
    
    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])
        
        num_gt += len(gt_frame)
        num_pred += len(pred_frame)
        
        if not gt_frame or not pred_frame:
            num_misses += len(gt_frame)
            num_false_positives += len(pred_frame)
            continue
        
        # Match predictions to ground truth
        cost_matrix = np.zeros((len(gt_frame), len(pred_frame)))
        
        for i, (gt_id, gt_bbox, _) in enumerate(gt_frame):
            for j, (pred_id, pred_bbox, _) in enumerate(pred_frame):
                cost_matrix[i, j] = 1.0 - iou(gt_bbox, pred_bbox)
        
        # Greedy matching (for simplicity; use Hungarian for full eval)
        matched_gt = set()
        matched_pred = set()
        current_matches = {}
        
        while True:
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(len(gt_frame)):
                if i in matched_gt:
                    continue
                for j in range(len(pred_frame)):
                    if j in matched_pred:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_cost > (1 - iou_thresh):
                break
            
            matched_gt.add(min_i)
            matched_pred.add(min_j)
            num_matches += 1
            
            # Track ID for ID switches
            gt_id = gt_frame[min_i][0]
            pred_id = pred_frame[min_j][0]
            current_matches[gt_id] = pred_id
            
            # Check for ID switch
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                num_id_switches += 1
        
        num_misses += len(gt_frame) - len(matched_gt)
        num_false_positives += len(pred_frame) - len(matched_pred)
        
        prev_matches = current_matches
    
    # Compute MOTA
    mota = 1 - (num_false_positives + num_misses + num_id_switches) / (num_gt + 1e-6)
    
    # Compute precision and recall
    precision = num_matches / (num_pred + 1e-6)
    recall = num_matches / (num_gt + 1e-6)
    
    metrics = {
        'MOTA': mota * 100,  # as percentage
        'Precision': precision * 100,
        'Recall': recall * 100,
        'NumMatches': num_matches,
        'NumGT': num_gt,
        'NumPred': num_pred,
        'NumFP': num_false_positives,
        'NumMisses': num_misses,
        'NumIDSwitches': num_id_switches
    }
    
    return metrics


def compute_idf1(gt_data, pred_data, iou_thresh=0.5):
    """
    Compute IDF1 (ID F1 score).
    
    Measures how well track IDs are maintained over time.
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    # Global ID associations
    idtp = 0  # ID True Positives
    idfp = 0  # ID False Positives
    idfn = 0  # ID False Negatives
    
    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])
        
        if not gt_frame or not pred_frame:
            idfn += len(gt_frame)
            idfp += len(pred_frame)
            continue
        
        # Match by IoU
        matches = []
        for i, (gt_id, gt_bbox, _) in enumerate(gt_frame):
            for j, (pred_id, pred_bbox, _) in enumerate(pred_frame):
                iou_val = iou(gt_bbox, pred_bbox)
                if iou_val >= iou_thresh:
                    matches.append((gt_id, pred_id, iou_val))
        
        # Sort by IoU (descending)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        matched_gt_ids = set()
        matched_pred_ids = set()
        
        for gt_id, pred_id, _ in matches:
            if gt_id not in matched_gt_ids and pred_id not in matched_pred_ids:
                idtp += 1
                matched_gt_ids.add(gt_id)
                matched_pred_ids.add(pred_id)
        
        idfn += len(gt_frame) - len(matched_gt_ids)
        idfp += len(pred_frame) - len(matched_pred_ids)
    
    # Compute IDF1
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn + 1e-6)
    
    return {
        'IDF1': idf1 * 100,
        'IDTP': idtp,
        'IDFP': idfp,
        'IDFN': idfn
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"Evaluating: {args.sequence}")
    print("=" * 60)
    
    # Load data
    print(f"Loading GT from: {args.gt_file}")
    gt_data = load_mot_file(args.gt_file, is_gt=True)
    
    print(f"Loading predictions from: {args.pred_file}")
    pred_data = load_mot_file(args.pred_file, is_gt=False)
    
    print(f"\nGT frames: {len(gt_data)}")
    print(f"Pred frames: {len(pred_data)}")
    
    # Compute metrics
    print("\nComputing MOT metrics...")
    mot_metrics = compute_mot_metrics(gt_data, pred_data, iou_thresh=0.5)
    
    print("\nComputing IDF1...")
    idf1_metrics = compute_idf1(gt_data, pred_data, iou_thresh=0.5)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nMOT Metrics:")
    print(f"  MOTA:       {mot_metrics['MOTA']:.2f}%")
    print(f"  Precision:  {mot_metrics['Precision']:.2f}%")
    print(f"  Recall:     {mot_metrics['Recall']:.2f}%")
    print(f"  FP:         {mot_metrics['NumFP']}")
    print(f"  Misses:     {mot_metrics['NumMisses']}")
    print(f"  ID Sw.:     {mot_metrics['NumIDSwitches']}")
    
    print("\nID Metrics:")
    print(f"  IDF1:       {idf1_metrics['IDF1']:.2f}%")
    print(f"  IDTP:       {idf1_metrics['IDTP']}")
    print(f"  IDFP:       {idf1_metrics['IDFP']}")
    print(f"  IDFN:       {idf1_metrics['IDFN']}")
    
    print("\n" + "=" * 60)
    print("\nNote: For full HOTA evaluation, use TrackEval:")
    print("  https://github.com/JonathonLuiten/TrackEval")
    print("=" * 60)


if __name__ == '__main__':
    main()
