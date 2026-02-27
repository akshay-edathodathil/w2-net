import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(boxA, boxB):
    # Standard IoU calculation [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def match_detections(tracks, detections, threshold=0.8):
    """
    Calculates cost matrix and uses Hungarian algorithm to match.
    tracks: list of dicts {'id': int, 'pred_bbox': list, 'signature': array, 'reliability': float}
    detections: list of dicts {'bbox': list, 'signature': array}
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    cost_matrix = np.zeros((len(tracks), len(detections)))
    
    for i, trk in enumerate(tracks):
        # Adaptive gating: if flow is highly reliable, trust motion dist more
        alpha = 1.0 - (0.5 * trk['reliability']) 
        beta = 1.0 - alpha
        
        for j, det in enumerate(detections):
            iou = calculate_iou(trk['pred_bbox'], det['bbox'])
            
            # Motion signature L2 distance
            sig_dist = np.linalg.norm(trk['signature'] - det['signature'])
            
            # The custom cost function
            cost_matrix[i, j] = alpha * (1 - iou) + beta * sig_dist

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches, unmatched_tracks, unmatched_dets = [], [], []
    
    # Filter by threshold
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:
            matches.append((r, c))
        else:
            unmatched_tracks.append(r)
            unmatched_dets.append(c)
            
    # Add tracks/dets that weren't in the assignment at all
    unmatched_tracks.extend(list(set(range(len(tracks))) - set(row_ind)))
    unmatched_dets.extend(list(set(range(len(detections))) - set(col_ind)))
    
    return matches, unmatched_tracks, unmatched_dets