"""
Multi-Object Tracker with Flow/Jacobian-Based Association

Implements:
- ByteTrack-style two-stage matching (high conf, low conf, unconfirmed)
- Custom cost: α(1 - IoU) + β * signature_distance
- Adaptive weighting based on motion reliability
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from flow_jacobian_extractor import FlowJacobianExtractor, signature_distance
from motion_predictor import JacobianMotionPredictor, iou
from mot_utils import Track, Detection
from collections import defaultdict


class FlowTracker:
    """Multi-object tracker using flow/Jacobian features."""
    
    def __init__(
        self,
        flow_model: str = 'farneback',
        device: str = 'cpu',
        use_jacobian: bool = True,
        high_conf_thresh: float = 0.35,
        low_conf_thresh: float = 0.05,
        match_thresh: float = 0.8,
        iou_weight: float = 0.85,
        sig_weight: float = 0.15,
        max_age: int = 60,
        min_hits: int = 1
    ):
        """
        Args:
            flow_model: 'raft' or 'farneback'
            device: 'cuda' or 'cpu'
            use_jacobian: use full Jacobian or just translation
            high_conf_thresh: threshold for high-confidence detections
            low_conf_thresh: threshold for low-confidence detections
            match_thresh: maximum cost for valid match
            iou_weight: α in cost function
            sig_weight: β in cost function
            max_age: max frames to keep lost track
            min_hits: min detections before outputting track
        """
        self.flow_extractor = FlowJacobianExtractor(flow_model, device)
        self.motion_predictor = JacobianMotionPredictor(use_full_jacobian=use_jacobian)
        
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.match_thresh = match_thresh
        self.iou_weight = iou_weight
        self.sig_weight = sig_weight
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Tracking state
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        
        # Cache for previous frame
        self.prev_img = None
        self.prev_flow = None
        self.prev_jacobian = None
    
    def update(
        self,
        img: np.ndarray,
        detections: List[Detection]
    ) -> List[Track]:
        """
        Update tracker with new frame.
        
        Args:
            img: HxWx3 RGB image
            detections: list of Detection objects
            
        Returns:
            active_tracks: list of Track objects that should be output
        """
        self.frame_count += 1
        
        # Compute flow and signatures
        if self.prev_img is not None:
            flow, jacobian, det_signatures = self._compute_signatures(img, detections)
        else:
            # First frame: no flow available
            flow, jacobian = None, None
            det_signatures = [self.flow_extractor._empty_signature() for _ in detections]
        
        # Predict track positions
        self._predict_tracks()
        
        # Split detections by confidence
        high_dets, low_dets, high_sigs, low_sigs = self._split_detections(
            detections, det_signatures
        )
        
        # Stage 1: Match high-confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks if len(t.frame_ids) >= self.min_hits]
        unmatched_tracks, unmatched_dets = self._match(
            confirmed_tracks, high_dets, high_sigs, stage='high'
        )
        
        # Stage 2: Match remaining tracks with low-confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        unmatched_tracks2, unmatched_low_dets = self._match(
            remaining_tracks, low_dets, low_sigs, stage='low'
        )
        
        # Stage 3: Match unconfirmed tracks with remaining high-conf detections
        unconfirmed_tracks = [t for t in self.tracks if len(t.frame_ids) < self.min_hits]
        remaining_high_dets = [high_dets[i] for i in unmatched_dets]
        remaining_high_sigs = [high_sigs[i] for i in unmatched_dets]
        
        if remaining_high_dets:
            self._match(unconfirmed_tracks, remaining_high_dets, remaining_high_sigs, stage='unconf')
        
        # Initialize new tracks from unmatched high-conf detections
        for i in unmatched_dets:
            self._init_track(high_dets[i], high_sigs[i])
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if self._is_alive(t)]
        
        # Update cache
        self.prev_img = img.copy()
        self.prev_flow = flow
        self.prev_jacobian = jacobian
        
        # Return tracks that should be output
        active_tracks = [t for t in self.tracks if len(t.frame_ids) >= self.min_hits]
        return active_tracks
    
    def _compute_signatures(
        self,
        img: np.ndarray,
        detections: List[Detection]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Compute flow and extract signatures for detections."""
        bboxes = np.array([d.bbox for d in detections])
        
        if len(bboxes) == 0:
            flow = self.flow_extractor.compute_flow(self.prev_img, img)
            jacobian = self.flow_extractor.compute_jacobian(flow)
            return flow, jacobian, []
        
        flow, jacobian, signatures = self.flow_extractor.batch_extract(
            self.prev_img, img, bboxes
        )
        
        return flow, jacobian, signatures
    
    def _predict_tracks(self):
        """Predict next position for all tracks."""
        for track in self.tracks:
            if track.signatures:
                sig = track.get_last_signature()
                bbox_pred = self.motion_predictor.predict_bbox(
                    track.get_last_bbox(), sig
                )
                # Store prediction (but don't add to track yet)
                track.bbox_pred = bbox_pred
                track.uncertainty = self.motion_predictor.compute_motion_uncertainty(sig)
            else:
                # No signature: use last bbox
                track.bbox_pred = track.get_last_bbox()
                track.uncertainty = 1.0
    
    def _split_detections(
        self,
        detections: List[Detection],
        signatures: List[Dict]
    ) -> Tuple[List[Detection], List[Detection], List[Dict], List[Dict]]:
        """Split detections into high and low confidence."""
        high_dets, low_dets = [], []
        high_sigs, low_sigs = [], []
        
        for det, sig in zip(detections, signatures):
            if det.confidence >= self.high_conf_thresh:
                high_dets.append(det)
                high_sigs.append(sig)
            elif det.confidence >= self.low_conf_thresh:
                low_dets.append(det)
                low_sigs.append(sig)
        
        return high_dets, low_dets, high_sigs, low_sigs
    
    def _match(
        self,
        tracks: List[Track],
        detections: List[Detection],
        signatures: List[Dict],
        stage: str
    ) -> Tuple[List[int], List[int]]:
        """
        Match tracks to detections using Hungarian algorithm.
        
        Returns:
            unmatched_tracks: indices into tracks
            unmatched_dets: indices into detections
        """
        if not tracks or not detections:
            return list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            bbox_pred = track.bbox_pred
            sig_track = track.get_last_signature() if track.signatures else {}
            uncertainty = getattr(track, 'uncertainty', 0.5)
            
            for j, (det, sig_det) in enumerate(zip(detections, signatures)):
                # IoU cost
                iou_val = iou(bbox_pred, det.bbox)
                cost_iou = 1.0 - iou_val
                
                # Signature distance
                if sig_track and sig_det:
                    dist_sig = signature_distance(sig_track, sig_det)
                else:
                    dist_sig = 0.0
                
                # Adaptive weighting based on uncertainty
                # High uncertainty → rely more on IoU
                alpha = self.iou_weight + uncertainty * 0.2
                beta = self.sig_weight - uncertainty * 0.2
                alpha = np.clip(alpha, 0.5, 0.9)
                beta = np.clip(beta, 0.1, 0.5)
                
                cost = alpha * cost_iou + beta * dist_sig
                cost_matrix[i, j] = cost
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        matches = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.match_thresh:
                matches.append((i, j))
                # Update track
                track = tracks[i]
                det = detections[j]
                sig = signatures[j]
                track.add_detection(det.bbox, self.frame_count, det.confidence, sig)
                track.age = 0  # reset age
        
        # Find unmatched
        matched_track_ids = set([m[0] for m in matches])
        matched_det_ids = set([m[1] for m in matches])
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_ids]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_ids]
        
        # Increment age for unmatched tracks
        for i in unmatched_tracks:
            if not hasattr(tracks[i], 'age'):
                tracks[i].age = 0
            tracks[i].age += 1
        
        return unmatched_tracks, unmatched_dets
    
    def _init_track(self, detection: Detection, signature: Dict):
        """Initialize new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            bboxes=[detection.bbox],
            frame_ids=[self.frame_count],
            confidences=[detection.confidence],
            signatures=[signature]
        )
        track.age = 0
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def _is_alive(self, track: Track) -> bool:
        """Check if track should be kept."""
        age = getattr(track, 'age', 0)
        return age < self.max_age
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        self.prev_img = None
        self.prev_flow = None
        self.prev_jacobian = None


def compute_track_metrics(
    pred_tracks: List[Track],
    gt_tracks: Dict[int, List[np.ndarray]],
    iou_thresh: float = 0.5
) -> Dict[str, float]:
    """
    Compute basic tracking metrics (simplified).
    
    Args:
        pred_tracks: list of predicted Track objects
        gt_tracks: dict mapping frame_id -> list of gt bboxes
        iou_thresh: IoU threshold for match
        
    Returns:
        metrics: dict with MOTA, MOTP, etc.
    """
    # This is a simplified version; use TrackEval for full metrics
    
    total_gt = 0
    total_pred = 0
    total_matches = 0
    
    for track in pred_tracks:
        for bbox, frame_id in zip(track.bboxes, track.frame_ids):
            total_pred += 1
            
            if frame_id in gt_tracks:
                gt_bboxes = gt_tracks[frame_id]
                total_gt += len(gt_bboxes)
                
                # Find best match
                max_iou = 0
                for gt_bbox in gt_bboxes:
                    iou_val = iou(bbox, gt_bbox)
                    max_iou = max(max_iou, iou_val)
                
                if max_iou >= iou_thresh:
                    total_matches += 1
    
    precision = total_matches / (total_pred + 1e-6)
    recall = total_matches / (total_gt + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'total_matches': total_matches
    }


if __name__ == '__main__':
    print("Testing FlowTracker...")
    
    # Dummy test
    tracker = FlowTracker(
        flow_model='farneback',
        device='cpu',
        high_conf_thresh=0.6,
        match_thresh=0.8
    )
    
    # Simulate two frames
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Frame 1: 2 detections
    dets1 = [
        Detection(np.array([100, 100, 200, 200]), 0.9, 1, 0),
        Detection(np.array([300, 300, 400, 400]), 0.8, 1, 1)
    ]
    
    # Frame 2: 2 detections (slightly moved)
    dets2 = [
        Detection(np.array([105, 102, 205, 202]), 0.85, 2, 0),
        Detection(np.array([310, 305, 410, 405]), 0.75, 2, 1)
    ]
    
    active1 = tracker.update(img1, dets1)
    print(f"Frame 1: {len(active1)} active tracks")
    
    active2 = tracker.update(img2, dets2)
    print(f"Frame 2: {len(active2)} active tracks")
    
    for track in active2:
        print(f"  Track {track.track_id}: {len(track.frame_ids)} frames")
