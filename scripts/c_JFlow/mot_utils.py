"""
MOT17 Dataset Utilities

Handles:
- Loading MOT17 sequences (images + detections)
- Reading/writing MOT format files
- Data structures for tracking
"""

import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection in a frame."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    frame_id: int
    detection_id: int  # unique ID within frame


@dataclass
class Track:
    """A track across frames."""
    track_id: int
    bboxes: List[np.ndarray]  # list of [x1, y1, x2, y2]
    frame_ids: List[int]
    confidences: List[float]
    signatures: List[Dict]  # motion signatures
    state: Optional[np.ndarray] = None  # for Kalman filter
    covariance: Optional[np.ndarray] = None
    
    def get_last_bbox(self) -> np.ndarray:
        return self.bboxes[-1]
    
    def get_last_signature(self) -> Dict:
        return self.signatures[-1] if self.signatures else {}
    
    def add_detection(
        self,
        bbox: np.ndarray,
        frame_id: int,
        confidence: float,
        signature: Optional[Dict] = None
    ):
        self.bboxes.append(bbox)
        self.frame_ids.append(frame_id)
        self.confidences.append(confidence)
        if signature is not None:
            self.signatures.append(signature)


class MOTDataset:
    """Loads and manages MOT17 dataset sequences."""
    
    def __init__(self, data_root: str, sequence: str, split: str = 'train'):
        """
        Args:
            data_root: path to MOT17 root directory
            sequence: sequence name (e.g., 'MOT17-02-FRCNN')
            split: 'train' or 'test'
        """
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.split = split
        
        self.seq_path = self.data_root / split / sequence
        self.img_path = self.seq_path / 'img1'
        self.det_path = self.seq_path / 'det' / 'det.txt'
        self.gt_path = self.seq_path / 'gt' / 'gt.txt'
        
        # Load sequence info
        self.seqinfo = self._load_seqinfo()
        self.num_frames = self.seqinfo['seqLength']
        self.frame_rate = self.seqinfo['frameRate']
        self.img_width = self.seqinfo['imWidth']
        self.img_height = self.seqinfo['imHeight']
        
        # Load detections
        self.detections = self._load_detections()
    
    def _load_seqinfo(self) -> Dict:
        """Parse seqinfo.ini file."""
        seqinfo_path = self.seq_path / 'seqinfo.ini'
        info = {}
        
        with open(seqinfo_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    # Try to parse as int
                    try:
                        info[key] = int(value)
                    except ValueError:
                        info[key] = value
        
        return info
    
    def _load_detections(self) -> Dict[int, List[Detection]]:
        """
        Load detection file.
        
        MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        
        Returns:
            detections: dict mapping frame_id -> list of Detection objects
        """
        detections = {}
        
        if not self.det_path.exists():
            print(f"Warning: {self.det_path} not found")
            return detections
        
        with open(self.det_path, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                det_id = int(parts[1])  # usually -1 for detections
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Convert to [x1, y1, x2, y2]
                bbox = np.array([x, y, x + w, y + h])
                
                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    frame_id=frame_id,
                    detection_id=idx
                )
                
                if frame_id not in detections:
                    detections[frame_id] = []
                detections[frame_id].append(det)
        
        return detections
    
    def get_image(self, frame_id: int) -> np.ndarray:
        """
        Load image for a frame.
        
        Args:
            frame_id: 1-indexed frame number
            
        Returns:
            img: HxWx3 uint8 RGB image
        """
        img_file = self.img_path / f'{frame_id:06d}.jpg'
        
        if not img_file.exists():
            raise FileNotFoundError(f"Image not found: {img_file}")
        
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def get_detections(self, frame_id: int) -> List[Detection]:
        """Get all detections for a frame."""
        return self.detections.get(frame_id, [])
    
    def get_detections_array(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get detections as arrays.
        
        Returns:
            bboxes: Nx4 array
            confidences: N array
        """
        dets = self.get_detections(frame_id)
        if not dets:
            return np.empty((0, 4)), np.empty((0,))
        
        bboxes = np.array([d.bbox for d in dets])
        confidences = np.array([d.confidence for d in dets])
        
        return bboxes, confidences


class MOTWriter:
    """Write tracking results in MOT format."""
    
    def __init__(self, output_path: str):
        """
        Args:
            output_path: path to output .txt file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_track(
        self,
        frame_id: int,
        track_id: int,
        bbox: np.ndarray,
        confidence: float = 1.0
    ):
        """
        Add a track result for a frame.
        
        Args:
            frame_id: 1-indexed frame number
            track_id: track ID
            bbox: [x1, y1, x2, y2]
            confidence: detection confidence
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
        result = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{confidence:.2f},-1,-1,-1"
        self.results.append(result)
    
    def write(self):
        """Write all results to file."""
        with open(self.output_path, 'w') as f:
            for result in self.results:
                f.write(result + '\n')
        
        print(f"Wrote {len(self.results)} tracking results to {self.output_path}")
    
    def clear(self):
        """Clear stored results."""
        self.results = []


def load_mot_gt(gt_path: str) -> Dict[int, List[np.ndarray]]:
    """
    Load ground truth annotations.
    
    Args:
        gt_path: path to gt.txt
        
    Returns:
        gt: dict mapping frame_id -> list of bboxes (only consider = 1, visible = 1)
    """
    gt = {}
    
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            cls = int(parts[7])  # 1 = pedestrian
            visibility = float(parts[8])
            
            # Only keep pedestrians with good visibility
            if cls != 1 or visibility < 0.25:
                continue
            
            bbox = np.array([x, y, x + w, y + h])
            
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append(bbox)
    
    return gt


if __name__ == '__main__':
    # Test loading a sequence
    print("Testing MOTDataset...")
    
    # This will fail without actual MOT17 data, but shows the API
    try:
        dataset = MOTDataset(
            data_root='/path/to/MOT17',
            sequence='MOT17-02-FRCNN',
            split='train'
        )
        
        print(f"Sequence: {dataset.sequence}")
        print(f"Frames: {dataset.num_frames}")
        print(f"Frame rate: {dataset.frame_rate}")
        print(f"Image size: {dataset.img_width}x{dataset.img_height}")
        
        # Get detections for frame 1
        dets = dataset.get_detections(1)
        print(f"Detections in frame 1: {len(dets)}")
        
        # Test writer
        writer = MOTWriter('/tmp/test_output.txt')
        writer.add_track(1, 1, np.array([100, 100, 200, 200]), 0.9)
        writer.add_track(2, 1, np.array([105, 102, 205, 202]), 0.9)
        writer.write()
        
    except Exception as e:
        print(f"Expected error (no data): {e}")
        print("API demonstrated successfully.")
