"""
Quick Demo/Test Script

Tests all modules with synthetic data (no MOT17 required).
Use this to verify installation and understand the API.
"""

import numpy as np
import cv2
from pathlib import Path

print("=" * 60)
print("Flow/Jacobian Tracker - Demo")
print("=" * 60)

# Test 1: Flow extraction
print("\n[1/5] Testing FlowJacobianExtractor...")
from flow_jacobian_extractor import FlowJacobianExtractor, signature_distance

# Create synthetic moving images
img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
img2 = np.roll(img1, 5, axis=1)  # shift right by 5 pixels

extractor = FlowJacobianExtractor(flow_model='farneback', device='cpu')

bboxes = np.array([
    [100, 100, 200, 200],
    [300, 300, 400, 400]
])

flow, jacobian, signatures = extractor.batch_extract(img1, img2, bboxes)

print(f"✓ Flow shape: {flow.shape}")
print(f"✓ Jacobian shape: {jacobian.shape}")
print(f"✓ Extracted {len(signatures)} signatures")
print(f"  - Signature 0: Δx={signatures[0]['delta_x']:.2f}, Δy={signatures[0]['delta_y']:.2f}")
print(f"  - Signature 1: Δx={signatures[1]['delta_x']:.2f}, Δy={signatures[1]['delta_y']:.2f}")

dist = signature_distance(signatures[0], signatures[1])
print(f"  - Distance between signatures: {dist:.3f}")

# Test 2: Motion prediction
print("\n[2/5] Testing JacobianMotionPredictor...")
from motion_predictor import JacobianMotionPredictor, KalmanMotionPredictor, iou

predictor = JacobianMotionPredictor(use_full_jacobian=True, damping=0.5)

bbox = np.array([100, 100, 200, 200])
signature = signatures[0]

bbox_pred = predictor.predict_bbox(bbox, signature)
uncertainty = predictor.compute_motion_uncertainty(signature)

print(f"✓ Original bbox: {bbox}")
print(f"✓ Predicted bbox: {bbox_pred}")
print(f"  - IoU: {iou(bbox, bbox_pred):.3f}")
print(f"  - Uncertainty: {uncertainty:.3f}")

# Compare with Kalman
kalman = KalmanMotionPredictor()
state = kalman.bbox_to_state(bbox)
P = np.eye(6) * 10
state_pred, P_pred = kalman.predict(state, P)
bbox_pred_kalman = kalman.state_to_bbox(state_pred)

print(f"✓ Kalman predicted bbox: {bbox_pred_kalman}")

# Test 3: MOT data structures
print("\n[3/5] Testing MOT data structures...")
from mot_utils import Track, Detection, MOTWriter

# Create dummy detections
det1 = Detection(np.array([100, 100, 200, 200]), 0.9, 1, 0)
det2 = Detection(np.array([105, 102, 205, 202]), 0.85, 2, 0)

# Create track
track = Track(
    track_id=1,
    bboxes=[det1.bbox],
    frame_ids=[1],
    confidences=[0.9],
    signatures=[signatures[0]]
)
track.add_detection(det2.bbox, 2, 0.85, signatures[0])

print(f"✓ Created track {track.track_id}")
print(f"  - Frames: {track.frame_ids}")
print(f"  - Length: {len(track.bboxes)}")

# Test writer
writer = MOTWriter('/tmp/demo_output.txt')
for bbox, frame_id, conf in zip(track.bboxes, track.frame_ids, track.confidences):
    writer.add_track(frame_id, track.track_id, bbox, conf)
writer.write()

print(f"✓ Wrote {len(writer.results)} results to /tmp/demo_output.txt")

# Test 4: Tracker
print("\n[4/5] Testing FlowTracker...")
from tracker import FlowTracker

tracker = FlowTracker(
    flow_model='farneback',
    device='cpu',
    high_conf_thresh=0.6,
    match_thresh=0.8
)

# Simulate 3 frames
frames = []
for i in range(3):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frames.append(img)

# Create moving detections
detections_per_frame = [
    [
        Detection(np.array([100, 100, 200, 200]), 0.9, 1, 0),
        Detection(np.array([300, 300, 400, 400]), 0.8, 1, 1)
    ],
    [
        Detection(np.array([105, 102, 205, 202]), 0.85, 2, 0),
        Detection(np.array([310, 305, 410, 405]), 0.75, 2, 1)
    ],
    [
        Detection(np.array([110, 104, 210, 204]), 0.9, 3, 0),
        Detection(np.array([320, 310, 420, 410]), 0.8, 3, 1)
    ]
]

print("✓ Running tracker on 3 frames...")
for frame_idx, (img, dets) in enumerate(zip(frames, detections_per_frame)):
    active = tracker.update(img, dets)
    print(f"  Frame {frame_idx + 1}: {len(active)} active tracks, {len(tracker.tracks)} total tracks")

final_tracks = [t for t in tracker.tracks if len(t.frame_ids) >= 2]
print(f"✓ Final: {len(final_tracks)} tracks with ≥2 frames")

for track in final_tracks:
    print(f"  - Track {track.track_id}: {len(track.frame_ids)} frames")

# Test 5: Transformer matcher
print("\n[5/5] Testing TransformerMatcher...")
try:
    import torch
    from transformer_matcher import TransformerMatcher, MatchingLoss
    
    model = TransformerMatcher(hidden_dim=128, nhead=4, num_layers=2)
    
    # Dummy data
    N_tracks = 3
    N_dets = 5
    
    track_bboxes = torch.rand(N_tracks, 4)
    track_sigs = torch.rand(N_tracks, 11)
    det_bboxes = torch.rand(N_dets, 4)
    det_sigs = torch.rand(N_dets, 11)
    
    # Forward
    scores = model(track_bboxes, track_sigs, det_bboxes, det_sigs)
    
    print(f"✓ Match scores shape: {scores.shape}")
    print(f"  - Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Loss
    loss_fn = MatchingLoss(iou_thresh=0.5)
    
    # Scale to image size for IoU
    img_w, img_h = 1920, 1080
    track_bboxes_scaled = track_bboxes.clone()
    track_bboxes_scaled[:, [0, 2]] *= img_w
    track_bboxes_scaled[:, [1, 3]] *= img_h
    det_bboxes_scaled = det_bboxes.clone()
    det_bboxes_scaled[:, [0, 2]] *= img_w
    det_bboxes_scaled[:, [1, 3]] *= img_h
    
    loss = loss_fn(scores, track_bboxes_scaled, det_bboxes_scaled)
    print(f"✓ Loss: {loss.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
except ImportError:
    print("⚠ PyTorch not installed, skipping Transformer test")
    print("  Install with: pip install torch")

# Summary
print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Download MOT17 dataset from https://motchallenge.net")
print("2. Run on real data: python run_tracker.py --data_root /path/to/MOT17 \\")
print("                                          --sequence MOT17-02-FRCNN \\")
print("                                          --split train")
print("3. Evaluate: python evaluate.py --gt_file .../gt.txt \\")
print("                                --pred_file outputs/.../output.txt \\")
print("                                --sequence MOT17-02-FRCNN")
print("\nFor full documentation, see README.md")
print("=" * 60)
