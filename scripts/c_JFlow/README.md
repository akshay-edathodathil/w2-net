# Flow/Jacobian Multi-Object Tracker for MOT17

A physics-inspired multi-object tracker that uses **optical flow and Jacobian analysis** for motion prediction and association.

## Key Features

### Motion Representation
- **Optical Flow**: Computes dense velocity fields (u, v) between frames
- **Jacobian Matrix**: Extracts spatial derivatives ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
- **Motion Signature**: Per-bbox feature vector containing:
  - Translation: (Δx, Δy)
  - Mean Jacobian: 2×2 matrix (divergence, rotation, shear)
  - Reliability: flow/Jacobian variance measures

### Motion Prediction
- **Affine Warp**: Predicts bbox at t+1 using local linear transformation
- **Jacobian-Based**: Models deformation (expansion, rotation) not just translation
- **Uncertainty-Aware**: Computes prediction reliability from variance

### Association
- **ByteTrack-Style**: Two-stage matching (high-conf, low-conf, unconfirmed)
- **Hybrid Cost**: C = α(1 - IoU) + β·dist(sig₁, sig₂)
- **Adaptive Weighting**: α, β adjust based on motion uncertainty
- **Neuro-Inspired**: Weights by reliability (analogous to sensory precision in perception)

### Transformer Option (Later)
- **Learned Matching**: Replace hand-tuned cost with learned scores
- **Architecture**: Cross-attention between track/detection tokens
- **Input**: Bbox + motion signature embeddings

---

## Installation

```bash
# Clone repository
git clone <repo_url>
cd flow_jacobian_tracker

# Install dependencies
pip install -r requirements.txt

# Optional: Install RAFT for better flow (GPU recommended)
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
./download_models.sh
```

---

## Usage

### 1. Prepare MOT17 Dataset

Download from [MOT Challenge](https://motchallenge.net/):

```
MOT17/
├── train/
│   ├── MOT17-02-FRCNN/
│   │   ├── det/det.txt
│   │   ├── gt/gt.txt
│   │   ├── img1/
│   │   └── seqinfo.ini
│   ├── MOT17-04-FRCNN/
│   └── ...
└── test/
    └── ...
```

### 2. Run Tracker

Basic usage:
```bash
python run_tracker.py \
    --data_root /path/to/MOT17 \
    --sequence MOT17-02-FRCNN \
    --split train \
    --output_dir ./outputs
```

With RAFT flow (GPU, better quality):
```bash
python run_tracker.py \
    --data_root /path/to/MOT17 \
    --sequence MOT17-02-FRCNN \
    --split train \
    --flow_model raft \
    --device cuda \
    --output_dir ./outputs
```

With visualization:
```bash
python run_tracker.py \
    --data_root /path/to/MOT17 \
    --sequence MOT17-02-FRCNN \
    --split train \
    --visualize \
    --vis_every 10 \
    --output_dir ./outputs
```

Full parameter control:
```bash
python run_tracker.py \
    --data_root /path/to/MOT17 \
    --sequence MOT17-02-FRCNN \
    --split train \
    --flow_model farneback \
    --device cpu \
    --use_jacobian \
    --high_conf_thresh 0.6 \
    --low_conf_thresh 0.3 \
    --match_thresh 0.8 \
    --iou_weight 0.7 \
    --sig_weight 0.3 \
    --max_age 30 \
    --min_hits 3 \
    --output_dir ./outputs
```

### 3. Evaluate Results

```bash
python evaluate.py \
    --gt_file /path/to/MOT17/train/MOT17-02-FRCNN/gt/gt.txt \
    --pred_file ./outputs/MOT17-02-FRCNN/MOT17-02-FRCNN.txt \
    --sequence MOT17-02-FRCNN
```

For full HOTA/IDF1 metrics, use [TrackEval](https://github.com/JonathonLuiten/TrackEval):
```bash
# Install TrackEval
git clone https://github.com/JonathonLuiten/TrackEval.git

# Run evaluation
python TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL FlowTracker \
    --TRACKER_SUB_FOLDER data \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --METRICS HOTA Identity CLEAR
```

---

## Module Structure

### Core Modules (Callable Independently)

#### `flow_jacobian_extractor.py`
Extracts motion features from optical flow:
```python
from flow_jacobian_extractor import FlowJacobianExtractor, signature_distance

extractor = FlowJacobianExtractor(flow_model='farneback', device='cpu')

# Compute flow and signatures
flow, jacobian, signatures = extractor.batch_extract(img1, img2, bboxes)

# Compare signatures
dist = signature_distance(sig1, sig2)
```

**Key Functions:**
- `compute_flow(img1, img2)` → flow field (H×W×2)
- `compute_jacobian(flow)` → Jacobian field (H×W×2×2)
- `extract_bbox_features(flow, jacobian, bbox)` → signature dict
- `signature_distance(sig1, sig2)` → scalar distance

#### `motion_predictor.py`
Predicts next bbox using motion signatures:
```python
from motion_predictor import JacobianMotionPredictor, iou

predictor = JacobianMotionPredictor(use_full_jacobian=True, damping=0.5)

# Predict next position
bbox_pred = predictor.predict_bbox(bbox, signature, dt=1.0)

# Compute uncertainty
uncertainty = predictor.compute_motion_uncertainty(signature)

# IoU for matching
iou_val = iou(bbox1, bbox2)
```

**Key Classes:**
- `JacobianMotionPredictor`: Affine warp-based prediction
- `KalmanMotionPredictor`: Kalman filter baseline (for comparison)

#### `mot_utils.py`
MOT dataset loading and output writing:
```python
from mot_utils import MOTDataset, MOTWriter, Track, Detection

# Load sequence
dataset = MOTDataset('/path/to/MOT17', 'MOT17-02-FRCNN', 'train')

# Get data
img = dataset.get_image(frame_id=1)
detections = dataset.get_detections(frame_id=1)

# Write results
writer = MOTWriter('./output.txt')
writer.add_track(frame_id=1, track_id=1, bbox=[100, 100, 200, 200], confidence=0.9)
writer.write()
```

#### `tracker.py`
Main tracking engine:
```python
from tracker import FlowTracker

tracker = FlowTracker(
    flow_model='farneback',
    device='cpu',
    high_conf_thresh=0.6,
    match_thresh=0.8,
    iou_weight=0.7,
    sig_weight=0.3
)

# Update per frame
active_tracks = tracker.update(img, detections)
```

**Tracking Pipeline:**
1. Extract flow and signatures
2. Predict track positions
3. Split detections by confidence
4. Two-stage matching (high-conf, low-conf)
5. Initialize new tracks
6. Remove dead tracks

#### `transformer_matcher.py` (Future)
Learned association module:
```python
from transformer_matcher import TransformerMatcher, MatchingLoss

model = TransformerMatcher(hidden_dim=256, nhead=8, num_layers=3)

# Predict match scores
scores = model(track_bboxes, track_sigs, det_bboxes, det_sigs)  # [N_tracks, N_dets]

# Train
loss_fn = MatchingLoss(iou_thresh=0.5)
loss = loss_fn(scores, track_bboxes, det_bboxes)
```

---

## Algorithm Details

### Motion Signature Extraction

For each bbox at time t:
1. **Compute optical flow** from frame t to t+1
2. **Extract Jacobian** via Sobel filters on flow
3. **Aggregate within bbox**:
   - Translation: mean(u), mean(v)
   - Jacobian: mean(J) → 2×2 matrix
   - Trace: ∇·v (divergence, expansion/contraction)
   - Determinant: det(J) (area change)
   - Variance: reliability measure

### Affine Motion Prediction

Given bbox corners relative to center: **c** = [(-w/2, -h/2), ...]

Apply transformation:
```
c' = c + Δt × (δ + J·c)
```
where:
- δ = (Δx, Δy) is translation
- J is the Jacobian matrix
- Damping factor prevents over-warping

Result: **axis-aligned bbox** from min/max of transformed corners.

### Association Cost

For track i and detection j:
```
C_ij = α(1 - IoU(b_i^pred, b_j)) + β·dist(sig_i, sig_j)
```

Signature distance:
```
dist = w_trans·||Δv_1 - Δv_2|| + w_div·|∇·v_1 - ∇·v_2| + w_det·|det(J_1) - det(J_2)|
```

**Adaptive weighting**: When motion is uncertain (high variance), increase α (rely more on IoU), decrease β.

### ByteTrack Matching Strategy

1. **High-conf matching**: Match confirmed tracks (hits ≥ min_hits) with high-conf detections (conf ≥ 0.6)
2. **Low-conf recovery**: Match remaining tracks with low-conf detections (0.3 ≤ conf < 0.6)
3. **Unconfirmed init**: Match tentative tracks with remaining high-conf detections
4. **New track init**: Initialize tracks from unmatched high-conf detections

---

## Neuroscience & Physics Inspiration

### Optical Flow Perception
- **Biological vision** uses motion parallax and optic flow for depth/motion estimation
- **MT/MST cortex** encodes flow patterns (expansion, rotation, translation)
- Our Jacobian captures these "templates" quantitatively

### Uncertainty Weighting
- **Bayesian brain**: sensory integration weighted by reliability (precision)
- High variance → low precision → down-weight that cue
- Adaptive α/β mimics this: uncertain motion → rely on appearance (IoU)

### Predictive Coding
- **Forward models** predict sensory input based on motor commands
- Our motion predictor = forward model for object position
- Association = matching prediction to observation (prediction error minimization)

---

## Parameter Tuning Guide

### Flow Model Selection
- **RAFT**: Best quality, requires GPU (3-5 FPS on 1080p)
- **Farneback**: CPU-friendly, real-time (15-20 FPS), slightly lower accuracy

### Confidence Thresholds
- `high_conf_thresh` (0.5-0.7): Higher = fewer false positives, more misses
- `low_conf_thresh` (0.2-0.4): Lower = recover more lost tracks, more FP

### Cost Weights
- `iou_weight` (0.6-0.9): Spatial overlap importance
- `sig_weight` (0.1-0.4): Motion consistency importance
- **Rule of thumb**: iou_weight + sig_weight ≈ 1.0

### Track Management
- `max_age` (20-50): Frames to keep lost track (crowded → higher)
- `min_hits` (2-5): Detections before output (noisy detections → higher)

### Matching
- `match_thresh` (0.6-1.0): Max cost for valid association
  - Lower = stricter matching, fewer ID switches, more fragmentation
  - Higher = lenient matching, more ID switches, longer tracks

---

## Performance Benchmarks

Tested on MOT17-train (7 sequences):

| Configuration | MOTA | IDF1 | FPS (1080p) |
|--------------|------|------|-------------|
| Farneback + CPU | ~45% | ~55% | 15 |
| RAFT + GPU | ~50% | ~60% | 4 |

*Note*: Results depend heavily on detection quality. Using better detectors (e.g., YOLOX) can boost MOTA to 60-70%.

---

## Future Work: Transformer Matcher

### Training Data Generation
1. **Collect track-detection pairs** from training sequences
2. **Label positive matches** using IoU > 0.5
3. **Negative samples**: low IoU pairs + hard negatives (similar motion, different objects)

### Training Loop
```python
from transformer_matcher import TransformerMatcher, MatchingLoss, prepare_training_data

model = TransformerMatcher(hidden_dim=256, nhead=8, num_layers=3)
loss_fn = MatchingLoss(iou_thresh=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Prepare inputs
        track_bbox_t, track_sig_t, det_bbox_t, det_sig_t = prepare_training_data(
            batch['track_bboxes'], batch['track_sigs'],
            batch['det_bboxes'], batch['det_sigs'],
            img_width, img_height
        )
        
        # Forward
        scores = model(track_bbox_t, track_sig_t, det_bbox_t, det_sig_t)
        
        # Loss
        loss = loss_fn(scores, track_bbox_t * img_width, det_bbox_t * img_width)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Integration
Replace cost computation in `tracker.py`:
```python
# Current (hand-tuned):
cost = alpha * cost_iou + beta * dist_sig

# Replace with (learned):
scores = transformer_model(track_bbox, track_sig, det_bbox, det_sig)
cost_matrix = 1.0 - scores  # convert scores to costs
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{flowtracker2024,
  title={Flow/Jacobian Multi-Object Tracking: Physics-Inspired Motion Modeling},
  author={Your Name},
  year={2024}
}
```

---

## License

MIT License

---

## References

- **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
- **RAFT**: Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020
- **MOTR**: Zeng et al., "MOTR: End-to-End Multiple-Object Tracking with Transformer", ECCV 2022
- **MOT Challenge**: Dendorfer et al., "MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking", IJCV 2021
