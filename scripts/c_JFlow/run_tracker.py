"""
Main Script to Run Flow/Jacobian Tracker on MOT17

Usage:
    python run_tracker.py --data_root /path/to/MOT17 --sequence MOT17-02-FRCNN --split train
"""

import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from mot_utils import MOTDataset, MOTWriter
from tracker import FlowTracker
from flow_jacobian_extractor import FlowJacobianExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Run Flow/Jacobian Tracker')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to MOT17 root directory')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Sequence name (e.g., MOT17-02-FRCNN)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Data split')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for tracking results')
    
    # Tracker parameters
    parser.add_argument('--flow_model', type=str, default='farneback',
                        choices=['raft', 'farneback'],
                        help='Optical flow model')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['mps','cpu', 'cuda'],
                        help='Device for computation')
    parser.add_argument('--use_jacobian', action='store_true', default=True,
                        help='Use full Jacobian for motion prediction')
    
    # Association parameters
    parser.add_argument('--high_conf_thresh', type=float, default=0.35,
                        help='High confidence threshold')
    parser.add_argument('--low_conf_thresh', type=float, default=0.05,
                        help='Low confidence threshold')
    parser.add_argument('--match_thresh', type=float, default=0.8,
                        help='Maximum cost for valid match')
    parser.add_argument('--iou_weight', type=float, default=0.85,
                        help='Weight for IoU cost (alpha)')
    parser.add_argument('--sig_weight', type=float, default=0.15,
                        help='Weight for signature distance (beta)')
    
    # Track management
    parser.add_argument('--max_age', type=int, default=60,
                        help='Max frames to keep lost track')
    parser.add_argument('--min_hits', type=int, default=1,
                        help='Min detections before outputting track')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization (requires cv2)')
    parser.add_argument('--vis_every', type=int, default=10,
                        help='Visualize every N frames')
    
    return parser.parse_args()


def visualize_tracking(
    img: np.ndarray,
    tracks: list,
    frame_id: int,
    output_path: Path
):
    """Save tracking visualization."""
    import cv2
    
    img_vis = img.copy()
    
    # Draw tracks
    for track in tracks:
        if frame_id not in track.frame_ids:
            continue
        
        idx = track.frame_ids.index(frame_id)
        bbox = track.bboxes[idx]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Color by track ID
        color = tuple(np.random.RandomState(track.track_id).randint(0, 255, 3).tolist())
        
        # Draw bbox
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        cv2.putText(img_vis, f'ID: {track.track_id}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    
    # Convert RGB to BGR for cv2
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_vis)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Flow/Jacobian Multi-Object Tracker")
    print("=" * 60)
    print(f"Sequence: {args.sequence}")
    print(f"Flow model: {args.flow_model}")
    print(f"Device: {args.device}")
    print(f"Use Jacobian: {args.use_jacobian}")
    print(f"IoU weight: {args.iou_weight}, Signature weight: {args.sig_weight}")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_root}...")
    dataset = MOTDataset(args.data_root, args.sequence, args.split)
    print(f"Loaded {dataset.num_frames} frames")
    
    # Initialize tracker
    tracker = FlowTracker(
        flow_model=args.flow_model,
        device=args.device,
        use_jacobian=args.use_jacobian,
        high_conf_thresh=args.high_conf_thresh,
        low_conf_thresh=args.low_conf_thresh,
        match_thresh=args.match_thresh,
        iou_weight=args.iou_weight,
        sig_weight=args.sig_weight,
        max_age=args.max_age,
        min_hits=args.min_hits
    )
    
    # Initialize writer
    output_dir = Path(args.output_dir) / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.sequence}.txt'
    writer = MOTWriter(output_file)
    
    if args.visualize:
        vis_dir = output_dir / 'visualization'
        vis_dir.mkdir(exist_ok=True)
    
    # Process frames
    print("\nProcessing frames...")
    start_time = time.time()
    
    all_tracks = {}  # Store all tracks for final output
    
    for frame_id in tqdm(range(1, dataset.num_frames + 1)):
        # Load image and detections
        img = dataset.get_image(frame_id)
        detections = dataset.get_detections(frame_id)
        
        # Update tracker
        active_tracks = tracker.update(img, detections)
        
        # Store tracks
        for track in active_tracks:
            if track.track_id not in all_tracks:
                all_tracks[track.track_id] = track
        
        # Visualize
        if args.visualize and frame_id % args.vis_every == 0:
            vis_path = vis_dir / f'{frame_id:06d}.jpg'
            visualize_tracking(img, list(all_tracks.values()), frame_id, vis_path)
    
    # Write results
    print("\nWriting tracking results...")
    for track_id, track in all_tracks.items():
        for bbox, frame_id, conf in zip(track.bboxes, track.frame_ids, track.confidences):
            writer.add_track(frame_id, track_id, bbox, conf)
    
    writer.write()
    
    elapsed = time.time() - start_time
    fps = dataset.num_frames / elapsed
    
    print(f"\nTracking completed!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Total tracks: {len(all_tracks)}")
    print(f"Output saved to: {output_file}")
    
    # Print statistics
    avg_track_length = np.mean([len(t.frame_ids) for t in all_tracks.values()])
    max_track_length = max([len(t.frame_ids) for t in all_tracks.values()])
    print(f"\nTrack statistics:")
    print(f"  Average length: {avg_track_length:.1f} frames")
    print(f"  Maximum length: {max_track_length} frames")


if __name__ == '__main__':
    main()
