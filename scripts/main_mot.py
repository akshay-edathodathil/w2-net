import torch
from torchvision.models.optical_flow import raft_small
# Assume you have a dataloader for MOT17 that yields (frame1, frame2, detections)
from src.tracker import FlowByteTracker
from src.flow_jacobian import compute_flow_signature
def run_mot_evaluation():
    device = torch.device("mps")
    flow_model = raft_small(pretrained=True).to(device).eval()
    tracker = FlowByteTracker()
    
    # Open file for MOT17 evaluation format
    with open('results/MOT17-04.txt', 'w') as f:
        
        for frame_idx, (img1, img2, frame_detections) in enumerate(mot17_loader):
            # Compute Dense Flow
            with torch.no_grad():
                flow_list = flow_model(img1.to(device), img2.to(device))
                flow_map = flow_list[-1][0] # Get final flow map
            
            # Enrich detections with flow signatures
            for det in frame_detections:
                sig, rel = compute_flow_signature(flow_map, det['bbox'])
                det['signature'] = sig
                det['reliability'] = rel
                
            # Update Tracker
            active_tracks = tracker.update(flow_map, frame_detections)
            
            # Write Output
            for track in active_tracks:
                x1, y1, x2, y2 = track['bbox']
                w, h = x2 - x1, y2 - y1
                # MOT Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                f.write(f"{frame_idx+1},{track['id']},{x1},{y1},{w},{h},1,-1,-1,-1\n")

if __name__ == "__main__":
    print("Starting Flow-Jacobian MOT17 Tracking Pipeline...")
    # run_mot_evaluation()