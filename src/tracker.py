from .association import match_detections
from .flow_jacobian import predict_bbox, compute_flow_signature

class FlowByteTracker:
    def __init__(self, track_buffer=30):
        self.tracks = [] # Active tracks
        self.lost_tracks = []
        self.next_id = 1
        self.track_buffer = track_buffer
        
    def update(self, flow_map, detections):
        """
        detections: list of dicts {'bbox': [x1,y1,x2,y2], 'score': float, 'signature': array}
        """
        # 1. Predict next locations for all active and lost tracks
        for track in self.tracks + self.lost_tracks:
            track['pred_bbox'] = predict_bbox(track['bbox'], track['signature'])
            
        # 2. Split detections by confidence (ByteTrack logic)
        high_dets = [d for d in detections if d['score'] >= 0.5]
        low_dets = [d for d in detections if d['score'] < 0.5]
        
        # 3. First Stage: Match high conf detections to active tracks
        matches_a, un_tracks_a, un_dets_a = match_detections(self.tracks, high_dets)
        
        # Update matched tracks
        for trk_idx, det_idx in matches_a:
            self.tracks[trk_idx]['bbox'] = high_dets[det_idx]['bbox']
            self.tracks[trk_idx]['signature'] = high_dets[det_idx]['signature']
            # update reliability...
            
        # 4. Second Stage: Match low conf detections to unmatched tracks from Stage 1
        leftover_tracks = [self.tracks[i] for i in un_tracks_a]
        matches_b, un_tracks_b, un_dets_b = match_detections(leftover_tracks, low_dets)
        
        # ... logic to update second-stage matches, move un_tracks_b to lost_tracks, 
        # and initialize new tracks from un_dets_a ...
        
        return self.tracks