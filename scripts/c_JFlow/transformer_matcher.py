"""
Transformer-Based Association Module (Option A)

Learns to predict match scores between tracks and detections.

Inputs:
- Track token: encoded from [bbox, velocity, motion signature]
- Detection token: encoded from [bbox, appearance, motion signature]
- Cross-attention to compute match probability

Architecture inspired by MOTR and TrackFormer but adapted for
motion signatures from optical flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MotionSignatureEncoder(nn.Module):
    """Encode motion signature into fixed-size embedding."""
    
    def __init__(self, hidden_dim: int = 256):
        """
        Args:
            hidden_dim: dimension of output embedding
        """
        super().__init__()
        
        # Motion signature has 11 components:
        # delta_x, delta_y (2)
        # J_mean (4)
        # J_trace, J_det (2)
        # flow_var, J_var (2)
        # uncertainty (1)
        
        self.mlp = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
    
    def forward(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signatures: [B, 11] motion signature features
            
        Returns:
            embeddings: [B, hidden_dim]
        """
        return self.mlp(signatures)


class BBoxEncoder(nn.Module):
    """Encode bounding box into embedding."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Bbox has 4 components: [x1, y1, x2, y2]
        # Normalize and encode
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: [B, 4] normalized bbox coordinates
            
        Returns:
            embeddings: [B, hidden_dim]
        """
        return self.mlp(bboxes)


class TransformerMatcher(nn.Module):
    """
    Transformer-based matcher for track-detection association.
    
    Similar to DETR/MOTR but focuses on motion features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: embedding dimension
            nhead: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.bbox_encoder = BBoxEncoder(hidden_dim)
        self.motion_encoder = MotionSignatureEncoder(hidden_dim)
        
        # Transformer encoder for tracks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.track_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer encoder for detections
        self.det_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention for matching
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.matcher = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head: predict match score
        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        track_bboxes: torch.Tensor,
        track_signatures: torch.Tensor,
        det_bboxes: torch.Tensor,
        det_signatures: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute match scores.
        
        Args:
            track_bboxes: [N_tracks, 4] normalized bbox coordinates
            track_signatures: [N_tracks, 11] motion signatures
            det_bboxes: [N_dets, 4] normalized bbox coordinates
            det_signatures: [N_dets, 11] motion signatures
            
        Returns:
            match_matrix: [N_tracks, N_dets] scores in [0, 1]
        """
        # Encode tracks
        track_bbox_emb = self.bbox_encoder(track_bboxes)  # [N_tracks, hidden_dim]
        track_motion_emb = self.motion_encoder(track_signatures)  # [N_tracks, hidden_dim]
        track_emb = track_bbox_emb + track_motion_emb  # residual
        
        # Encode detections
        det_bbox_emb = self.bbox_encoder(det_bboxes)  # [N_dets, hidden_dim]
        det_motion_emb = self.motion_encoder(det_signatures)  # [N_dets, hidden_dim]
        det_emb = det_bbox_emb + det_motion_emb
        
        # Self-attention on tracks and detections
        track_emb = track_emb.unsqueeze(0)  # [1, N_tracks, hidden_dim]
        det_emb = det_emb.unsqueeze(0)  # [1, N_dets, hidden_dim]
        
        track_encoded = self.track_encoder(track_emb)  # [1, N_tracks, hidden_dim]
        det_encoded = self.det_encoder(det_emb)  # [1, N_dets, hidden_dim]
        
        # Cross-attention: query = tracks, memory = detections
        matched = self.matcher(track_encoded, det_encoded)  # [1, N_tracks, hidden_dim]
        
        # Compute pairwise scores via dot product + MLP
        # For each track, compute score with each detection
        track_feat = matched.squeeze(0)  # [N_tracks, hidden_dim]
        det_feat = det_encoded.squeeze(0)  # [N_dets, hidden_dim]
        
        # Outer product: [N_tracks, hidden_dim] x [N_dets, hidden_dim]
        # -> [N_tracks, N_dets, hidden_dim]
        track_expanded = track_feat.unsqueeze(1).expand(-1, det_feat.size(0), -1)
        det_expanded = det_feat.unsqueeze(0).expand(track_feat.size(0), -1, -1)
        
        pairwise = track_expanded * det_expanded  # element-wise product
        
        # Predict match scores
        scores = self.match_head(pairwise).squeeze(-1)  # [N_tracks, N_dets]
        
        return scores


class MatchingLoss(nn.Module):
    """
    Loss for training the matcher.
    
    Uses binary cross-entropy with ground truth matches from IoU.
    """
    
    def __init__(self, iou_thresh: float = 0.5):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        pred_scores: torch.Tensor,
        track_bboxes: torch.Tensor,
        det_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred_scores: [N_tracks, N_dets] predicted match scores
            track_bboxes: [N_tracks, 4] track bboxes
            det_bboxes: [N_dets, 4] detection bboxes
            
        Returns:
            loss: scalar
        """
        # Compute ground truth IoU matrix
        ious = self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        # Binary labels: 1 if IoU > threshold
        gt_labels = (ious > self.iou_thresh).float()
        
        # BCE loss
        loss = self.bce(pred_scores, gt_labels)
        
        return loss
    
    def _compute_iou_matrix(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise IoU.
        
        Args:
            bboxes1: [N, 4]
            bboxes2: [M, 4]
            
        Returns:
            ious: [N, M]
        """
        # Expand for broadcasting
        b1 = bboxes1.unsqueeze(1)  # [N, 1, 4]
        b2 = bboxes2.unsqueeze(0)  # [1, M, 4]
        
        # Intersection
        x1_inter = torch.max(b1[:, :, 0], b2[:, :, 0])
        y1_inter = torch.max(b1[:, :, 1], b2[:, :, 1])
        x2_inter = torch.min(b1[:, :, 2], b2[:, :, 2])
        y2_inter = torch.min(b1[:, :, 3], b2[:, :, 3])
        
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * \
                     torch.clamp(y2_inter - y1_inter, min=0)
        
        # Union
        area1 = (b1[:, :, 2] - b1[:, :, 0]) * (b1[:, :, 3] - b1[:, :, 1])
        area2 = (b2[:, :, 2] - b2[:, :, 0]) * (b2[:, :, 3] - b2[:, :, 1])
        union_area = area1 + area2 - inter_area
        
        ious = inter_area / (union_area + 1e-6)
        
        return ious


def prepare_training_data(
    track_bboxes: np.ndarray,
    track_signatures: list,
    det_bboxes: np.ndarray,
    det_signatures: list,
    img_width: int,
    img_height: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy data to torch tensors for training.
    
    Args:
        track_bboxes: [N, 4] numpy array
        track_signatures: list of N dicts
        det_bboxes: [M, 4] numpy array
        det_signatures: list of M dicts
        img_width, img_height: image dimensions for normalization
        
    Returns:
        track_bboxes_t, track_sigs_t, det_bboxes_t, det_sigs_t: torch tensors
    """
    # Normalize bboxes
    track_bboxes_norm = track_bboxes.copy()
    track_bboxes_norm[:, [0, 2]] /= img_width
    track_bboxes_norm[:, [1, 3]] /= img_height
    
    det_bboxes_norm = det_bboxes.copy()
    det_bboxes_norm[:, [0, 2]] /= img_width
    det_bboxes_norm[:, [1, 3]] /= img_height
    
    # Convert signatures to arrays
    def sig_to_array(sig):
        return np.array([
            sig['delta_x'], sig['delta_y'],
            sig['J_mean'][0], sig['J_mean'][1], sig['J_mean'][2], sig['J_mean'][3],
            sig['J_trace'], sig['J_det'],
            sig['flow_var'], sig['J_var'],
            0.0  # placeholder for uncertainty
        ])
    
    track_sigs = np.stack([sig_to_array(s) for s in track_signatures])
    det_sigs = np.stack([sig_to_array(s) for s in det_signatures])
    
    # To torch
    track_bboxes_t = torch.from_numpy(track_bboxes_norm).float()
    track_sigs_t = torch.from_numpy(track_sigs).float()
    det_bboxes_t = torch.from_numpy(det_bboxes_norm).float()
    det_sigs_t = torch.from_numpy(det_sigs).float()
    
    return track_bboxes_t, track_sigs_t, det_bboxes_t, det_sigs_t


if __name__ == '__main__':
    print("Testing TransformerMatcher...")
    
    # Dummy data
    N_tracks = 5
    N_dets = 8
    img_w, img_h = 1920, 1080
    
    # Random bboxes and signatures
    track_bboxes = torch.rand(N_tracks, 4)
    track_sigs = torch.rand(N_tracks, 11)
    det_bboxes = torch.rand(N_dets, 4)
    det_sigs = torch.rand(N_dets, 11)
    
    # Model
    model = TransformerMatcher(hidden_dim=256, nhead=8, num_layers=2)
    
    # Forward pass
    scores = model(track_bboxes, track_sigs, det_bboxes, det_sigs)
    
    print(f"Match scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Loss
    loss_fn = MatchingLoss(iou_thresh=0.5)
    
    # Denormalize for IoU computation
    track_bboxes_denorm = track_bboxes.clone()
    track_bboxes_denorm[:, [0, 2]] *= img_w
    track_bboxes_denorm[:, [1, 3]] *= img_h
    
    det_bboxes_denorm = det_bboxes.clone()
    det_bboxes_denorm[:, [0, 2]] *= img_w
    det_bboxes_denorm[:, [1, 3]] *= img_h
    
    loss = loss_fn(scores, track_bboxes_denorm, det_bboxes_denorm)
    
    print(f"Loss: {loss.item():.4f}")
    
    print("\nModel summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
