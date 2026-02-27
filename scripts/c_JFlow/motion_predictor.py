"""
Jacobian-Based Motion Prediction for Bounding Boxes

Predicts bbox at t+1 using affine transformation derived from:
- Translation: (Δx, Δy)
- Local Jacobian: J = [[∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y]]

The prediction models the bbox as a deformable patch undergoing
local linear transformation (similar to first-order optic flow expansion).
"""

import numpy as np
from typing import Dict, Tuple, Optional


class JacobianMotionPredictor:
    """Predict next bounding box using Jacobian-based affine warp."""
    
    def __init__(self, use_full_jacobian: bool = True, damping: float = 0.5):
        """
        Args:
            use_full_jacobian: If True, apply full affine warp. If False, only translate.
            damping: Factor to dampen Jacobian effects (0-1), prevents over-warping
        """
        self.use_full_jacobian = use_full_jacobian
        self.damping = damping
    
    def predict_bbox(
        self,
        bbox: np.ndarray,
        signature: Dict[str, float],
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Predict next bounding box position using motion signature.
        
        Args:
            bbox: [x1, y1, x2, y2] current bbox
            signature: motion signature with delta_x, delta_y, J_mean
            dt: time step (usually 1.0 for frame-to-frame)
            
        Returns:
            bbox_pred: [x1, y1, x2, y2] predicted bbox
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        # Extract motion parameters
        dx = signature['delta_x'] * dt
        dy = signature['delta_y'] * dt
        
        if not self.use_full_jacobian:
            # Simple translation
            cx_new = cx + dx
            cy_new = cy + dy
            w_new, h_new = w, h
        else:
            # Affine warp using Jacobian
            J = np.array(signature['J_mean']).reshape(2, 2)
            J_damped = np.eye(2) + self.damping * J * dt
            
            # Transform center point
            delta_c = np.array([dx, dy])
            c_new = np.array([cx, cy]) + delta_c
            
            # Transform width and height vectors
            # Bbox corners relative to center
            corners_rel = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
            ])
            
            # Apply Jacobian transformation
            corners_warped = corners_rel @ J_damped.T
            
            # Compute new axis-aligned bbox from warped corners
            x_coords = corners_warped[:, 0] + c_new[0]
            y_coords = corners_warped[:, 1] + c_new[1]
            
            x1_new = np.min(x_coords)
            y1_new = np.min(y_coords)
            x2_new = np.max(x_coords)
            y2_new = np.max(y_coords)
            
            return np.array([x1_new, y1_new, x2_new, y2_new])
        
        # Simple case: translation only
        x1_new = cx_new - w_new / 2
        y1_new = cy_new - h_new / 2
        x2_new = cx_new + w_new / 2
        y2_new = cy_new + h_new / 2
        
        return np.array([x1_new, y1_new, x2_new, y2_new])
    
    def predict_bbox_velocity(
        self,
        bbox: np.ndarray,
        velocity: Tuple[float, float],
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Simple constant-velocity prediction (fallback when no Jacobian).
        
        Args:
            bbox: [x1, y1, x2, y2]
            velocity: (vx, vy) in pixels/frame
            dt: time step
            
        Returns:
            bbox_pred: [x1, y1, x2, y2]
        """
        vx, vy = velocity
        return bbox + np.array([vx * dt, vy * dt, vx * dt, vy * dt])
    
    def compute_motion_uncertainty(
        self,
        signature: Dict[str, float]
    ) -> float:
        """
        Compute uncertainty/reliability of motion prediction.
        
        Based on flow variance and Jacobian consistency.
        Lower values = more reliable prediction.
        
        Args:
            signature: motion signature
            
        Returns:
            uncertainty: scalar in [0, 1]
        """
        flow_var = signature['flow_var']
        J_var = signature['J_var']
        
        # Normalize variances (heuristic thresholds)
        flow_uncertainty = np.clip(flow_var / 100.0, 0, 1)
        J_uncertainty = np.clip(J_var / 0.1, 0, 1)
        
        # Combined uncertainty
        uncertainty = 0.7 * flow_uncertainty + 0.3 * J_uncertainty
        
        return uncertainty


class KalmanMotionPredictor:
    """
    Kalman filter for comparison/ablation.
    
    State: [x, y, w, h, vx, vy] (center + size + velocity)
    """
    
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        """
        Args:
            process_noise: Q covariance scale
            measurement_noise: R covariance scale
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1],  # y = y + vy
            [0, 0, 1, 0, 0, 0],  # w = w
            [0, 0, 0, 1, 0, 0],  # h = h
            [0, 0, 0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1],  # vy = vy
        ])
        
        # Measurement matrix (observe position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])
        
        # Covariances
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(4) * measurement_noise
    
    def bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h, 0, 0]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h, 0, 0])
    
    def state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h, vx, vy] to [x1, y1, x2, y2]."""
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self, state: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman prediction step.
        
        Args:
            state: current state [6]
            P: current covariance [6x6]
            
        Returns:
            state_pred: predicted state
            P_pred: predicted covariance
        """
        state_pred = self.F @ state
        P_pred = self.F @ P @ self.F.T + self.Q
        return state_pred, P_pred
    
    def update(
        self,
        state_pred: np.ndarray,
        P_pred: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step.
        
        Args:
            state_pred: predicted state [6]
            P_pred: predicted covariance [6x6]
            measurement: observed bbox converted to [cx, cy, w, h]
            
        Returns:
            state: updated state
            P: updated covariance
        """
        # Innovation
        z = measurement  # [cx, cy, w, h]
        z_pred = self.H @ state_pred
        y = z - z_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update
        state = state_pred + K @ y
        P = (np.eye(6) - K @ self.H) @ P_pred
        
        return state, P


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Intersection over Union for two bboxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2]
        
    Returns:
        iou: scalar in [0, 1]
    """
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


if __name__ == '__main__':
    print("Testing JacobianMotionPredictor...")
    
    # Test Jacobian predictor
    predictor = JacobianMotionPredictor(use_full_jacobian=True, damping=0.5)
    
    bbox = np.array([100, 100, 200, 200])
    signature = {
        'delta_x': 5.0,
        'delta_y': 2.0,
        'J_mean': [0.1, 0.0, 0.0, 0.1],  # slight expansion
        'J_trace': 0.2,
        'J_det': 0.01,
        'flow_var': 10.0,
        'J_var': 0.01
    }
    
    bbox_pred = predictor.predict_bbox(bbox, signature)
    print(f"Original bbox: {bbox}")
    print(f"Predicted bbox: {bbox_pred}")
    print(f"IoU: {iou(bbox, bbox_pred):.3f}")
    
    uncertainty = predictor.compute_motion_uncertainty(signature)
    print(f"Motion uncertainty: {uncertainty:.3f}")
    
    # Test Kalman filter
    print("\nTesting KalmanMotionPredictor...")
    kalman = KalmanMotionPredictor()
    
    state = kalman.bbox_to_state(bbox)
    P = np.eye(6) * 10
    
    state_pred, P_pred = kalman.predict(state, P)
    bbox_pred_kalman = kalman.state_to_bbox(state_pred)
    print(f"Kalman predicted bbox: {bbox_pred_kalman}")
