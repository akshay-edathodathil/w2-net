"""
Flow and Jacobian Feature Extraction for Object Tracking

Extracts per-bbox motion signatures from optical flow fields:
- Translation: (Δx, Δy) from mean flow
- Jacobian: spatial derivatives ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
- Reliability: variance and temporal consistency metrics

Physics-inspired: treats flow as a continuous velocity field, Jacobian
captures local deformation (expansion, rotation, shear).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import cv2


class FlowJacobianExtractor:
    """Extract motion features from optical flow fields per bounding box."""
    
    def __init__(self, flow_model: str = 'raft', device: str = 'cuda'):
        """
        Args:
            flow_model: 'raft' or 'farneback' (CPU fallback)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.flow_model_name = flow_model
        
        if flow_model == 'raft' and device == 'cuda':
            try:
                from torchvision.models.optical_flow import raft_large
                self.flow_model = raft_large(pretrained=True).to(device).eval()
                self.use_raft = True
            except:
                print("RAFT unavailable, falling back to Farneback")
                self.use_raft = False
        else:
            self.use_raft = False
    
    def compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between consecutive frames.
        
        Args:
            img1, img2: HxWx3 uint8 images
            
        Returns:
            flow: HxWx2 float32, (u, v) velocities
        """
        if self.use_raft:
            return self._compute_raft_flow(img1, img2)
        else:
            return self._compute_farneback_flow(img1, img2)
    
    def _compute_raft_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """RAFT-based flow computation."""
        # Preprocess: RGB to tensor [1, 3, H, W]
        def preprocess(img):
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            return img_t.to(self.device)
        
        with torch.no_grad():
            img1_t = preprocess(img1)
            img2_t = preprocess(img2)
            flow_predictions = self.flow_model(img1_t, img2_t)
            flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()
        
        return flow
    
    def _compute_farneback_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """CPU fallback using Farneback."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow
    
    def compute_jacobian(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute spatial Jacobian of flow field using Sobel filters.
        
        Jacobian matrix J = [[∂u/∂x, ∂u/∂y],
                             [∂v/∂x, ∂v/∂y]]
        
        Args:
            flow: HxWx2 flow field
            
        Returns:
            jacobian: HxWx2x2 Jacobian at each pixel
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        
        # Sobel derivatives (scale=1, normalized)
        du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        du_dy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        dv_dx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        
        # Stack into HxWx2x2
        jacobian = np.stack([
            np.stack([du_dx, du_dy], axis=-1),
            np.stack([dv_dx, dv_dy], axis=-1)
        ], axis=-2)
        
        return jacobian
    
    def extract_bbox_features(
        self,
        flow: np.ndarray,
        jacobian: np.ndarray,
        bbox: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract motion signature for a single bounding box.
        
        Args:
            flow: HxWx2 flow field
            jacobian: HxWx2x2 Jacobian field
            bbox: [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            signature: dict with keys:
                - delta_x, delta_y: mean translation
                - J_mean: mean Jacobian (2x2 matrix, flattened to 4 values)
                - J_trace: trace(J) = ∂u/∂x + ∂v/∂y (divergence)
                - J_det: det(J) (area change)
                - flow_var: variance of flow within bbox
                - J_var: variance of Jacobian trace
        """
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(flow.shape[1], x2), min(flow.shape[0], y2)
        
        # Extract bbox region
        flow_roi = flow[y1:y2, x1:x2]
        jac_roi = jacobian[y1:y2, x1:x2]
        
        if flow_roi.size == 0:
            return self._empty_signature()
        
        # Mean translation
        delta_x = float(np.mean(flow_roi[:, :, 0]))
        delta_y = float(np.mean(flow_roi[:, :, 1]))
        
        # Mean Jacobian (2x2)
        J_mean = np.mean(jac_roi, axis=(0, 1))  # 2x2
        J_trace = float(J_mean[0, 0] + J_mean[1, 1])  # divergence
        J_det = float(J_mean[0, 0] * J_mean[1, 1] - J_mean[0, 1] * J_mean[1, 0])
        
        # Variance measures (reliability)
        flow_var = float(np.var(flow_roi))
        
        # Jacobian trace variance (measures consistency)
        jac_trace_field = jac_roi[:, :, 0, 0] + jac_roi[:, :, 1, 1]
        J_var = float(np.var(jac_trace_field))
        
        signature = {
            'delta_x': delta_x,
            'delta_y': delta_y,
            'J_mean': J_mean.flatten().tolist(),  # [J11, J12, J21, J22]
            'J_trace': J_trace,
            'J_det': J_det,
            'flow_var': flow_var,
            'J_var': J_var
        }
        
        return signature
    
    def _empty_signature(self) -> Dict[str, float]:
        """Return zero signature for edge cases."""
        return {
            'delta_x': 0.0,
            'delta_y': 0.0,
            'J_mean': [0.0, 0.0, 0.0, 0.0],
            'J_trace': 0.0,
            'J_det': 0.0,
            'flow_var': 0.0,
            'J_var': 0.0
        }
    
    def batch_extract(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Extract features for multiple bboxes.
        
        Args:
            img1, img2: consecutive frames
            bboxes: Nx4 array of [x1, y1, x2, y2]
            
        Returns:
            flow: HxWx2
            jacobian: HxWx2x2
            signatures: list of N signature dicts
        """
        flow = self.compute_flow(img1, img2)
        jacobian = self.compute_jacobian(flow)
        
        signatures = []
        for bbox in bboxes:
            sig = self.extract_bbox_features(flow, jacobian, bbox)
            signatures.append(sig)
        
        return flow, jacobian, signatures


def signature_distance(sig1: Dict, sig2: Dict, weights: Optional[Dict] = None) -> float:
    """
    Compute distance between two motion signatures.
    
    Inspired by neuroscience: weights motion components by reliability.
    
    Args:
        sig1, sig2: motion signatures from extract_bbox_features
        weights: optional dict of weights for each component
        
    Returns:
        distance: scalar
    """
    if weights is None:
        weights = {
            'translation': 1.0,
            'divergence': 0.5,
            'det': 0.3,
            'var_penalty': 0.2
        }
    
    # Translation difference
    d_trans = np.sqrt(
        (sig1['delta_x'] - sig2['delta_x'])**2 +
        (sig1['delta_y'] - sig2['delta_y'])**2
    )
    
    # Divergence difference (∇·v)
    d_div = abs(sig1['J_trace'] - sig2['J_trace'])
    
    # Determinant difference (area change)
    d_det = abs(sig1['J_det'] - sig2['J_det'])
    
    # Variance penalty (high variance = less reliable)
    var_penalty = (sig1['flow_var'] + sig2['flow_var']) / 2
    
    distance = (
        weights['translation'] * d_trans +
        weights['divergence'] * d_div +
        weights['det'] * d_det +
        weights['var_penalty'] * np.log1p(var_penalty)
    )
    
    return distance


if __name__ == '__main__':
    # Test on dummy data
    print("Testing FlowJacobianExtractor...")
    
    # Create synthetic flow
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.roll(img1, 5, axis=1)  # shift right
    
    extractor = FlowJacobianExtractor(flow_model='farneback', device='cpu')
    
    bboxes = np.array([
        [100, 100, 200, 200],
        [300, 300, 400, 400]
    ])
    
    flow, jacobian, signatures = extractor.batch_extract(img1, img2, bboxes)
    
    print(f"Flow shape: {flow.shape}")
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Extracted {len(signatures)} signatures")
    print(f"Signature 0: Δx={signatures[0]['delta_x']:.2f}, Δy={signatures[0]['delta_y']:.2f}")
    
    dist = signature_distance(signatures[0], signatures[1])
    print(f"Distance between signatures: {dist:.3f}")
