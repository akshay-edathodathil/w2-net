import torch
import numpy as np

def compute_flow_signature(flow_map, bbox):
    """
    Computes translation, Jacobian, and variances for a given bounding box.
    flow_map: torch.Tensor of shape (2, H, W) on 'mps'
    bbox: list or array [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Crop flow to bbox (u is x-displacement, v is y-displacement)
    u_crop = flow_map[0, y1:y2, x1:x2]
    v_crop = flow_map[1, y1:y2, x1:x2]
    
    # 1. Translation (\Delta x, \Delta y)
    dx = torch.mean(u_crop).item()
    dy = torch.mean(v_crop).item()
    var_flow = torch.var(u_crop).item() + torch.var(v_crop).item()
    
    # 2. Jacobian J = [[du/dx, du/dy], [dv/dx, dv/dy]]
    # torch.gradient returns gradients along dimensions (y, then x)
    du_dy, du_dx = torch.gradient(u_crop)
    dv_dy, dv_dx = torch.gradient(v_crop)
    
    J_xx = torch.mean(du_dx).item()
    J_xy = torch.mean(du_dy).item()
    J_yx = torch.mean(dv_dx).item()
    J_yy = torch.mean(dv_dy).item()
    
    var_J = torch.var(du_dx).item() + torch.var(dv_dy).item()
    
    signature = np.array([dx, dy, J_xx, J_xy, J_yx, J_yy, var_flow, var_J])
    
    # Reliability score (inversely proportional to variance)
    # High variance means chaotic flow (e.g., occlusions or legs crossing)
    reliability = 1.0 / (1.0 + var_flow + var_J) 
    
    return signature, reliability

def predict_bbox(bbox, signature):
    """
    Applies the affine warp derived from \bar{J} + translation.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    xc, yc = x1 + w/2, y1 + h/2
    
    dx, dy, J_xx, J_xy, J_yx, J_yy, _, _ = signature
    
    # Predict new center using translation
    xc_pred = xc + dx
    yc_pred = yc + dy
    
    # Predict new scale using the diagonal of the Jacobian
    w_pred = w * (1 + J_xx)
    h_pred = h * (1 + J_yy)
    
    # Reconstruct bbox
    x1_pred = xc_pred - w_pred/2
    y1_pred = yc_pred - h_pred/2
    x2_pred = xc_pred + w_pred/2
    y2_pred = yc_pred + h_pred/2
    
    return [x1_pred, y1_pred, x2_pred, y2_pred]