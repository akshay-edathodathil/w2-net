import os
import glob
import argparse

import cv2
import numpy as np
import torch

def read_mot_gt(gt_path):
    # MOT gt.txt format: frame, id, x, y, w, h, conf, class, visibility
    # frame is 1-indexed
    data = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
            bbox = [x, y, x + w, y + h]
            data.setdefault(frame, {})[tid] = bbox
    return data

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return inter / denom

def warp_bbox_with_flow(bbox, flow, H, W, center_crop=0.6):
    # Keep original bbox for output
    x1o, y1o, x2o, y2o = bbox

    # Crop region for robust motion estimate
    cx, cy = (x1o + x2o) / 2.0, (y1o + y2o) / 2.0
    hw = (x2o - x1o) / 2.0 * center_crop
    hh = (y2o - y1o) / 2.0 * center_crop

    x1c, y1c, x2c, y2c = cx - hw, cy - hh, cx + hw, cy + hh

    x1i = int(max(0, min(W - 1, round(x1c))))
    x2i = int(max(0, min(W,     round(x2c))))
    y1i = int(max(0, min(H - 1, round(y1c))))
    y2i = int(max(0, min(H,     round(y2c))))

    if x2i <= x1i + 1 or y2i <= y1i + 1:
        return [x1o, y1o, x2o, y2o], 0.0

    roi = flow[y1i:y2i, x1i:x2i]
    dx = float(np.median(roi[..., 0]))
    dy = float(np.median(roi[..., 1]))

    mag = np.sqrt(roi[..., 0] ** 2 + roi[..., 1] ** 2)
    flow_confidence = float(np.median(mag))

    # Apply dx,dy to the ORIGINAL bbox
    return [x1o + dx, y1o + dy, x2o + dx, y2o + dy], flow_confidence

def compute_flow_gmflow_ptlflow(model, img1_bgr, img2_bgr, device):
    import ptlflow
    from ptlflow.utils.io_adapter import IOAdapter

    model.eval()

    images = [img1_bgr, img2_bgr]
    io_adapter = IOAdapter(model, images[0].shape[:2])

    inputs = io_adapter.prepare_inputs(images)
    # Move tensors to device
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device).float()

    with torch.no_grad():
        preds = model(inputs)

    flows = preds["flows"]  # BNCHW, expect (1,1,2,H,W)
    flow = flows[0, 0].permute(1, 2, 0).detach().cpu().numpy()  # HxWx2
    return flow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to MOT17-XX-YY folder (contains img1/ gt/ det/)")
    ap.add_argument("--max_pairs", type=int, default=200, help="How many frame pairs to process")
    ap.add_argument("--device", default="cpu", help="cpu or mps (Mac) or cuda (GPU)")
    # ap.add_argument("--ckpt", default="gmflow-things-5a18a9e8.ckpt", help="PTLFlow GMFlow checkpoint name")
    ap.add_argument("--ckpt", default="things", help="PTLFlow checkpoint preset: chairs|things|sintel|kitti")
    ap.add_argument("--out_dir", default="results",
                    help="Directory to save CSV results into")
    ap.add_argument("--out_csv", default="auto",
                    help="CSV filename, or 'auto' to use <sequence>.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_csv == "auto":
        seq_name = os.path.basename(os.path.normpath(args.seq))
        args.out_csv = os.path.join(args.out_dir, seq_name + ".csv")
    else:
        # If user passes a filename, save it under out_dir unless they gave a path
        if os.path.dirname(args.out_csv) == "":
            args.out_csv = os.path.join(args.out_dir, args.out_csv)
        else:
            os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    img_dir = os.path.join(args.seq, "img1")
    gt_path = os.path.join(args.seq, "gt", "gt.txt")

    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if len(frames) < 2:
        raise RuntimeError("No frames found in img1/")

    gt = read_mot_gt(gt_path)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    import ptlflow
    model = ptlflow.get_model("gmflow", ckpt_path=args.ckpt)
    model = model.to(device).float()
    # ious = []
    ious_flow = []
    ious_identity = []
    ious_gated = []
    pairs = min(args.max_pairs, len(frames) - 1)

    for i in range(pairs):
        f1 = i + 1
        f2 = i + 2

        img1 = cv2.imread(frames[i])
        img2 = cv2.imread(frames[i + 1])
        if img1 is None or img2 is None:
            continue

        H, W = img1.shape[:2]
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        flow = compute_flow_gmflow_ptlflow(model, img1_rgb, img2_rgb, device)
        # flow = compute_flow_gmflow_ptlflow(model, img1, img2, device)

        if f1 not in gt or f2 not in gt:
            continue

        # Compare per-track bbox warps
        for tid, bbox_t in gt[f1].items():
            if tid not in gt[f2]:
                continue
            bbox_t1_gt = gt[f2][tid]
            bbox_t1_flow, flow_conf = warp_bbox_with_flow(bbox_t, flow, H, W)

            iou_flow_raw = bbox_iou(bbox_t1_flow, bbox_t1_gt)      # always the flow warp
            iou_identity = bbox_iou(bbox_t, bbox_t1_gt)             # always the no-motion prior

            # Gated strategy (what your tracker would actually do)
            if flow_conf > 1.5:
                bbox_t1_pred = bbox_t1_flow
            else:
                bbox_t1_pred = bbox_t
            iou_gated = bbox_iou(bbox_t1_pred, bbox_t1_gt)

            ious_flow.append(iou_flow_raw)
            ious_identity.append(iou_identity)
            ious_gated.append(iou_gated)

        if (i + 1) % 25 == 0:
            print("Processed pairs:", (i + 1))
            print("Flow IoU mean so far:", float(np.mean(ious_flow)) if len(ious_flow) > 0 else 0.0)
            print("Identity IoU mean so far:", float(np.mean(ious_identity)) if len(ious_identity) > 0 else 0.0)
            print("Gated IoU mean so far:", float(np.mean(ious_gated)) if len(ious_gated) > 0 else 0.0)
    if len(ious_flow) == 0:
        print("No IoUs computed (maybe no overlapping tracks in chosen window).")
        return

    # ious = np.array(ious, dtype=np.float32)
    ious_flow = np.array(ious_flow, dtype=np.float32)
    ious_identity = np.array(ious_identity, dtype=np.float32)
    ious_gated = np.array(ious_gated, dtype=np.float32)
    print("Done.")
    # print("IoU mean:", float(ious.mean()))
    # print("IoU median:", float(np.median(ious)))
    # print("IoU > 0.5:", float((ious > 0.5).mean()))
    print("Flow IoU mean:", float(ious_flow.mean()) if len(ious_flow) > 0 else 0.0)
    print("Identity IoU mean:", float(ious_identity.mean()) if len(ious_identity) > 0 else 0.0)
    print("Gated IoU mean:", float(ious_gated.mean()) if len(ious_gated) > 0 else 0.0)
    print(f"{'Metric':<25} {'Flow':>8} {'Identity':>10} {'Gated':>10}")
    print(f"{'IoU mean':<25} {ious_flow.mean():>8.3f} {ious_identity.mean():>10.3f} {ious_gated.mean():>10.3f}")
    print(f"{'IoU median':<25} {np.median(ious_flow):>8.3f} {np.median(ious_identity):>10.3f} {np.median(ious_gated):>10.3f}")
    print(f"{'IoU > 0.5 (%)':<25} {(ious_flow>0.5).mean()*100:>8.1f} {(ious_identity>0.5).mean()*100:>10.1f} {(ious_gated>0.5).mean()*100:>10.1f}")
    # print(f"{'Flow wins (%)':<25} {(ious_flow > ious_identity).mean()*100:>8.1f}")
    flow_wins_pct = float((ious_flow > ious_identity).mean() * 100.0)
    print(f"{'Flow wins (%)':<25} {flow_wins_pct:>8.1f}")
    import csv
    from datetime import datetime

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sequence": os.path.basename(args.seq.rstrip("/")),
        "ckpt": args.ckpt,
        "device": args.device,
        "max_pairs": pairs,

        "flow_iou_mean": float(ious_flow.mean()),
        "flow_iou_median": float(np.median(ious_flow)),
        "flow_iou_gt_0p5": float((ious_flow > 0.5).mean()),

        "identity_iou_mean": float(ious_identity.mean()),
        "identity_iou_median": float(np.median(ious_identity)),
        "identity_iou_gt_0p5": float((ious_identity > 0.5).mean()),

        "gated_iou_mean": float(ious_gated.mean()),
        "gated_iou_median": float(np.median(ious_gated)),
        "gated_iou_gt_0p5": float((ious_gated > 0.5).mean()),

        # Ensure flow_wins_pct is defined before use
        "flow_wins_pct": float(flow_wins_pct),   # <-- NameError
    }

    fieldnames = list(row.keys())
    file_exists = os.path.isfile(args.out_csv)

    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Saved metrics to:", args.out_csv)

if __name__ == "__main__":
    main()