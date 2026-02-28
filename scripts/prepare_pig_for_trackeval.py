import os
import shutil
import argparse


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def copy_gt(pig_root, out_root):
    gt_src = os.path.join(pig_root, "gt")
    gt_dst_root = os.path.join(out_root, "gt")

    ensure_dir(gt_dst_root)

    for f in os.listdir(gt_src):
        if not f.endswith("_gt.txt"):
            continue

        seq = f.replace("_gt.txt", "")
        seq_dir = os.path.join(gt_dst_root, seq, "gt")
        ensure_dir(seq_dir)

        shutil.copy(
            os.path.join(gt_src, f),
            os.path.join(seq_dir, "gt.txt")
        )
        print("GT copied:", seq)


def copy_tracker(pig_root, out_root, tracker_name):
    tracker_src = os.path.join(pig_root, "results", tracker_name)
    tracker_dst = os.path.join(
        out_root, "trackers", tracker_name, "data"
    )
    ensure_dir(tracker_dst)

    for f in os.listdir(tracker_src):
        if not f.endswith(".txt"):
            continue

        shutil.copy(
            os.path.join(tracker_src, f),
            os.path.join(tracker_dst, f)
        )
        print("Tracker result copied:", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pig_root", required=True)
    ap.add_argument("--tracker_name", required=True)
    ap.add_argument("--out_root", default="data_mot/pig_eval")

    args = ap.parse_args()

    copy_gt(args.pig_root, args.out_root)
    copy_tracker(args.pig_root, args.out_root, args.tracker_name)


if __name__ == "__main__":
    main()