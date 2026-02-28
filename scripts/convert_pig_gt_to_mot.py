import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_gt", required=True, help="input gt.txt (your pig format)")
    ap.add_argument("--out_gt", required=True, help="output gt.txt in MOTChallenge format")
    ap.add_argument("--class_id", type=int, default=1, help="use 1 to match TrackEval 'pedestrian'")
    args = ap.parse_args()

    df = pd.read_csv(args.in_gt, header=None)

    # Accept common variants:
    # - 6 cols: frame,id,x,y,w,h
    # - 7 cols: + score/conf
    # - 9 cols: already MOT-like
    if df.shape[1] == 6:
        df.columns = ["frame","id","x","y","w","h"]
        df["conf"] = 1
        df["cls"] = args.class_id
        df["vis"] = 1
    elif df.shape[1] == 7:
        df.columns = ["frame","id","x","y","w","h","conf"]
        df["conf"] = 1  # IMPORTANT: GT conf must be 1 to be valid in TrackEval
        df["cls"] = args.class_id
        df["vis"] = 1
    elif df.shape[1] >= 9:
        # Assume first 6 are frame,id,x,y,w,h and then conf,cls,vis exist
        df = df.iloc[:, :9].copy()
        df.columns = ["frame","id","x","y","w","h","conf","cls","vis"]
        df["conf"] = 1
        df["cls"] = args.class_id
        df["vis"] = df["vis"].fillna(1)
    else:
        raise RuntimeError(f"Unsupported gt format with {df.shape[1]} columns")

    # Ensure ints where needed
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    os.makedirs(os.path.dirname(args.out_gt), exist_ok=True)
    df.to_csv(args.out_gt, header=False, index=False)

if __name__ == "__main__":
    main()