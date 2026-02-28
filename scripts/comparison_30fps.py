import os
import glob
import pandas as pd

# Run this FROM TrackEval/ as:
#   python ../scripts/compare_pig30fps.py
TRACK_EVAL_ROOT = os.getcwd()
BENCH = "Pig30fps-train"
ROOT = os.path.join(TRACK_EVAL_ROOT, "data", "trackers", "mot_challenge", BENCH)

# Pick your baseline tracker folder name (must exist under ROOT)
BASELINE = "pig30fps_bytetrack_detonly"   # or "baseline_detonly"

csvs = glob.glob(os.path.join(ROOT, "*", "pedestrian_detailed.csv"))
if not csvs:
    raise SystemExit(f"No pedestrian_detailed.csv found under: {ROOT}")

rows = []
for csv_path in csvs:
    tracker = os.path.basename(os.path.dirname(csv_path))
    df = pd.read_csv(csv_path)

    # TrackEval detailed.csv usually has a "seq" column with "COMBINED"
    seq_col = None
    for c in df.columns:
        if c.lower() in ["seq", "sequence", "name"]:
            seq_col = c
            break
    if seq_col is None:
        # fallback: try first col
        seq_col = df.columns[0]

    comb = df[df[seq_col].astype(str).str.upper() == "COMBINED"]
    if comb.empty:
        # Sometimes it's "COMBINED" without a column name; fallback: last row
        comb = df.tail(1)

    comb = comb.iloc[0].to_dict()
    comb["tracker"] = tracker
    rows.append(comb)

out = pd.DataFrame(rows)

# normalize key metric column names (TrackEval can prefix them)
def find_col(target):
    # exact
    if target in out.columns:
        return target
    # case-insensitive contains
    candidates = [c for c in out.columns if c.lower() == target.lower()]
    if candidates:
        return candidates[0]
    candidates = [c for c in out.columns if target.lower() in c.lower()]
    return candidates[0] if candidates else None

col_hota = find_col("HOTA")
col_idf1 = find_col("IDF1")
col_mota = find_col("MOTA")
col_ids  = find_col("IDSW")  # often IDSW
col_frag = find_col("Frag")

needed = {"HOTA": col_hota, "IDF1": col_idf1, "MOTA": col_mota, "IDSW": col_ids, "Frag": col_frag}
missing = [k for k,v in needed.items() if v is None]
if missing:
    print("WARNING: missing columns:", missing)
    print("Available columns:", list(out.columns))

# Keep a clean view
keep_cols = ["tracker"]
for k,v in needed.items():
    if v is not None:
        keep_cols.append(v)

view = out[keep_cols].copy()

# convert numeric cols
for c in view.columns:
    if c != "tracker":
        view[c] = pd.to_numeric(view[c], errors="coerce")

# add deltas vs baseline
if BASELINE in view["tracker"].values:
    base = view[view["tracker"] == BASELINE].iloc[0]
    for k,v in needed.items():
        if v is not None:
            view[f"d{k}_vs_{BASELINE}"] = view[v] - base[v]
else:
    print(f"WARNING: baseline '{BASELINE}' not found under {ROOT}")
    print("Trackers found:", sorted(view["tracker"].tolist()))

# sort primarily by HOTA, then IDF1
sort_cols = [c for c in [col_hota, col_idf1, col_mota] if c is not None]
view = view.sort_values(sort_cols, ascending=False)

print("\n=== Pig30fps comparison (COMBINED) ===")
print(view.to_string(index=False))

out_csv = os.path.join(TRACK_EVAL_ROOT, f"pig30fps_compare_COMBINED_vs_{BASELINE}.csv")
view.to_csv(out_csv, index=False)
print("\nWrote:", out_csv)