import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CAT_FEATURES = ["tr_id", "target_stop_id", "stop_key"]
DROP = {"sample_id", "T", "target_s", "target_delay_s", "target_class"}
CLIP_LO, CLIP_HI = -420.0, 700.0


def main() -> None:
    ap = argparse()
    args = ap.parse_args()

    model = CatBoostRegressor()
    model.load_model(args.model)
    meta = json.loads(Path(args.model).parent.joinpath("meta.json").read_text(encoding="utf-8"))
    cols = meta["features"]

    df = pd.read_parquet(args.features)
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    num = [c for c in cols if c not in CAT_FEATURES]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    pred = np.clip(model.predict(df[cols]), CLIP_LO, CLIP_HI)
    sub = pd.read_csv(args.sample, sep=";")
    mapping = dict(zip(df["sample_id"], pred, strict=False))
    sub["prediction"] = sub["sample_id"].map(mapping)
    assert sub["prediction"].notna().all(), "uncovered sample_ids"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, sep=";", index=False, encoding="utf-8")
    print(f"[predict] {len(sub)} rows -> {out}")
    print(sub["prediction"].describe().round(1).to_string())


def argparse():
    import argparse as ap_mod

    p = ap_mod.ArgumentParser()
    p.add_argument("--features", default="data/gt/features_validate.parquet")
    p.add_argument("--model", default="data/gt/models/catboost_mae.cbm")
    p.add_argument("--sample", default="dataset/sample_submission.csv")
    p.add_argument("--out", default="data/gt/submission.csv")
    return p


if __name__ == "__main__":
    main()
