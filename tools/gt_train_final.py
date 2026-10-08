import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CAT_FEATURES = ["tr_id", "target_stop_id", "stop_key"]
DROP = {"sample_id", "T", "target_s", "target_delay_s", "target_class"}
TARGET = "target_delay_s"


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(Path("data/gt") / f"features_{name}.parquet")
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    return df


def main() -> None:
    train = load_split("train")
    test = load_split("test")
    full = pd.concat([train, test], ignore_index=True)

    cols = [c for c in full.columns if c not in DROP]
    num = [c for c in cols if c not in CAT_FEATURES]
    full[num] = full[num].apply(pd.to_numeric, errors="coerce")

    X, y = full[cols], full[TARGET].astype(float)
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=1000,
        allow_writing_files=False,
        l2_leaf_reg=3.0,
    )
    model.fit(Pool(X, y, cat_features=CAT_FEATURES))

    out = Path("data/gt/models")
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "catboost_final.cbm"))
    meta = {"features": cols, "cat": CAT_FEATURES, "trained_on": "train+test"}
    (out / "meta_final.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[final] saved model on {len(full)} rows -> {out / 'catboost_final.cbm'}")


if __name__ == "__main__":
    main()
