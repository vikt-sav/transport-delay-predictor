import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CAT_FEATURES = ["tr_id", "target_stop_id", "stop_key"]
DROP = ["sample_id", "T", "target_s", "target_delay_s", "target_class"]
TARGET = "target_delay_s"


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(Path("data/gt") / f"features_{name}.parquet")
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP]


def evaluate(name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y - pred)))
    mae_zero = float(np.mean(np.abs(y)))
    return {"mae": mae, "mae_zero": mae_zero}


def main() -> None:
    train = load_split("train")
    test = load_split("test")

    cols = feature_cols(train)
    cols = [c for c in cols if c in test.columns]
    num = [c for c in cols if c not in CAT_FEATURES]
    train[num] = train[num].apply(pd.to_numeric, errors="coerce")
    test[num] = test[num].apply(pd.to_numeric, errors="coerce")

    Xtr, ytr = train[cols], train[TARGET].astype(float)
    Xte, yte = test[cols], test[TARGET].astype(float)

    mae_zero = float(np.mean(np.abs(yte)))
    mae_cur = float(np.mean(np.abs(yte - Xte["cur_dev_s"].astype(float))))
    print(f"[train] test baseline: mae_zero={mae_zero:.1f}, mae_cur_dev={mae_cur:.1f}")

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=500,
        allow_writing_files=False,
        l2_leaf_reg=3.0,
    )
    tr_pool = Pool(Xtr, ytr, cat_features=CAT_FEATURES)
    te_pool = Pool(Xte, yte, cat_features=CAT_FEATURES)
    model.fit(tr_pool, eval_set=te_pool, use_best_model=True)
    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(yte - pred)))
    print(f"[train] TEST MAE={mae:.1f} (baseline cur_dev {mae_cur:.1f}, zero {mae_zero:.1f})")

    imp = model.get_feature_importance()
    order = np.argsort(-imp)
    print("=== top-15 features ===")
    for i in order[:15]:
        print(f"{cols[i]:28s} {imp[i]:6.2f}")

    out = Path("data/gt/models")
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "catboost_mae.cbm"))
    Path(out / "meta.json").write_text(
        json_meta(cols, mae, mae_cur, mae_zero), encoding="utf-8"
    )


def json_meta(cols, mae, mae_cur, mae_zero) -> str:
    import json

    return json.dumps({"features": cols, "cat": CAT_FEATURES, "test_mae": mae, "mae_cur": mae_cur, "mae_zero": mae_zero}, indent=2)


if __name__ == "__main__":
    main()
