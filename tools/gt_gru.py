"""PyTorch GRU sequence model over recent telemetry + static features.

Trains with L1 loss (MAE-consistent) and ensembles with the CatBoost model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtp.gt import load_points, load_traffic, lonlat_to_xy

K = 40
SEQ_DIM = 6
STATIC_COLS = ["cur_dev_s", "horizon_s", "prior_mean", "prior_known", "stops_left_in_trip", "speed_mean_10m", "age_last_s"]
STATIC_SCALE = np.array([300.0, 900.0, 300.0, 1.0, 10.0, 60.0, 300.0])
HIDDEN = 48


def build_sequences(traffic: pd.DataFrame, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    seqs = np.zeros((len(points), K, SEQ_DIM), dtype=np.float32)
    masks = np.zeros((len(points), K), dtype=np.float32)

    tr_points = points.reset_index(drop=True)
    for tr_id, grp in tr_points.groupby("tr_id"):
        veh = traffic[traffic["tr_id"] == tr_id]
        veh = veh[(veh["location_valid"] == True) & veh["lat"].notna()]
        t = veh["ts"].values.astype(np.int64)
        lat = veh["lat"].values.astype(float)
        lon = veh["lon"].values.astype(float)
        spd = veh["speed"].fillna(0.0).values.astype(float)
        hdg = veh["heading"].fillna(0.0).values.astype(float)
        lat0 = float(np.median(lat)) if len(lat) else 55.75
        lon0 = float(np.median(lon)) if len(lon) else 37.6
        y, x = lonlat_to_xy(lat, lon, lat0, lon0)

        for row_pos, p in grp.iterrows():
            cut = int(np.searchsorted(t, int(p["T_s"]), side="right"))
            lo = max(cut - K, 0)
            n = cut - lo
            if n >= 2:
                dt = np.diff(t[lo:cut]).astype(np.float32) / 60.0
                dx = np.diff(x[lo:cut]).astype(np.float32) / 100.0
                dy = np.diff(y[lo:cut]).astype(np.float32) / 100.0
                sp = spd[lo + 1 : cut].astype(np.float32) / 60.0
                hd = np.deg2rad(hdg[lo + 1 : cut]).astype(np.float32)
                m = n - 1
                seqs[row_pos, -m:, 0] = dt
                seqs[row_pos, -m:, 1] = sp
                seqs[row_pos, -m:, 2] = np.sin(hd)
                seqs[row_pos, -m:, 3] = np.cos(hd)
                seqs[row_pos, -m:, 4] = dx
                seqs[row_pos, -m:, 5] = dy
                masks[row_pos, -m:] = 1.0
    return seqs, masks


def static_matrix(feats: pd.DataFrame) -> np.ndarray:
    cols = feats[STATIC_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    cols = np.nan_to_num(cols, nan=0.0)
    return (cols / STATIC_SCALE).astype(np.float32)


class GRUPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(SEQ_DIM, HIDDEN, batch_first=True)
        self.static = nn.Sequential(nn.Linear(len(STATIC_COLS), 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(HIDDEN + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, mask, static):
        out, _ = self.gru(seq)
        denom = mask.sum(1, keepdim=True).clamp(min=1e-6)
        last = (out * mask.unsqueeze(-1)).sum(1) / denom
        h = torch.cat([last, self.static(static)], dim=1)
        return self.head(h).squeeze(-1)


def train(args) -> None:
    base = Path("dataset")
    tr_feats = pd.read_parquet("data/gt/features_train.parquet")
    te_feats = pd.read_parquet("data/gt/features_test.parquet")
    tr_traffic = load_traffic(base / "train" / "traffic.csv")
    te_traffic = load_traffic(base / "test" / "traffic.csv")
    tr_points = load_points(base / "labels" / "labels_train.csv")
    te_points = load_points(base / "labels" / "labels_test.csv")

    Xs, Ms = build_sequences(tr_traffic, tr_points)
    Xt, Mt = build_sequences(te_traffic, te_points)
    Ss, St = static_matrix(tr_feats), static_matrix(te_feats)
    ytr = torch.tensor(tr_feats["target_delay_s"].values, dtype=torch.float32)
    yte = te_feats["target_delay_s"].values

    torch.manual_seed(42)
    model = GRUPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xs), torch.tensor(Ms), torch.tensor(Ss), ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for seq, mask, static, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(seq, mask, static), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(yb)
        model.eval()
        with torch.no_grad():
            pt = model(torch.tensor(Xt), torch.tensor(Mt), torch.tensor(St)).numpy()
        print(f"[gru] epoch {epoch}: train_l1={total/len(ds):.1f}, test_mae={float(np.mean(np.abs(yte - pt))):.1f}")

    out = Path("data/gt/models")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "gru.pt")
    print(f"[gru] saved -> {out / 'gru.pt'}")


def run_model(feats: pd.DataFrame, traffic: pd.DataFrame, points: pd.DataFrame) -> np.ndarray:
    X, M = build_sequences(traffic, points)
    S = static_matrix(feats)
    model = GRUPredictor()
    model.load_state_dict(torch.load("data/gt/models/gru.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X), torch.tensor(M), torch.tensor(S)).numpy()


def predict(args) -> None:
    base = Path("dataset")
    va_feats = pd.read_parquet("data/gt/features_validate.parquet")
    va_traffic = load_traffic(base / "validate" / "traffic.csv")
    va_points = load_points(base / "validate" / "points.csv")
    pred = run_model(va_feats, va_traffic, va_points)
    np.save("data/gt/gru_preds_validate.npy", pred)
    print(f"[gru] saved {len(pred)} predictions -> data/gt/gru_preds_validate.npy")


def ensemble(args) -> None:
    from catboost import CatBoostRegressor

    te_feats = pd.read_parquet("data/gt/features_test.parquet")
    yte = te_feats["target_delay_s"].values
    cb = CatBoostRegressor()
    cb.load_model("data/gt/models/catboost_mae.cbm")
    meta = json.loads(Path("data/gt/models/meta.json").read_text(encoding="utf-8"))
    cols = meta["features"]
    for c in meta["cat"]:
        te_feats[c] = te_feats[c].astype(str)
    num = [c for c in cols if c not in meta["cat"]]
    te_feats[num] = te_feats[num].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        if c not in te_feats.columns:
            te_feats[c] = np.nan
    cb_pred = np.clip(cb.predict(te_feats[cols]), -420, 700)

    base = Path("dataset")
    te_traffic = load_traffic(base / "test" / "traffic.csv")
    te_points = load_points(base / "labels" / "labels_test.csv")
    nn_pred = run_model(te_feats, te_traffic, te_points)

    mae_cb = float(np.mean(np.abs(yte - cb_pred)))
    mae_nn = float(np.mean(np.abs(yte - nn_pred)))
    print(f"[ensemble] cat mae={mae_cb:.1f}, gru mae={mae_nn:.1f}")
    best = (0.0, float("inf"))
    for w in np.arange(0.0, 1.01, 0.05):
        p = w * cb_pred + (1 - w) * nn_pred
        mae = float(np.mean(np.abs(yte - p)))
        if mae < best[1]:
            best = (float(w), mae)
    print(f"[ensemble] best w_cat={best[0]:.2f}, mae={best[1]:.1f}")
    Path("data/gt/models/ensemble.json").write_text(
        json.dumps({"w_cat": best[0], "test_mae": best[1], "cat_mae": mae_cb, "gru_mae": mae_nn}),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=25)
    sub.add_parser("predict")
    sub.add_parser("ensemble")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        ensemble(args)


if __name__ == "__main__":
    main()
