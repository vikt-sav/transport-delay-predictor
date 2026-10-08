"""unit_id → tr_id маппинг и выравнивание времени потока на день расписания."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtp.online import OnlinePipeline

BASE = ROOT / "dataset"


def make_pipeline(split: str = "validate") -> OnlinePipeline:
    sched = BASE / split / ("schedule_plan.csv" if split == "validate" else "schedule.csv")
    if not sched.exists():
        pytest.skip("dataset not available (gitignored) - CI-safe skip")
    return OnlinePipeline(str(sched), "data/gt/models/catboost_final.cbm", use_hmm_matching=False)


def test_map_tr_id_dataset_units():
    p = make_pipeline("validate")
    tr = pd.read_csv(BASE / "validate" / "traffic.csv", usecols=["tr_id", "unit_id"]).dropna()
    pairs = tr.drop_duplicates("unit_id")
    hits = sum(p.map_tr_id(int(r.unit_id)) == int(r.tr_id) for r in pairs.itertuples(index=False))
    assert hits == len(pairs), f"{hits}/{len(pairs)}"


def test_map_tr_id_unknown_unit_falls_back():
    p = make_pipeline("validate")
    assert p.map_tr_id(1166336) == 1166336


def test_align_ts_moves_stream_day_to_schedule_day():
    p = make_pipeline("validate")
    plan = p.schedule["plan_s"]
    lo, hi = float(plan.min()), float(plan.max())
    ref = float(plan.median())
    today = ref + 264 * 86400  # «сегодняшний» поток: далеко от дня расписания
    aligned = p.align_ts(today)
    assert lo - 86400 <= aligned <= hi + 86400  # попали в сутки расписания
    assert abs((aligned % 86400) - (today % 86400)) < 1e-6  # время суток сохранено


def test_align_ts_noop_for_dataset_stream():
    p = make_pipeline("validate")
    plan = p.schedule["plan_s"]
    t = float(plan.median())
    assert p.align_ts(t) == t


def test_align_ts_monotone():
    p = make_pipeline("validate")
    a = p.align_ts(1_790_000_000.0)
    b = p.align_ts(1_790_000_600.0)
    assert b > a
