"""Train -> register -> serve on a tiny dataset in an isolated temp registry."""
import numpy as np
import pytest

from src import metrics as M
from src.features import FEATURES, build_features
from src.registry import update_metadata
from src.scoring import predict_frame
from src.serving import ScoringService
from src.train import MODEL_NAME, split_frames, train
from tests.helpers import small_tables, tmp_cfg


def _trained():
    cfg0, _, clean = small_tables(n=500, seed=11)
    f = build_features(clean, cfg0)
    cfg = tmp_cfg()
    version, meta = train(cfg, f)
    return cfg, f, version, meta


def test_train_registers_and_beats_random_out_of_time():
    cfg, f, version, meta = _trained()
    from src.registry import load_bundle
    bundle, _ = load_bundle(cfg, MODEL_NAME)
    _, _, te = split_frames(f, cfg["split"]["train_end"], cfg["split"]["valid_end"])
    p = predict_frame(bundle, te)
    assert M.auc(te["default"], p["pd"]) > 0.58
    assert p["pd"].between(0, 1).all() and p["score_200_800"].between(200, 800).all()
    assert meta["champion"] in {"scorecard", "gbm"} and version == "v1"


def test_serving_returns_decision_reasons_and_handles_missing():
    cfg, f, _, _ = _trained()
    update_metadata(cfg, MODEL_NAME, "latest", {"policy": {
        "cutoff_pd": 0.12, "tier_pd_edges": [0.06, 0.09], "tier_names": ["T1", "T2", "T3"], "tier_limit_mult": [1.5, 1.0, 0.6]}})
    svc = ScoringService(cfg)
    rows = f[FEATURES].head(60).to_dict("records")
    outs = svc.score_batch(rows)
    assert len(outs) == 60
    for o in outs:
        assert {"pd", "decision", "scorecard_points", "reason_codes", "model_version"} <= set(o)
        assert o["decision"] == ("APPROVE" if o["pd"] <= 0.12 else "DECLINE")
        assert (o["risk_tier"] is not None) == (o["decision"] == "APPROVE")
    assert svc.score({})["decision"] in {"APPROVE", "DECLINE"}          # all-missing input still scores
    pdv = np.array([o["pd"] for o in outs]); pts = np.array([o["scorecard_points"] for o in outs])
    if svc.bundle["champion"] == "scorecard":
        assert np.corrcoef(pdv, pts)[0, 1] < -0.95


def test_fastapi_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import api.app as app_module
    cfg, f, _, _ = _trained()
    app_module._service = ScoringService(cfg)
    c = TestClient(app_module.app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/model").json()["model"] == MODEL_NAME
    r = c.post("/score", json={"customer_id": "C1", "features": f[FEATURES].iloc[0].dropna().to_dict()})
    assert r.status_code == 200 and r.json()["customer_id"] == "C1" and 0 <= r.json()["pd"] <= 1
    rb = c.post("/score/batch", json={"items": [{"customer_id": "a", "features": {}}, {"customer_id": "b", "features": {}}]})
    assert rb.status_code == 200 and len(rb.json()) == 2


def test_evaluate_and_strategy_handle_retrained_model_without_test_window():
    from src import evaluate, strategy
    from src.config import layer_dir
    from src.utils import write_table
    cfg0, _, clean = small_tables(n=500, seed=11)
    f = build_features(clean, cfg0)
    cfg = tmp_cfg()
    write_table(f, layer_dir(cfg, "features") / "features.csv.gz")
    train(cfg, f, cfg["split"]["train_end"], str(f["snapshot_date"].max().date()))   # no loans after valid_end
    res = evaluate.run(cfg)
    assert res["holdout_split"] == "valid" and "test" not in res["models"]["champion"]
    assert strategy.run(cfg) == {}
