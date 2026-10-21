"""Run the whole pipeline end to end:  python -m src.pipeline [--n-customers N] [--skip-generate]"""
from __future__ import annotations

import argparse

from . import etl, evaluate, generate_data, monitoring, strategy, train
from .config import layer_dir, load_config
from .utils import get_logger, write_layer

log = get_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-customers", type=int, default=None)
    ap.add_argument("--skip-generate", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not a.skip_generate:
        write_layer(generate_data.generate(cfg, n_customers=a.n_customers), layer_dir(cfg, "raw"))
    etl.run(cfg)
    train.train(cfg)
    evaluate.run(cfg)
    strategy.run(cfg)
    monitoring.monitor(cfg)
    log.info("pipeline complete; see reports/")


if __name__ == "__main__":
    main()
