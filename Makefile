.PHONY: install data etl train evaluate strategy monitor retrain docs all test api docker lint clean
PY ?= python

install:        ; pip install -r requirements.txt
data:           ; $(PY) -m src.generate_data
etl:            ; $(PY) -m src.etl
train:          ; $(PY) -m src.train
evaluate:       ; $(PY) -m src.evaluate
strategy:       ; $(PY) -m src.strategy
monitor:        ; $(PY) -m src.monitoring
retrain:        ; $(PY) -m src.retrain
docs:           ; $(PY) -m src.make_docs
all:            ; $(PY) -m src.pipeline && $(PY) -m src.retrain && $(PY) -m src.baseline_legacy && $(PY) -m src.make_docs
test:           ; $(PY) -m pytest -q
lint:           ; ruff check src api tests
api:            ; uvicorn api.app:app --reload --port 8000
docker:         ; docker build -t telco-credit-scoring . && docker run -p 8000:8000 telco-credit-scoring
clean:          ; rm -rf data/raw/* data/clean/* data/features/* models/registry/* mlruns
