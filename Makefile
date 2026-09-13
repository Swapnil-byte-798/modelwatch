VENV := .venv39/bin
PY   := $(VENV)/python

.PHONY: help data model corpus aa benchmark figures card report test reproduce clean-cache

help:
	@echo "make data       # download ACS cells -> data/windows/*.parquet (slow, network)"
	@echo "make model      # train + freeze the monitored model"
	@echo "make corpus     # ground truth + detector battery per window"
	@echo "make aa         # A/A harness: false-alarm rate on known-no-drift data"
	@echo "make benchmark  # grade every detector against realised degradation"
	@echo "make figures    # the quadrant scatter"
	@echo "make card       # regenerate MODEL_CARD.md from measured results"
	@echo "make report     # regenerate README.md from measured results"
	@echo "make test       # unit tests"
	@echo "make reproduce  # everything except the download"

data:
	$(PY) -m modelwatch.data

model:
	$(PY) -m modelwatch.model

corpus:
	$(PY) -m modelwatch.corpus

aa:
	$(PY) -m modelwatch.aa_harness

benchmark:
	$(PY) -m modelwatch.benchmark

figures:
	$(PY) -m modelwatch.figures

card:
	$(PY) -m modelwatch.model_card

test:
	$(VENV)/pytest -q tests/

# Regenerates every number in the README from the cached windows, fixed seeds.
report:
	$(PY) -m modelwatch.report

# Regenerates every number in the README from the cached windows, fixed seeds.
reproduce: model corpus aa benchmark figures card report
	@echo "reproduced: artifacts/ and report/ regenerated"

clean-cache:
	rm -rf data/acs_cache
