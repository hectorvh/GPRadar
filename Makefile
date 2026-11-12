# GPRadar: one target per pipeline step. `make help` lists them in order.
#
# Every target runs its script inside the right conda environment with
# `conda run`, so no environment needs to be activated first. Override the
# environment names on the command line, e.g. `make test GPRMAX_ENV=myenv`.
# Long jobs (scenes, train, evaluate) need CONFIRM=1. No target deletes
# anything under data/raw/.

CONDA      ?= conda
GPRMAX_ENV ?= gprMax-v4
GPRPY_ENV  ?= gprpy
DET_ENV    ?= gprdet
PYTHON     ?= python
CONFIRM    ?= 0
N_SCENES   ?= 200
SCENE_MIN  ?= 10

SIM   = $(CONDA) run -n $(GPRMAX_ENV) --no-capture-output $(PYTHON)
GPRPY = $(CONDA) run -n $(GPRPY_ENV) --no-capture-output $(PYTHON)
DET   = $(CONDA) run -n $(DET_ENV) --no-capture-output $(PYTHON)

ONNX        = runs/detect/A_f100_seed2/weights/best.onnx
SPLIT_CSV   = data/splits/mendeley_originals.csv
SPLIT_CHECK = data/splits/mendeley_originals.check.csv
TEST_LOCK   = configs/mendeley_test_lock.txt

export MPLBACKEND := Agg

.DEFAULT_GOAL := help
.PHONY: help test figures-sim figure-real gprpy-reference gprpy-compare inventory \
	split scenes synth real-yolo train-smoke train evaluate compare export-onnx \
	demo-samples sample-data demo

help: ## List the targets in pipeline order
	@echo "Usage: make <target> [CONFIRM=1] [GPRMAX_ENV=...] [GPRPY_ENV=...] [DET_ENV=...]"
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z0-9-]*:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test: ## Run the test suite (gprMax-v4); uses data/sample/ when the big data is absent
	$(SIM) -m pytest test -q

figures-sim: ## Pipe-study B-scans and figure (01a, needs gprMax) and the sim processing figure (03a)
	$(SIM) src/01a_pipe_bscan_study.py
	$(SIM) src/03a_plot_preprocessing.py --profile sim

figure-real: ## Processing figures for the Dinner Point Avenue field profile (03a)
	$(SIM) src/03a_plot_preprocessing.py --profile dinnerpoint

gprpy-reference: ## GPRPy dewow/AGC reference arrays for Dinner Point Avenue (02c, gprpy env)
	$(GPRPY) src/02c_gprpy_reference.py

gprpy-compare: ## Compare 02b dewow/AGC with the GPRPy reference (03b)
	$(SIM) src/03b_compare_gprpy.py

inventory: ## Read-only inventory of the Mendeley dataset (04a)
	$(SIM) src/04a_inventory_mendeley.py

split: ## Rebuild the Mendeley split and check it equals the committed, locked one (04b)
	@test -f $(TEST_LOCK) || { echo "$(TEST_LOCK) is missing: refusing to create a new locked split"; exit 1; }
	$(SIM) src/04b_split_mendeley.py split --split-csv $(SPLIT_CHECK)
	@if cmp -s $(SPLIT_CHECK) $(SPLIT_CSV); then \
		echo "split reproduced: identical to $(SPLIT_CSV)"; rm -f $(SPLIT_CHECK); \
	else \
		echo "rebuilt split differs from $(SPLIT_CSV); the committed split was NOT changed (see $(SPLIT_CHECK))"; exit 1; \
	fi

scenes: ## Simulate N_SCENES field_400mhz scenes with gprMax (01c); about 10 min each; needs CONFIRM=1
	@echo "$(N_SCENES) scenes x about $(SCENE_MIN) min (400 MHz, 8 threads) = about $$(( $(N_SCENES) * $(SCENE_MIN) / 60 )) hours. Finished scenes are skipped."
	@test "$(CONFIRM)" = "1" || { echo "Run again with CONFIRM=1 to start."; exit 1; }
	$(SIM) src/01c_scene_generator.py --preset field_400mhz --seed 0 --n $(N_SCENES) --out data/sim/scenes --run

synth: ## Render the synthetic YOLO dataset and check figures from the scenes (05a)
	$(SIM) src/05a_render_synthetic.py

real-yolo: ## Real YOLO folders, validation split and nested label subsets (06a, gprdet)
	$(DET) src/06a_prepare_real_yolo.py

train-smoke: ## Two-epoch smoke run of the detector training (06b --smoke, gprdet)
	$(DET) src/06b_train_detectors.py --smoke

train: ## Full training grid, 12 fine-tunes + 3 stage-1 runs (06b, gprdet); hours; needs CONFIRM=1
	@echo "Full grid: about 19 h on a Quadro P1000 without early stopping. Finished runs are skipped."
	@test "$(CONFIRM)" = "1" || { echo "Run again with CONFIRM=1 to start."; exit 1; }
	@mkdir -p results
	$(DET) src/06b_train_detectors.py --seeds 0 1 2 --fractions 0.25 1.0 --regimes A C 2>&1 | tee results/train_all.log

evaluate: ## Evaluate all 15 detectors on the locked test set (06c, gprdet); use once; needs CONFIRM=1
	@echo "WARNING: this reads the locked test set. It is meant to be used once, after all training,"
	@echo "and the reported results already used it. For a check, run: $(DET) src/06c_evaluate_detectors.py --dry-run"
	@test "$(CONFIRM)" = "1" || { echo "Run again with CONFIRM=1 to evaluate on the test set."; exit 1; }
	$(DET) src/06c_evaluate_detectors.py

compare: ## Rank the runs and confirm the ship candidate (06d, gprdet)
	$(DET) src/06d_compare_runs.py

export-onnx: ## Export the shipped detector to ONNX and check parity on 20 train images (06f, gprdet)
	$(DET) src/06f_export_onnx.py --check 20

demo-samples: ## Web demo samples, manifest and golden detections in exports/ (07a)
	$(SIM) src/07a_export_demo_samples.py export
	$(DET) src/07a_export_demo_samples.py golden

sample-data: ## Rebuild the small committed data/sample/ from the full data (00)
	$(SIM) src/00_make_sample_data.py

demo: ## Process the sample scans and draw figures/demo/*.png from data/sample/ only (08a)
	@start=$$(date +%s); \
	$(SIM) src/08a_demo.py run || exit 1; \
	if [ -f $(ONNX) ] && $(CONDA) env list | grep -qE '^$(DET_ENV)[[:space:]]'; then \
		$(DET) src/08a_demo.py detect || exit 1; \
	else \
		echo "detect: skipped ($(DET_ENV) environment or $(ONNX) not found)"; \
	fi; \
	echo "demo finished in $$(( $$(date +%s) - start )) s"
