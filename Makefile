# ==========================================
# Makefile
# ==========================================
# This Makefile automates the entire simulation,
# profiling, complexity estimation, and figure-generation pipeline.
#
# Main targets:
#   make all                      : Run simulations + profiling + complexity + all figures
#   make simulate_all             : Run all simulation variants
#   make simulate_baseline
#   make simulate_cholesky
#   make simulate_cholesky_parallelization
#   make profile                  : Build aggregated profiling CSVs
#   make complexity               : Estimate log-log complexity from profiling outputs
#   make result_figures          : Generate figures from simulation outputs
#   make profiling_figures       : Generate figures from profiling outputs
#   make complexity_figures      : Generate complexity bar charts
#   make clean                   : Remove generated outputs
#   make test                    : Run pytest
# ==========================================

# ---- Basic variables ----
PY                  ?= python3
SRC_DIR             ?= src
PERFORMANCE_DIR     ?= performance

RESULTS_DIR         ?= results
RESULTS_TABLES_DIR  ?= $(RESULTS_DIR)/tables
RESULTS_FIGURES_DIR ?= $(RESULTS_DIR)/figures

TIMINGS_DIR         ?= timings
TIMINGS_TABLES_DIR  ?= $(TIMINGS_DIR)/tables
TIMINGS_FIGURES_DIR ?= $(TIMINGS_DIR)/figures

SIM_SCRIPT_BASELINE                 := $(SRC_DIR)/simulation_baseline.py
SIM_SCRIPT_CHOLESKY                 := $(SRC_DIR)/simulation_cholesky.py
SIM_SCRIPT_CHOLESKY_PARALLELIZATION := $(SRC_DIR)/simulation_cholesky+parallelization.py

PLOT_SCRIPT            := $(SRC_DIR)/plot.py
PLOT_PROFILING_SCRIPT  := $(PERFORMANCE_DIR)/plot_profiling.py
PLOT_COMPLEXITY_SCRIPT := $(PERFORMANCE_DIR)/plot_complexity.py

PROFILING_SCRIPT       := $(PERFORMANCE_DIR)/profiling.py
COMPLEXITY_SCRIPT      := $(PERFORMANCE_DIR)/complexity.py

.PHONY: all simulate simulate_all \
        simulate_baseline simulate_cholesky simulate_cholesky_parallelization \
        profile complexity \
        result_figures profiling_figures complexity_figures \
        clean test dirs

# ------------------------------------------
# Full pipeline
# ------------------------------------------

all: simulate_all profile complexity result_figures profiling_figures complexity_figures
	@echo "[all] Full pipeline complete."

simulate: simulate_all

simulate_all: simulate_baseline simulate_cholesky simulate_cholesky_parallelization
	@echo "[simulate_all] All simulation variants completed."

# ------------------------------------------
# Simulation targets
# ------------------------------------------

simulate_baseline: dirs
	@echo "[simulate_baseline] Running baseline simulation..."
	@$(PY) $(SIM_SCRIPT_BASELINE)
	@echo "[simulate_baseline] Done."

simulate_cholesky: dirs
	@echo "[simulate_cholesky] Running Cholesky simulation..."
	@$(PY) $(SIM_SCRIPT_CHOLESKY)
	@echo "[simulate_cholesky] Done."

simulate_cholesky_parallelization: dirs
	@echo "[simulate_cholesky_parallelization] Running Cholesky + parallelization simulation..."
	@$(PY) $(SIM_SCRIPT_CHOLESKY_PARALLELIZATION)
	@echo "[simulate_cholesky_parallelization] Done."

# ------------------------------------------
# Profiling and Complexity
# ------------------------------------------

profile: dirs
	@echo "[profile] Building profiling CSVs..."
	@$(PY) $(PROFILING_SCRIPT)
	@echo "[profile] Profiling complete."

complexity: profile
	@echo "[complexity] Estimating complexity..."
	@$(PY) $(COMPLEXITY_SCRIPT)
	@echo "[complexity] Complexity CSV written under $(TIMINGS_TABLES_DIR)."

# ------------------------------------------
# Figures
# ------------------------------------------

result_figures: $(RESULTS_FIGURES_DIR)
	@echo "[result_figures] Generating simulation result figures..."
	@$(PY) $(PLOT_SCRIPT)
	@echo "[result_figures] Figures saved to $(RESULTS_FIGURES_DIR)."

profiling_figures: $(TIMINGS_FIGURES_DIR)
	@echo "[profiling_figures] Generating profiling figures..."
	@$(PY) $(PLOT_PROFILING_SCRIPT)
	@echo "[profiling_figures] Figures saved to $(TIMINGS_FIGURES_DIR)."

complexity_figures: $(TIMINGS_FIGURES_DIR)
	@echo "[complexity_figures] Generating complexity figures..."
	@$(PY) $(PLOT_COMPLEXITY_SCRIPT)
	@echo "[complexity_figures] Figures saved to $(TIMINGS_FIGURES_DIR)."

# Backward compatibility alias
figures: result_figures
	@echo "[figures] Alias for result_figures."

# ------------------------------------------
# Directory utilities
# ------------------------------------------

dirs: $(RESULTS_TABLES_DIR) $(RESULTS_FIGURES_DIR) $(TIMINGS_TABLES_DIR) $(TIMINGS_FIGURES_DIR)

$(RESULTS_TABLES_DIR):
	@mkdir -p $(RESULTS_TABLES_DIR)

$(RESULTS_FIGURES_DIR):
	@mkdir -p $(RESULTS_FIGURES_DIR)

$(TIMINGS_TABLES_DIR):
	@mkdir -p $(TIMINGS_TABLES_DIR)

$(TIMINGS_FIGURES_DIR):
	@mkdir -p $(TIMINGS_FIGURES_DIR)

# ------------------------------------------
# Clean
# ------------------------------------------

clean:
	@echo "[clean] Removing generated outputs..."
	@rm -rf $(RESULTS_DIR) $(TIMINGS_DIR)
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "[clean] Done."

# ------------------------------------------
# Tests
# ------------------------------------------

test:
	@echo "[test] Running pytest..."
	@$(PY) -m pytest -q
	@echo "[test] All tests passed."