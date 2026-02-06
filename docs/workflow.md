# Reproduction Workflow

## Synthetic Lethality QSP Project
Step-by-step guide to reproduce all results from the manuscript.

---

## Prerequisites

- **Python 3.9+** with packages listed in `requirements.txt`
- **R 4.2+** with packages listed in `install.R` (for statistical analysis/visualization)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
Rscript install.R

# 2. Run the full pipeline
python scripts/run_all.py
```

---

## Detailed Pipeline Steps

### Phase 1: Initial Drug Screening

**Script:** `src/run_enhanced_screening.py`

Runs the 12-state ODE model for all 9 drugs in both ATM-proficient (WT) and ATM-deficient genotypes. Produces:
- `data/screening/complete_screening_results.csv` - SL scores, therapeutic indices, pathway metrics
- `data/time_series/*.csv` - 48-hour time-course simulations (16 files)

```bash
cd github_QSP_SL
python -c "import sys; sys.path.insert(0,'src'); from run_enhanced_screening import *; main()"
```

### Phase 2a: Parameter Optimization

**Script:** `scripts/run_phase2_optimization.py`

Optimizes apoptosis rate constants to achieve realistic SL scores (2-5x). Tests multiple reduction factors and drug concentrations.

- **Output:** `data/parameters/optimal_apoptosis_parameters.json`, `data/dose_response/*.csv`

### Phase 2b: Sensitivity Analysis

**Script:** `scripts/run_sensitivity_analysis.py`

Local sensitivity analysis on all key model parameters. Identifies which parameters have the greatest impact on SL scores.

- **Output:** `data/sensitivity_analysis/sensitivity_analysis_results.csv`, `data/sensitivity_analysis/sensitivity_summary.csv`

### Phase 2c: Concentration-Response Curves

**Script:** `scripts/run_dose_response.py`

Full dose-response analysis for all 9 drugs with IC50 calculations and therapeutic window identification.

- **Output:** `data/concentration_response/concentration_response_data.csv`, `data/concentration_response/ic50_summary.csv`

### Phase 3: Publication Figures

**Script:** `scripts/generate_figures.py`

Generates publication-quality figures from the screening results and time-series data.

- **Output:** `figures/generated/`

### Phase 4: Statistical Analysis (R)

**Scripts:** `R/statistical_analysis_framework.R`, `R/statistical_correction_integration.R`

Comprehensive statistical analysis including FDR corrections, pathway correlations, and reproducibility assessment.

- **Output:** `data/statistical_analysis/*.json`

### Phase 5: Publication Visualization (R)

**Script:** `R/publication_visualization_framework.R`

R-based publication figures with ggplot2 styling.

---

## Pipeline Automation

The master script `scripts/run_all.py` runs Phase 2a through Phase 3 in sequence:

```bash
python scripts/run_all.py
```

This executes:
1. Parameter optimization (Phase 2a)
2. Sensitivity analysis (Phase 2b)
3. Concentration-response curves (Phase 2c)
4. Publication figure generation (Phase 3)

Phase 1 data (screening results, time series) is pre-computed and included in `data/`.

---

## Key Output Files

| File | Description |
|------|-------------|
| `data/screening/complete_screening_results.csv` | Full drug screening with SL scores |
| `data/parameters/optimal_apoptosis_parameters.json` | Optimized model parameters |
| `data/concentration_response/ic50_summary.csv` | IC50 values for all drugs |
| `data/sensitivity_analysis/sensitivity_summary.csv` | Parameter sensitivity ranking |
| `figures/Figure_1.pdf` | Main manuscript Figure 1 |
| `figures/Figure_2.png` | Main manuscript Figure 2 |
| `figures/Figure_3.png` | Main manuscript Figure 3 |

---

## Model Description

The QSP model consists of **12 coupled ordinary differential equations** representing the DDR pathway:

1. DNA double-strand breaks (DSB)
2. ATM kinase activation
3. ATR kinase activation
4. CHK1 kinase activation
5. CHK2 kinase activation
6. p53 tumor suppressor activation
7. p21 CDK inhibitor activation
8. PARP1/2 activation
9. RAD51 foci (HR marker)
10. Cell cycle arrest signal
11. Apoptosis signal
12. Survival signal

See `docs/supplementary_methods.md` for the complete mathematical formulation.

---

## Troubleshooting

**Import errors:** Ensure you run scripts from the repository root or use `scripts/run_all.py`.

**Missing `tqdm`:** Install with `pip install tqdm` (optional progress bars in sensitivity analysis).

**R package errors:** Run `Rscript install.R` to install all required R packages.

**Font warnings in matplotlib:** Install Arial font or ignore the warning (fallback font is used).
