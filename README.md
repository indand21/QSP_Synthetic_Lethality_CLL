# Quantitative Systems Pharmacology Model for Synthetic Lethality Drug Screening in ATM-deficient CLL

A computational framework for identifying synthetic lethal drug candidates in ATM-deficient chronic lymphocytic leukemia (CLL) using a 12-state ODE model of the DNA Damage Response (DDR) pathway.

## Overview

This repository contains the code, data, and analysis pipeline described in:

> **Quantitative Systems Pharmacology Modeling of DNA Damage Response Inhibitor Synthetic Lethality in ATM-Deficient Chronic Lymphocytic Leukemia**
>
> *Citation details to be added upon publication*

ATM kinase is frequently mutated in CLL (~10-15% of cases) and is associated with poor prognosis and resistance to standard therapies. We developed a mechanistic QSP model that simulates DDR pathway dynamics under pharmacological perturbation to identify drugs and combinations that selectively kill ATM-deficient cells while sparing normal (ATM-proficient) cells -- a synthetic lethality strategy.

### Key Results

- **ATR inhibitors** (AZD6738, VE-822) showed the highest therapeutic indices (4.48x) and selectivity (6.0x IC50 ratio)
- **PARP inhibitors** (Olaparib, Talazoparib) demonstrated strong synergy with ATR inhibitors (CI = 0.68)
- Parameter sensitivity analysis identified apoptosis rate constants and ATR activation as critical model parameters

## Repository Structure

```
github_QSP_SL/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore
├── requirements.txt             # Python dependencies
├── install.R                    # R package installation
│
├── src/                         # Core Python modules (15 files)
│   ├── enhanced_ddr_qsp_model.py          # 12-state ODE QSP model
│   ├── cll_drug_repurposing_model.py      # Simplified DDR model
│   ├── dose_response_modeling.py          # Hill equation, IC50, Emax
│   ├── pharmacokinetic_modeling.py        # PK modeling
│   ├── drug_concentration_simulator.py    # PK-PD integration
│   ├── statistical_testing_correction.py  # FDR/Bonferroni corrections
│   ├── cross_validation_framework.py      # K-fold, LOOCV, bootstrap CV
│   ├── gdsc_validation_framework.py       # GDSC database validation
│   ├── concentration_dose_response_validation.py
│   ├── integrated_statistical_pipeline.py # Unified statistics pipeline
│   ├── model_validation_framework.py      # Validation metrics
│   ├── model_validation_comparison.py     # Model comparison
│   ├── enhanced_visualization_framework.py
│   ├── peposertib_evaluation.py           # NHEJ inhibitor evaluation
│   └── run_enhanced_screening.py          # Main screening entry point
│
├── R/                           # R statistical scripts (5 files)
│   ├── statistical_analysis_framework.R
│   ├── publication_visualization_framework.R
│   ├── statistical_correction_integration.R
│   ├── python_integration_pipeline.R
│   └── reproducible_workflows.R
│
├── scripts/                     # Pipeline orchestration scripts
│   ├── run_all.py               # Master reproduction script
│   ├── run_phase2_optimization.py
│   ├── run_sensitivity_analysis.py
│   ├── run_dose_response.py
│   └── generate_figures.py
│
├── data/
│   ├── time_series/             # 16 CSVs (8 drugs x 2 genotypes)
│   ├── dose_response/           # 5 drug dose-response CSVs
│   ├── screening/               # Complete screening results
│   ├── parameters/              # Optimized model parameters (JSON)
│   ├── concentration_response/  # IC50 summary, concentration-response data
│   ├── sensitivity_analysis/    # Parameter sensitivity results
│   └── statistical_analysis/    # Summary statistics (JSON)
│
├── figures/                     # Manuscript figures
│   ├── Figure_1.pdf
│   ├── Figure_2.png
│   ├── Figure_3.png
│   ├── Supplementary_Figure_1.png
│   └── Supplementary_Figure_2.png
│
└── docs/
    ├── supplementary_methods.md # Complete ODE equations
    ├── drug_inventory.md        # Drug library reference
    └── workflow.md              # Reproduction workflow guide
```

## Quick Start

### Prerequisites

- Python 3.9 or later
- R 4.2 or later (for statistical analysis)

### Installation

```bash
git clone https://github.com/<username>/github_QSP_SL.git
cd github_QSP_SL

# Python dependencies
pip install -r requirements.txt

# R packages (optional, for statistical analysis)
Rscript install.R
```

### Run the Full Pipeline

```bash
python scripts/run_all.py
```

This executes:
1. **Parameter optimization** -- finds optimal apoptosis rate constants
2. **Sensitivity analysis** -- identifies critical model parameters
3. **Concentration-response curves** -- computes IC50 values for all drugs
4. **Figure generation** -- produces publication-quality plots

Pre-computed data from Phase 1 (screening, time-series) is included in `data/`.

## Model Description

The QSP model consists of **12 coupled ordinary differential equations** representing the temporal dynamics of DDR pathway components:

| State | Variable | Description |
|-------|----------|-------------|
| 1 | DSB(t) | DNA double-strand breaks |
| 2 | ATM_active(t) | Activated ATM kinase |
| 3 | ATR_active(t) | Activated ATR kinase |
| 4 | CHK1_active(t) | Activated CHK1 kinase |
| 5 | CHK2_active(t) | Activated CHK2 kinase |
| 6 | p53_active(t) | Activated p53 tumor suppressor |
| 7 | p21_active(t) | Activated p21 CDK inhibitor |
| 8 | PARP_active(t) | Activated PARP1/2 |
| 9 | RAD51_focus(t) | RAD51 foci (HR activity marker) |
| 10 | CellCycleArrest(t) | Cell cycle arrest signal |
| 11 | Apoptosis(t) | Apoptosis signal |
| 12 | Survival(t) | Survival signal |

ATM deficiency is modeled by reducing the ATM activation rate constant from 1.50 h^-1 (WT) to 0.45 h^-1 (ATM-deficient), with compensatory upregulation of ATR signaling. Drug effects are computed using Hill equations with experimentally derived IC50 values.

See `docs/supplementary_methods.md` for the complete mathematical formulation.

## Drug Library

| Drug | Target | Mechanism |
|------|--------|-----------|
| AZD6738 (Ceralasertib) | ATR | Selective ATR kinase inhibitor |
| VE-822 (Berzosertib) | ATR | Potent ATR kinase inhibitor |
| Prexasertib | CHK1 | Selective CHK1 kinase inhibitor |
| Olaparib | PARP | PARP1/2 inhibitor (FDA-approved) |
| Talazoparib | PARP | PARP inhibitor with PARP trapping |
| Adavosertib | WEE1 | Selective WEE1 kinase inhibitor |
| KU-55933 | ATM | ATM kinase inhibitor (reference) |
| Peposertib | DNA-PKcs (NHEJ) | DNA-PKcs inhibitor |
| Dual ATR+PARP | ATR + PARP | Combination regimen |

## Module Descriptions

### Core Model (`src/`)

| Module | Description |
|--------|-------------|
| `enhanced_ddr_qsp_model.py` | 12-state ODE system, drug library, simulation engine |
| `cll_drug_repurposing_model.py` | Simplified 6-state DDR model for rapid screening |
| `dose_response_modeling.py` | Hill equation fitting, IC50/EC50 estimation, Emax models |
| `pharmacokinetic_modeling.py` | One- and two-compartment PK models |
| `drug_concentration_simulator.py` | PK-PD integration and time-varying drug exposure |
| `statistical_testing_correction.py` | Benjamini-Hochberg FDR, Bonferroni corrections |
| `cross_validation_framework.py` | K-fold, leave-one-out, and bootstrap cross-validation |
| `gdsc_validation_framework.py` | Validation against Genomics of Drug Sensitivity in Cancer |
| `concentration_dose_response_validation.py` | Dose-response curve validation |
| `integrated_statistical_pipeline.py` | Unified statistical analysis pipeline |
| `model_validation_framework.py` | Model accuracy and validation metrics |
| `model_validation_comparison.py` | Comparison between model variants |
| `enhanced_visualization_framework.py` | Publication-quality visualization utilities |
| `peposertib_evaluation.py` | Post-hoc evaluation of NHEJ inhibitor |
| `run_enhanced_screening.py` | Main entry point for virtual drug screening |

### R Analysis (`R/`)

| Script | Description |
|--------|-------------|
| `statistical_analysis_framework.R` | Comprehensive statistical testing in R |
| `publication_visualization_framework.R` | ggplot2-based publication figures |
| `statistical_correction_integration.R` | Multiple testing correction integration |
| `python_integration_pipeline.R` | Bridge between Python outputs and R analysis |
| `reproducible_workflows.R` | Reproducibility and workflow utilities |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data in your research, please cite:

```
@article{synthetic_lethality_qsp_2025,
  title={Quantitative Systems Pharmacology Modeling of DNA Damage Response Inhibitor
         Synthetic Lethality in ATM-Deficient Chronic Lymphocytic Leukemia},
  author={...},
  journal={Clinical Pharmacology \& Therapeutics},
  year={2025}
}
```

*Citation to be updated upon publication.*
