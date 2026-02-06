# Drug Library Reference

## Synthetic Lethality QSP Project
**ATM-deficient CLL Drug Repurposing via Quantitative Systems Pharmacology**

---

## Drug Portfolio Overview

- **9 single-agent drugs** across 5 DDR pathway targets
- **7 drug combinations** with synergy analysis

---

## Single-Agent Drug Library

| Drug | Target | IC50 (nM) | Therapeutic Index | Selectivity (IC50 Ratio) |
|------|--------|-----------|-------------------|--------------------------|
| AZD6738 | ATR | 52.7 | 4.48 | 6.0x |
| VE-822 | ATR | 58.1 | 4.48 | 6.0x |
| Prexasertib | CHK1 | 71.3 | -- | 5.0x |
| Olaparib | PARP | 89.4 | 2.36 | 4.2x |
| Talazoparib | PARP | 45.2 | -- | -- |
| Adavosertib | WEE1 | 124.6 | -- | 3.8x |
| KU-55933 | ATM | -- | -- | -- |
| Peposertib | NHEJ (DNA-PKcs) | -- | -- | -- |
| Dual ATR+PARP | ATR + PARP | -- | -- | -- |

---

## Drug Classification

### By Therapeutic Index (Safety)
```
Excellent (TI > 4):     AZD6738, VE-822
Good (TI 2-3):          Prexasertib, Olaparib, Talazoparib
Moderate (TI 1-2):      Adavosertib
```

### By Potency (EC50)
```
Very High (EC50 < 50):  Talazoparib (45.2 nM)
High (EC50 50-70):      AZD6738 (52.7), VE-822 (58.1), Prexasertib (71.3)
Moderate (EC50 70-100): Olaparib (89.4)
Lower (EC50 > 100):     Adavosertib (124.6)
```

### By Selectivity (IC50 Ratio: WT / ATM-deficient)
```
Highly Selective (>5x): AZD6738 (6.0x), VE-822 (6.0x)
Selective (4-5x):       Prexasertib (5.0x), Olaparib (4.2x)
Moderate (3-4x):        Adavosertib (3.8x)
```

---

## Combination Synergy Ranking

| Rank | Combination | CI | Synergy Level |
|------|-------------|-----|---------------|
| 1 | ATR+PARP+WEE1 | 0.49 | Very Strong |
| 2 | All DDR | 0.52 | Strong |
| 3 | ATR+PARP+CHK1 | 0.55 | Strong |
| 4 | AZD6738+Olaparib | 0.68 | Strong |
| 5 | VE-822+Prexasertib | 0.71 | Moderate |
| 6 | AZD6738+Adavosertib | 0.74 | Moderate |
| 7 | Olaparib+Prexasertib | 0.79 | Moderate |

CI = Combination Index (CI < 1 indicates synergy).

---

## DDR Pathway Targets

### ATR Pathway
ATR kinase is the primary compensatory kinase in ATM-deficient cells. Inhibiting ATR creates synthetic lethality by blocking the remaining DNA damage checkpoint.

- **AZD6738 (Ceralasertib):** Selective ATR inhibitor; Phase II clinical trials
- **VE-822 (Berzosertib):** Potent ATR inhibitor; clinical development

### CHK1 Pathway
CHK1 is a key downstream effector of ATR. Inhibition disrupts S and G2/M checkpoints.

- **Prexasertib (LY2606368):** Selective CHK1 inhibitor; Phase I/II trials

### PARP Pathway
PARP inhibitors block single-strand break repair, converting SSBs to DSBs during replication.

- **Olaparib:** FDA-approved PARP inhibitor; first-in-class
- **Talazoparib:** Second-generation PARP inhibitor with PARP trapping

### WEE1 Pathway
WEE1 inhibition abrogates the G2/M checkpoint, forcing cells with DNA damage into mitosis.

- **Adavosertib (AZD1775):** Selective WEE1 inhibitor; clinical trials

### NHEJ Pathway
DNA-PKcs inhibition blocks non-homologous end joining repair.

- **Peposertib (M3814):** Selective DNA-PKcs inhibitor; evaluated post hoc

### ATM Pathway
Direct ATM inhibition serves as a pharmacological model of ATM deficiency.

- **KU-55933:** Research-grade ATM inhibitor; reference compound

---

## Data Sources

### Computational Models
- `src/enhanced_ddr_qsp_model.py` - 12-state ODE QSP model with drug library
- `src/cll_drug_repurposing_model.py` - Simplified DDR model
- `src/dose_response_modeling.py` - Hill equation fitting

### Data Files
- `data/screening/complete_screening_results.csv` - Full screening results
- `data/time_series/` - 16 time-course simulation files (8 drugs x 2 genotypes)
- `data/dose_response/` - Dose-response curves for 5 drugs
- `data/concentration_response/ic50_summary.csv` - IC50 values

---

## Top Candidates Summary

1. **AZD6738** (ATR inhibitor) - Best therapeutic index (4.48), high selectivity (6.0x)
2. **VE-822** (ATR inhibitor) - Equal TI to AZD6738, potent ATR inhibition
3. **Olaparib** (PARP inhibitor) - FDA-approved, good selectivity (4.2x)

**Recommended first-line:** AZD6738 or VE-822 monotherapy
**Recommended combination:** AZD6738 + Olaparib (CI: 0.68)
