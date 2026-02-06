"""
PHASE 2 TASK 4: CONCENTRATION-RESPONSE CURVES
Implement full dose-response analysis for all 9 drugs with IC50 calculations and therapeutic window identification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from enhanced_ddr_qsp_model import EnhancedDDRModel, enhanced_drug_library
from tqdm import tqdm

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path(__file__).resolve().parent.parent / "data" / "concentration_response"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 2 TASK 4: CONCENTRATION-RESPONSE CURVES")
print("=" * 80)

# ============================================================================
# DEFINE CONCENTRATION RANGE
# ============================================================================

# Test concentrations (nM): 1, 3, 10, 30, 100, 300, 1000
concentrations = np.array([1, 3, 10, 30, 100, 300, 1000])

print(f"\nConcentrations to test: {concentrations} nM")
print(f"Number of drugs: {len(enhanced_drug_library)}")
print(f"Total simulations: {len(enhanced_drug_library) * len(concentrations) * 2} (WT + ATM-def)")

# ============================================================================
# HILL EQUATION FOR IC50 FITTING
# ============================================================================

def hill_equation(concentration, ic50, hill_coeff, emax):
    """
    Hill equation for dose-response curve fitting

    Args:
        concentration: Drug concentration (nM)
        ic50: Half-maximal inhibitory concentration (nM)
        hill_coeff: Hill coefficient (slope)
        emax: Maximum effect (apoptosis signal)

    Returns:
        Effect (apoptosis) at given concentration
    """
    return emax * (concentration ** hill_coeff) / (ic50 ** hill_coeff + concentration ** hill_coeff)

# ============================================================================
# RUN CONCENTRATION-RESPONSE ANALYSIS
# ============================================================================

print("\nRunning concentration-response analysis...")

all_results = []

for drug_name, drug_props in tqdm(enhanced_drug_library.items(), desc="Analyzing drugs"):
    drug_results = []

    for conc in concentrations:
        # Calculate drug effects at this concentration
        target = drug_props['target']
        ic50_values = drug_props['ic50_values']
        hill_coeff = drug_props.get('hill_coefficient', 1.0)

        # Calculate inhibition for each target
        drug_effects = {}
        for target_name, ic50 in ic50_values.items():
            effect = hill_equation(conc, ic50, hill_coeff, 1.0)
            drug_effects[target_name] = effect

        # Add cross-reactivity effects
        if 'cross_reactivity' in drug_props:
            for target_name, cross_ic50_multiplier in drug_props['cross_reactivity'].items():
                if target_name in ic50_values:
                    cross_ic50 = ic50_values[target_name] * cross_ic50_multiplier
                else:
                    # Estimate cross-reactivity IC50
                    primary_ic50 = list(ic50_values.values())[0]
                    cross_ic50 = primary_ic50 / cross_ic50_multiplier

                cross_effect = hill_equation(conc, cross_ic50, hill_coeff, 1.0)
                drug_effects[target_name] = cross_effect

        # Run simulations for WT and ATM-def
        model_wt = EnhancedDDRModel(atm_proficient=True)
        model_atm_def = EnhancedDDRModel(atm_proficient=False)

        results_wt = model_wt.run_simulation(duration=48, drug_effects=drug_effects)
        results_atm_def = model_atm_def.run_simulation(duration=48, drug_effects=drug_effects)

        # Extract final apoptosis
        apoptosis_wt = results_wt['ApoptosisSignal'].iloc[-1]
        apoptosis_atm_def = results_atm_def['ApoptosisSignal'].iloc[-1]
        sl_score = apoptosis_atm_def / apoptosis_wt if apoptosis_wt > 0.01 else 0

        drug_results.append({
            'Drug': drug_name,
            'Target': target,
            'Concentration_nM': conc,
            'Apoptosis_WT': apoptosis_wt,
            'Apoptosis_ATM_def': apoptosis_atm_def,
            'SL_Score': sl_score
        })

    all_results.extend(drug_results)

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

# Save results
df_results.to_csv(output_dir / "concentration_response_data.csv", index=False)
print(f"\n✓ Saved concentration-response data to: {output_dir}/concentration_response_data.csv")

# ============================================================================
# FIT IC50 VALUES
# ============================================================================

print("\nFitting IC50 values...")

ic50_summary = []

for drug_name in df_results['Drug'].unique():
    drug_data = df_results[df_results['Drug'] == drug_name]

    # Fit WT data
    try:
        popt_wt, _ = curve_fit(
            hill_equation,
            drug_data['Concentration_nM'],
            drug_data['Apoptosis_WT'],
            p0=[100, 1.0, 1.0],  # Initial guess: IC50=100, hill=1, emax=1
            bounds=([1, 0.1, 0], [10000, 5, 1]),  # Bounds
            maxfev=10000
        )
        ic50_wt, hill_wt, emax_wt = popt_wt
    except:
        ic50_wt, hill_wt, emax_wt = np.nan, np.nan, np.nan

    # Fit ATM-def data
    try:
        popt_atm_def, _ = curve_fit(
            hill_equation,
            drug_data['Concentration_nM'],
            drug_data['Apoptosis_ATM_def'],
            p0=[100, 1.0, 1.0],
            bounds=([1, 0.1, 0], [10000, 5, 1]),
            maxfev=10000
        )
        ic50_atm_def, hill_atm_def, emax_atm_def = popt_atm_def
    except:
        ic50_atm_def, hill_atm_def, emax_atm_def = np.nan, np.nan, np.nan

    # Calculate IC50 ratio (measure of synthetic lethality)
    ic50_ratio = ic50_wt / ic50_atm_def if not np.isnan(ic50_wt) and not np.isnan(ic50_atm_def) and ic50_atm_def > 0 else np.nan

    # Identify therapeutic window (concentration range with SL > 2 and WT apoptosis < 0.3)
    therapeutic_window = drug_data[(drug_data['SL_Score'] > 2) & (drug_data['Apoptosis_WT'] < 0.3)]

    if len(therapeutic_window) > 0:
        tw_min = therapeutic_window['Concentration_nM'].min()
        tw_max = therapeutic_window['Concentration_nM'].max()
        tw_range = f"{tw_min:.0f}-{tw_max:.0f} nM"
    else:
        tw_range = "None"

    ic50_summary.append({
        'Drug': drug_name,
        'Target': drug_data['Target'].iloc[0],
        'IC50_WT_nM': ic50_wt,
        'IC50_ATM_def_nM': ic50_atm_def,
        'IC50_Ratio': ic50_ratio,
        'Hill_WT': hill_wt,
        'Hill_ATM_def': hill_atm_def,
        'Emax_WT': emax_wt,
        'Emax_ATM_def': emax_atm_def,
        'Therapeutic_Window': tw_range
    })

df_ic50 = pd.DataFrame(ic50_summary)
df_ic50 = df_ic50.sort_values('IC50_Ratio', ascending=False)

# Save IC50 summary
df_ic50.to_csv(output_dir / "ic50_summary.csv", index=False)
print(f"✓ Saved IC50 summary to: {output_dir}/ic50_summary.csv")

print("\nIC50 Summary (Top 5 by IC50 Ratio):")
print(df_ic50.head(5)[['Drug', 'IC50_WT_nM', 'IC50_ATM_def_nM', 'IC50_Ratio', 'Therapeutic_Window']].to_string(index=False))

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

# 1. Individual dose-response curves for each drug
for drug_name in tqdm(df_results['Drug'].unique(), desc="  Creating dose-response plots"):
    drug_data = df_results[df_results['Drug'] == drug_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Apoptosis vs Concentration
    ax = axes[0]
    ax.plot(drug_data['Concentration_nM'], drug_data['Apoptosis_WT'],
            'o-', color='steelblue', linewidth=2, markersize=8, label='WT')
    ax.plot(drug_data['Concentration_nM'], drug_data['Apoptosis_ATM_def'],
            's--', color='firebrick', linewidth=2, markersize=8, label='ATM-deficient')

    # Add fitted curves if IC50 was successfully calculated
    drug_ic50 = df_ic50[df_ic50['Drug'] == drug_name].iloc[0]
    if not np.isnan(drug_ic50['IC50_WT_nM']):
        conc_smooth = np.logspace(0, 3, 100)
        apoptosis_fit_wt = hill_equation(conc_smooth, drug_ic50['IC50_WT_nM'],
                                         drug_ic50['Hill_WT'], drug_ic50['Emax_WT'])
        ax.plot(conc_smooth, apoptosis_fit_wt, '-', color='steelblue', alpha=0.3, linewidth=1)

    if not np.isnan(drug_ic50['IC50_ATM_def_nM']):
        conc_smooth = np.logspace(0, 3, 100)
        apoptosis_fit_atm_def = hill_equation(conc_smooth, drug_ic50['IC50_ATM_def_nM'],
                                              drug_ic50['Hill_ATM_def'], drug_ic50['Emax_ATM_def'])
        ax.plot(conc_smooth, apoptosis_fit_atm_def, '--', color='firebrick', alpha=0.3, linewidth=1)

    ax.set_xscale('log')
    ax.set_xlabel('Concentration (nM)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Apoptosis Signal (48h)', fontsize=11, fontweight='bold')
    ax.set_title(f'{drug_name}\nDose-Response Curve', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Plot 2: SL Score vs Concentration
    ax = axes[1]
    ax.plot(drug_data['Concentration_nM'], drug_data['SL_Score'],
            'o-', color='mediumseagreen', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Concentration (nM)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Synthetic Lethality Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{drug_name}\nSL Score vs Concentration', fontsize=11, fontweight='bold')
    ax.axhspan(2, 5, alpha=0.2, color='green', label='Target SL range (2-5×)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add IC50 annotations
    if not np.isnan(drug_ic50['IC50_WT_nM']):
        ax.text(0.05, 0.95, f"IC50 (WT): {drug_ic50['IC50_WT_nM']:.1f} nM\n"
                            f"IC50 (ATM-def): {drug_ic50['IC50_ATM_def_nM']:.1f} nM\n"
                            f"IC50 Ratio: {drug_ic50['IC50_Ratio']:.2f}×\n"
                            f"Therapeutic Window: {drug_ic50['Therapeutic_Window']}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save with safe filename
    safe_filename = drug_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
    plt.savefig(output_dir / f"{safe_filename}_dose_response.png", bbox_inches='tight')
    plt.close()

print(f"  ✓ Saved {len(df_results['Drug'].unique())} dose-response plots")

# 2. Summary plot: All drugs on one figure
n_drugs = len(df_results['Drug'].unique())
n_cols = 3
n_rows = math.ceil(n_drugs / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
fig.suptitle('Concentration-Response Curves: All Drugs', fontsize=14, fontweight='bold')

if n_rows == 1:
    axes = np.array([axes])
if n_cols == 1:
    axes = axes[:, np.newaxis]

for idx, drug_name in enumerate(df_results['Drug'].unique()):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    drug_data = df_results[df_results['Drug'] == drug_name]

    ax.plot(drug_data['Concentration_nM'], drug_data['Apoptosis_WT'],
            'o-', color='steelblue', linewidth=1.5, markersize=6, label='WT')
    ax.plot(drug_data['Concentration_nM'], drug_data['Apoptosis_ATM_def'],
            's--', color='firebrick', linewidth=1.5, markersize=6, label='ATM-def')

    ax.set_xscale('log')
    ax.set_xlabel('Concentration (nM)', fontsize=9)
    ax.set_ylabel('Apoptosis Signal', fontsize=9)
    ax.set_title(drug_name.replace(' (', '\n('), fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for idx in range(n_drugs, n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "all_drugs_dose_response_summary.png", bbox_inches='tight')
print(f"  ✓ Saved: all_drugs_dose_response_summary.png")
plt.close()

# 3. IC50 ratio bar chart
fig, ax = plt.subplots(figsize=(12, 6))

df_ic50_sorted = df_ic50.sort_values('IC50_Ratio', ascending=False)
x = np.arange(len(df_ic50_sorted))

colors = ['green' if 2 <= ratio <= 10 else 'orange' if ratio > 10 else 'red'
          for ratio in df_ic50_sorted['IC50_Ratio']]

bars = ax.bar(x, df_ic50_sorted['IC50_Ratio'], color=colors, alpha=0.8,
              edgecolor='black', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(df_ic50_sorted['Drug'].str.replace(' (', '\n('), rotation=45, ha='right', fontsize=9)
ax.set_ylabel('IC50 Ratio (WT / ATM-def)', fontsize=11, fontweight='bold')
ax.set_title('IC50 Ratio: Measure of Synthetic Lethality\n(Higher = More Selective for ATM-deficient Cells)',
             fontsize=12, fontweight='bold')
ax.axhspan(2, 10, alpha=0.2, color='green', label='Good selectivity (2-10×)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df_ic50_sorted['IC50_Ratio'])):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "ic50_ratio_comparison.png", bbox_inches='tight')
print(f"  ✓ Saved: ic50_ratio_comparison.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("CONCENTRATION-RESPONSE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAll results saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. concentration_response_data.csv - Full dose-response data")
print("  2. ic50_summary.csv - IC50 values and therapeutic windows")
print("  3. [drug]_dose_response.png - Individual dose-response plots (9 files)")
print("  4. all_drugs_dose_response_summary.png - Summary plot")
print("  5. ic50_ratio_comparison.png - IC50 ratio bar chart")
print("\n" + "=" * 80)
