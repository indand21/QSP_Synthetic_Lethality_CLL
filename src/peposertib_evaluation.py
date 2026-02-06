"""
Evaluation script for Peposertib (NHEJ inhibitor) as single agent and in combination with ATR inhibitors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add the current directory to the path to import the model
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_ddr_qsp_model import (
    EnhancedDDRModel,
    calculate_drug_effects,
    enhanced_drug_library
)

def run_peposertib_evaluation():
    """
    Evaluate Peposertib as single agent and in combination with ATR inhibitors
    """
    print("=" * 80)
    print("PEPOSERTIB (NHEJ INHIBITOR) EVALUATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Drugs to evaluate
    drugs_to_test = [
        'Peposertib (NHEJ inhibitor)',
        'Peposertib + AZD6738 (NHEJ+ATR)',
        'Peposertib + VE-822 (NHEJ+ATR)',
        'AZD6738 (ATR inhibitor)',  # For comparison
        'VE-822 (ATR inhibitor)'   # For comparison
    ]
    
    print("Running simulations...")
    print("-" * 80)
    
    for drug_name in drugs_to_test:
        print(f"\nEvaluating: {drug_name}")
        
        try:
            # Calculate drug effects
            drug_effects = calculate_drug_effects(drug_name, enhanced_drug_library)
            print(f"  Drug effects: {drug_effects}")
            
            # Simulate in ATM-proficient cells
            model_wt = EnhancedDDRModel(atm_proficient=True, cell_cycle_phase='S')
            sim_wt = model_wt.run_simulation(48, drug_effects)
            apoptosis_wt = max(0, sim_wt['ApoptosisSignal'].iloc[-1])
            
            # Simulate in ATM-deficient cells
            model_atm_def = EnhancedDDRModel(atm_proficient=False, cell_cycle_phase='S')
            sim_atm_def = model_atm_def.run_simulation(48, drug_effects)
            apoptosis_atm_def = max(0, sim_atm_def['ApoptosisSignal'].iloc[-1])
            
            # Calculate synthetic lethality metrics
            sl_score = apoptosis_atm_def / (apoptosis_wt + 1e-9)
            therapeutic_index = apoptosis_atm_def / (apoptosis_wt + 1e-9)
            
            # Get pathway activity
            pathway_metrics = model_atm_def.get_pathway_activity(sim_atm_def)
            pathway_metrics_wt = model_wt.get_pathway_activity(sim_wt)
            
            result = {
                'Drug': drug_name,
                'Apoptosis_WT': apoptosis_wt,
                'Apoptosis_ATM_def': apoptosis_atm_def,
                'Synthetic_Lethality_Score': sl_score,
                'Therapeutic_Index': therapeutic_index,
                'DSB_ATM_def': pathway_metrics['dsb_level'],
                'DSB_WT': pathway_metrics_wt['dsb_level'],
                'HR_Activity_ATM_def': pathway_metrics['hr_activity'],
                'ATR_Activity_ATM_def': pathway_metrics['atr_activity'],
                'PARP_Activity_ATM_def': pathway_metrics['parp_activity']
            }
            
            results.append(result)
            
            print(f"  Apoptosis (WT): {apoptosis_wt:.4f}")
            print(f"  Apoptosis (ATM-def): {apoptosis_atm_def:.4f}")
            print(f"  Synthetic Lethality Score: {sl_score:.2f}")
            print(f"  Therapeutic Index: {therapeutic_index:.2f}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path('screening_data')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'peposertib_evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}\n")
    
    # Print summary table
    print("\nSUMMARY TABLE")
    print("=" * 80)
    print(results_df[['Drug', 'Apoptosis_WT', 'Apoptosis_ATM_def', 
                      'Synthetic_Lethality_Score', 'Therapeutic_Index']].to_string(index=False))
    print()
    
    # Generate comparison plot
    create_comparison_plot(results_df)
    
    return results_df

def create_comparison_plot(results_df):
    """Create visualization comparing Peposertib effects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Peposertib (NHEJ Inhibitor) Evaluation: Single Agent vs Combinations', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Apoptosis comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax1.bar(x_pos - width/2, results_df['Apoptosis_WT'], width, 
            label='ATM-Proficient', alpha=0.8, color='lightblue')
    ax1.bar(x_pos + width/2, results_df['Apoptosis_ATM_def'], width,
            label='ATM-Deficient', alpha=0.8, color='coral')
    ax1.set_xlabel('Treatment')
    ax1.set_ylabel('Apoptosis Signal')
    ax1.set_title('Apoptosis: WT vs ATM-Deficient')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([d.replace(' (', '\n(') for d in results_df['Drug']], 
                        rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Synthetic Lethality Scores
    ax2 = axes[0, 1]
    colors = ['green' if sl > 2.0 else 'orange' if sl > 1.5 else 'red' 
              for sl in results_df['Synthetic_Lethality_Score']]
    ax2.barh(results_df['Drug'], results_df['Synthetic_Lethality_Score'], color=colors, alpha=0.7)
    ax2.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='SL Threshold (2.0)')
    ax2.set_xlabel('Synthetic Lethality Score')
    ax2.set_title('Synthetic Lethality Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DSB levels
    ax3 = axes[1, 0]
    ax3.bar(x_pos - width/2, results_df['DSB_WT'], width,
            label='ATM-Proficient', alpha=0.8, color='lightblue')
    ax3.bar(x_pos + width/2, results_df['DSB_ATM_def'], width,
            label='ATM-Deficient', alpha=0.8, color='coral')
    ax3.set_xlabel('Treatment')
    ax3.set_ylabel('DSB Level')
    ax3.set_title('DNA Double-Strand Breaks (DSB)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([d.replace(' (', '\n(') for d in results_df['Drug']], 
                        rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pathway activity comparison
    ax4 = axes[1, 1]
    pathway_data = {
        'HR Activity': results_df['HR_Activity_ATM_def'].values,
        'ATR Activity': results_df['ATR_Activity_ATM_def'].values,
        'PARP Activity': results_df['PARP_Activity_ATM_def'].values
    }
    x = np.arange(len(results_df))
    bottom = np.zeros(len(results_df))
    colors_pathway = ['skyblue', 'lightcoral', 'lightgreen']
    for i, (pathway, values) in enumerate(pathway_data.items()):
        ax4.bar(x, values, bottom=bottom, label=pathway, alpha=0.7, color=colors_pathway[i])
        bottom += values
    ax4.set_xlabel('Treatment')
    ax4.set_ylabel('Pathway Activity (ATM-Def)')
    ax4.set_title('Pathway Activity in ATM-Deficient Cells')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.replace(' (', '\n(') for d in results_df['Drug']], 
                        rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('screening_data')
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / 'peposertib_evaluation_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.close()

if __name__ == '__main__':
    results = run_peposertib_evaluation()
    print("\nEvaluation complete!")

