"""
Phase 2: Parameter Optimization for Realistic Synthetic Lethality Scores

This script optimizes drug concentrations and apoptosis parameters to achieve
realistic SL scores (2-5×) while maintaining biological validity.

Strategy:
1. Test multiple drug concentrations (10-200 nM, 0.4-8× IC50)
2. Reduce apoptosis rate constants by 50% to prevent saturation
3. Generate dose-response curves for each drug
4. Identify optimal concentrations for demonstrating SL
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from scipy.optimize import minimize
import logging

from enhanced_ddr_qsp_model import (
    EnhancedDDRModel,
    calculate_drug_effects,
    enhanced_drug_library
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directories
_base = Path(__file__).resolve().parent.parent / "data"
output_dir = _base
output_dir.mkdir(parents=True, exist_ok=True)
(_base / "dose_response").mkdir(parents=True, exist_ok=True)
(_base / "parameters").mkdir(parents=True, exist_ok=True)
(_base / "screening").mkdir(parents=True, exist_ok=True)

class ParameterOptimizer:
    """Optimize model parameters for realistic synthetic lethality"""

    def __init__(self):
        self.concentration_range = [10, 25, 50, 75, 100, 150, 200]  # nM
        self.target_sl_range = (2.0, 5.0)  # Target SL scores
        self.max_wt_apoptosis = 0.3  # Maximum acceptable WT apoptosis

    def test_apoptosis_reduction(self, reduction_factor: float = 0.5) -> Dict:
        """
        Test effect of reducing apoptosis rate constants

        Args:
            reduction_factor: Factor to reduce apoptosis rates (0.5 = 50% reduction)

        Returns:
            Dictionary with results for each drug
        """
        logger.info(f"Testing apoptosis reduction factor: {reduction_factor}")

        results = {}

        for drug_name in enhanced_drug_library.keys():
            # Create models with reduced apoptosis rates
            model_wt = EnhancedDDRModel(atm_proficient=True, cell_cycle_phase='S')
            model_def = EnhancedDDRModel(atm_proficient=False, cell_cycle_phase='S')

            # Reduce apoptosis rate constants
            model_wt.params['k_apoptosis_p53'] *= reduction_factor
            model_wt.params['k_apoptosis_damage'] *= reduction_factor
            model_def.params['k_apoptosis_p53'] *= reduction_factor
            model_def.params['k_apoptosis_damage'] *= reduction_factor

            # Calculate drug effects at standard concentration
            drug_effects = calculate_drug_effects(drug_name, enhanced_drug_library)

            # Run simulations
            sim_wt = model_wt.run_simulation(48, drug_effects)
            sim_def = model_def.run_simulation(48, drug_effects)

            apoptosis_wt = sim_wt['ApoptosisSignal'].iloc[-1]
            apoptosis_def = sim_def['ApoptosisSignal'].iloc[-1]
            sl_score = apoptosis_def / (apoptosis_wt + 1e-9)

            results[drug_name] = {
                'apoptosis_wt': apoptosis_wt,
                'apoptosis_def': apoptosis_def,
                'sl_score': sl_score,
                'reduction_factor': reduction_factor
            }

            logger.info(f"{drug_name}: WT={apoptosis_wt:.3f}, ATM-def={apoptosis_def:.3f}, SL={sl_score:.2f}×")

        return results

    def generate_dose_response_curve(self, drug_name: str,
                                     concentrations: List[float],
                                     reduction_factor: float = 0.5) -> pd.DataFrame:
        """
        Generate dose-response curve for a single drug

        Args:
            drug_name: Name of drug to test
            concentrations: List of concentrations to test (nM)
            reduction_factor: Apoptosis rate reduction factor

        Returns:
            DataFrame with dose-response data
        """
        logger.info(f"Generating dose-response curve for {drug_name}")

        results = []

        for conc in concentrations:
            # Create temporary drug library with this concentration
            temp_library = enhanced_drug_library.copy()
            temp_library[drug_name]['concentration'] = conc

            # Create models with reduced apoptosis rates
            model_wt = EnhancedDDRModel(atm_proficient=True, cell_cycle_phase='S')
            model_def = EnhancedDDRModel(atm_proficient=False, cell_cycle_phase='S')

            model_wt.params['k_apoptosis_p53'] *= reduction_factor
            model_wt.params['k_apoptosis_damage'] *= reduction_factor
            model_def.params['k_apoptosis_p53'] *= reduction_factor
            model_def.params['k_apoptosis_damage'] *= reduction_factor

            # Calculate drug effects
            drug_effects = calculate_drug_effects(drug_name, temp_library)

            # Run simulations
            sim_wt = model_wt.run_simulation(48, drug_effects)
            sim_def = model_def.run_simulation(48, drug_effects)

            apoptosis_wt = sim_wt['ApoptosisSignal'].iloc[-1]
            apoptosis_def = sim_def['ApoptosisSignal'].iloc[-1]
            sl_score = apoptosis_def / (apoptosis_wt + 1e-9)
            ti = apoptosis_def / (apoptosis_wt + 1e-9)

            # Get pathway metrics
            pathway_wt = model_wt.get_pathway_activity(sim_wt)
            pathway_def = model_def.get_pathway_activity(sim_def)

            results.append({
                'Drug': drug_name,
                'Concentration_nM': conc,
                'Apoptosis_WT': apoptosis_wt,
                'Apoptosis_ATM_def': apoptosis_def,
                'SL_Score': sl_score,
                'Therapeutic_Index': ti,
                'ATR_Activity_WT': pathway_wt['atr_activity'],
                'ATR_Activity_ATM_def': pathway_def['atr_activity'],
                'p53_Pathway_WT': pathway_wt['p53_pathway'],
                'p53_Pathway_ATM_def': pathway_def['p53_pathway']
            })

        return pd.DataFrame(results)

    def find_optimal_concentration(self, dose_response_df: pd.DataFrame) -> Dict:
        """
        Find optimal concentration that maximizes SL while keeping WT apoptosis low

        Args:
            dose_response_df: DataFrame with dose-response data

        Returns:
            Dictionary with optimal concentration and metrics
        """
        # Filter for acceptable WT apoptosis
        acceptable = dose_response_df[dose_response_df['Apoptosis_WT'] <= self.max_wt_apoptosis]

        if len(acceptable) == 0:
            logger.warning("No concentrations meet WT apoptosis criteria")
            # Use lowest concentration
            optimal = dose_response_df.iloc[0]
        else:
            # Find concentration with highest SL score
            optimal = acceptable.loc[acceptable['SL_Score'].idxmax()]

        return {
            'drug': optimal['Drug'],
            'optimal_concentration_nM': optimal['Concentration_nM'],
            'sl_score': optimal['SL_Score'],
            'apoptosis_wt': optimal['Apoptosis_WT'],
            'apoptosis_atm_def': optimal['Apoptosis_ATM_def'],
            'therapeutic_index': optimal['Therapeutic_Index']
        }

    def optimize_all_drugs(self, reduction_factors: List[float] = [0.3, 0.4, 0.5, 0.6]) -> Dict:
        """
        Test multiple reduction factors and find optimal for each drug

        Args:
            reduction_factors: List of apoptosis reduction factors to test

        Returns:
            Dictionary with optimal parameters for each drug
        """
        logger.info("Optimizing parameters for all drugs...")

        all_results = {}
        best_reduction_factor = None
        best_avg_sl = 0

        for rf in reduction_factors:
            logger.info(f"\nTesting reduction factor: {rf}")

            drug_results = {}
            sl_scores = []

            for drug_name in list(enhanced_drug_library.keys())[:5]:  # Test top 5 drugs
                # Generate dose-response curve
                dose_response = self.generate_dose_response_curve(
                    drug_name,
                    self.concentration_range,
                    reduction_factor=rf
                )

                # Find optimal concentration
                optimal = self.find_optimal_concentration(dose_response)

                drug_results[drug_name] = {
                    'dose_response': dose_response,
                    'optimal': optimal
                }

                sl_scores.append(optimal['sl_score'])

                logger.info(f"  {drug_name}: Optimal conc={optimal['optimal_concentration_nM']} nM, SL={optimal['sl_score']:.2f}×")

            avg_sl = np.mean(sl_scores)
            logger.info(f"Average SL score for rf={rf}: {avg_sl:.2f}×")

            all_results[rf] = drug_results

            # Track best reduction factor
            if avg_sl > best_avg_sl and self.target_sl_range[0] <= avg_sl <= self.target_sl_range[1]:
                best_avg_sl = avg_sl
                best_reduction_factor = rf

        if best_reduction_factor is None:
            # If no factor in target range, choose closest
            best_reduction_factor = reduction_factors[len(reduction_factors)//2]
            logger.warning(f"No reduction factor achieved target SL range, using {best_reduction_factor}")

        logger.info(f"\nBest reduction factor: {best_reduction_factor} (avg SL: {best_avg_sl:.2f}×)")

        return {
            'best_reduction_factor': best_reduction_factor,
            'all_results': all_results,
            'best_results': all_results[best_reduction_factor]
        }


def main():
    """Main execution function"""
    print("=" * 80)
    print("PHASE 2: PARAMETER OPTIMIZATION FOR REALISTIC SYNTHETIC LETHALITY")
    print("=" * 80)

    optimizer = ParameterOptimizer()

    # Step 1: Test different apoptosis reduction factors
    print("\n[STEP 1] Testing apoptosis reduction factors...")
    print("-" * 80)

    test_results = {}
    for rf in [0.3, 0.4, 0.5, 0.6]:
        test_results[rf] = optimizer.test_apoptosis_reduction(rf)

    # Step 2: Optimize all drugs
    print("\n[STEP 2] Optimizing parameters for all drugs...")
    print("-" * 80)

    optimization_results = optimizer.optimize_all_drugs()

    # Step 3: Save results
    print("\n[STEP 3] Saving optimization results...")
    print("-" * 80)

    # Save optimal parameters
    optimal_params = {
        'reduction_factor': optimization_results['best_reduction_factor'],
        'k_apoptosis_p53_multiplier': optimization_results['best_reduction_factor'],
        'k_apoptosis_damage_multiplier': optimization_results['best_reduction_factor']
    }

    with open(_base / "parameters" / "optimal_apoptosis_parameters.json", 'w') as f:
        json.dump(optimal_params, f, indent=2)

    logger.info(f"Saved optimal parameters to optimal_apoptosis_parameters.json")

    # Save dose-response data for each drug
    best_rf = optimization_results['best_reduction_factor']
    for drug_name, drug_data in optimization_results['best_results'].items():
        dose_response_df = drug_data['dose_response']
        filename = drug_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        dose_response_df.to_csv(
            _base / "dose_response" / f"{filename}_dose_response.csv",
            index=False
        )
        logger.info(f"Saved dose-response data for {drug_name}")

    # Create summary report
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Best reduction factor: {best_rf}")
    print(f"Optimal parameters:")
    print(f"  k_apoptosis_p53 multiplier: {best_rf}")
    print(f"  k_apoptosis_damage multiplier: {best_rf}")
    print("\nOptimal concentrations for each drug:")
    print("-" * 80)

    for drug_name, drug_data in optimization_results['best_results'].items():
        opt = drug_data['optimal']
        print(f"{drug_name}:")
        print(f"  Concentration: {opt['optimal_concentration_nM']} nM")
        print(f"  SL Score: {opt['sl_score']:.2f}×")
        print(f"  WT Apoptosis: {opt['apoptosis_wt']:.3f}")
        print(f"  ATM-def Apoptosis: {opt['apoptosis_atm_def']:.3f}")
        print()

    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    return optimization_results


if __name__ == '__main__':
    results = main()
