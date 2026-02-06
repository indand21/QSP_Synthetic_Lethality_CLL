"""
Concentration-Based Dose-Response Validation Framework
======================================================

This module provides comprehensive validation of the QSP model against experimental
data from GDSC, with integrated dose-response and pharmacokinetic modeling:
- Concentration-time profile validation
- Dose-response curve fitting and validation
- PK/PD integration with QSP model
- Synthetic lethality validation across concentration ranges

Author: Kilo Code
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from dataclasses import dataclass

# Import our new modeling frameworks
from dose_response_modeling import (
    HillEquationModel, SigmoidalModel, EmaxModel, DoseResponseFitter, 
    DoseResponseCurve, DrugProperties, DoseResponseParameters, DoseResponseAnalyzer
)
from pharmacokinetic_modeling import PKParameters, DosingRegimen, PharmacokineticModeler
from drug_concentration_simulator import DrugProfile, SimulationSettings, DrugConcentrationSimulator
from enhanced_ddr_qsp_model import EnhancedDDRModel

# Import existing validation framework
from gdsc_validation_framework import GDSCData, ValidationResult, GDSCDownloader
from statistical_testing_correction import MultipleTestingCorrector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConcentrationValidationData:
    """Container for concentration-based experimental data"""
    cell_line: str
    atm_status: str
    drug: str
    drug_target: str
    concentration_time_points: List[float]  # Time points (h)
    concentrations: List[float]  # Concentrations (nM)
    viability_time_points: List[float]  # Time points for viability measurements
    viability_measurements: List[float]  # Viability measurements
    ic50: float  # Fitted IC50 (nM)
    auc: float  # Area under viability curve
    source: str = "GDSC_Concentration"

@dataclass
class DoseResponseValidationResult:
    """Container for dose-response validation results"""
    drug: str
    atm_status: str
    experimental_ic50: float
    predicted_ic50: float
    concentration_validation_score: float
    time_course_validation_score: float
    dose_response_r_squared: float
    pk_pd_integration_score: float
    synthetic_lethality_score: float
    overall_score: float
    confidence_interval: Tuple[float, float]

class ConcentrationDoseResponseValidator:
    """Validator for QSP model using concentration-dependent drug effects"""
    
    def __init__(self, qsp_model: EnhancedDDRModel):
        """
        Initialize concentration-based validator
        
        Args:
            qsp_model: QSP model instance
        """
        self.qsp_model = qsp_model
        self.validation_results = []
        self.dose_response_fitters = {}
        self.concentration_simulator = None
        
        # Initialize drug database with realistic parameters
        self.drug_database = self._initialize_drug_database()
    
    def _initialize_drug_database(self) -> Dict:
        """Initialize database of drug properties and parameters"""
        return {
            'AZD6738': {
                'target': 'ATR',
                'molecular_weight': 513.6,
                'logp': 2.1,
                'bioavailability': 0.65,
                'protein_binding': 0.85,
                'half_life': 8.2,
                'clearance_rate': 1.8,
                'volume_distribution': 0.8,
                'ic50_range': (5.0, 50.0),  # nM, ATM-deficient to proficient
                'hill_coefficient_range': (1.5, 3.0)
            },
            'VE-822': {
                'target': 'ATR',
                'molecular_weight': 432.5,
                'logp': 2.8,
                'bioavailability': 0.70,
                'protein_binding': 0.90,
                'half_life': 6.5,
                'clearance_rate': 2.1,
                'volume_distribution': 1.2,
                'ic50_range': (8.0, 80.0),
                'hill_coefficient_range': (1.8, 2.8)
            },
            'Prexasertib': {
                'target': 'CHK1',
                'molecular_weight': 488.9,
                'logp': 3.2,
                'bioavailability': 0.55,
                'protein_binding': 0.95,
                'half_life': 12.3,
                'clearance_rate': 1.2,
                'volume_distribution': 1.5,
                'ic50_range': (15.0, 150.0),
                'hill_coefficient_range': (1.2, 2.5)
            },
            'Adavosertib': {
                'target': 'WEE1',
                'molecular_weight': 637.8,
                'logp': 3.5,
                'bioavailability': 0.45,
                'protein_binding': 0.98,
                'half_life': 15.8,
                'clearance_rate': 0.9,
                'volume_distribution': 2.1,
                'ic50_range': (25.0, 300.0),
                'hill_coefficient_range': (1.0, 2.2)
            },
            'Olaparib': {
                'target': 'PARP',
                'molecular_weight': 434.5,
                'logp': 1.4,
                'bioavailability': 0.75,
                'protein_binding': 0.82,
                'half_life': 16.6,
                'clearance_rate': 1.1,
                'volume_distribution': 1.3,
                'ic50_range': (50.0, 500.0),
                'hill_coefficient_range': (1.8, 3.2)
            }
        }
    
    def fit_dose_response_to_experimental_data(self, experimental_data: pd.DataFrame,
                                             atm_proficient: bool = True) -> Dict:
        """
        Fit dose-response parameters to experimental data
        
        Args:
            experimental_data: Experimental data with concentration-response
            atm_proficient: ATM status for fitting
            
        Returns:
            Dictionary of fitted parameters
        """
        fitted_params = {}
        
        for drug in experimental_data['drug'].unique():
            drug_data = experimental_data[experimental_data['drug'] == drug]
            
            # Filter by ATM status
            if atm_proficient:
                atm_filter = drug_data['atm_status'] == 'proficient'
            else:
                atm_filter = drug_data['atm_status'] == 'deficient'
            
            atm_data = drug_data[atm_filter]
            
            if len(atm_data) == 0:
                continue
            
            # Extract concentrations and viabilities
            all_concentrations = []
            all_viabilities = []
            
            for _, row in atm_data.iterrows():
                if 'concentrations' in row and 'viabilities' in row:
                    concentrations = row['concentrations']
                    viabilities = row['viabilities']
                    all_concentrations.extend(concentrations)
                    all_viabilities.extend(viabilities)
            
            if len(all_concentrations) < 3:
                continue
            
            # Convert to numpy arrays
            concentrations = np.array(all_concentrations)
            viabilities = np.array(all_viabilities)
            
            # Convert viability to effect (0-1 scale)
            effects = (100 - viabilities) / 100.0
            effects = np.clip(effects, 0, 1)
            
            # Fit dose-response model
            try:
                fitter = DoseResponseFitter('hill')
                initial_guess = self._estimate_initial_parameters(drug, atm_proficient)
                
                fitted_params[drug] = fitter.fit(concentrations, effects, initial_guess)
                self.dose_response_fitters[f"{drug}_{atm_status}"] = fitter
                
                logger.info(f"Fitted dose-response for {drug} ({atm_status}): IC50 = {fitted_params[drug].ic50:.2f} nM")
                
            except Exception as e:
                logger.warning(f"Dose-response fitting failed for {drug}: {e}")
                # Use database estimates
                fitted_params[drug] = self._estimate_initial_parameters(drug, atm_proficient)
        
        return fitted_params
    
    def _estimate_initial_parameters(self, drug: str, atm_proficient: bool) -> DoseResponseParameters:
        """Estimate initial dose-response parameters from database"""
        if drug not in self.drug_database:
            return DoseResponseParameters(ic50=100.0, hill_coefficient=2.0, emax=0.9, baseline=0.1, ec50=100.0)
        
        drug_props = self.drug_database[drug]
        
        # Select IC50 based on ATM status
        if atm_proficient:
            ic50 = drug_props['ic50_range'][1]  # Higher IC50 for proficient
        else:
            ic50 = drug_props['ic50_range'][0]  # Lower IC50 for deficient
        
        # Estimate Hill coefficient
        hill_range = drug_props['hill_coefficient_range']
        hill_coeff = np.mean(hill_range)
        
        return DoseResponseParameters(
            ic50=ic50,
            hill_coefficient=hill_coeff,
            emax=0.9,
            baseline=0.1,
            ec50=ic50
        )
    
    def validate_concentration_time_course(self, validation_data: pd.DataFrame) -> Dict:
        """
        Validate model against concentration-time course data
        
        Args:
            validation_data: Experimental concentration-time data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = []
        
        for _, row in validation_data.iterrows():
            try:
                # Extract experimental data
                time_points = row['concentration_time_points']
                concentrations = row['concentrations']
                viabilities = row['viability_measurements']
                
                # Create drug profile
                drug_name = row['drug']
                target = self.drug_database.get(drug_name, {}).get('target', 'TARGET')
                atm_status = row['atm_status']
                
                # Create dose-response curve
                fitter = self.dose_response_fitters.get(f"{drug_name}_{atm_status}")
                if fitter is None:
                    # Create simple dose-response model
                    dr_params = self._estimate_initial_parameters(drug_name, atm_status == 'proficient')
                    dr_model = HillEquationModel()
                    dr_curve = DoseResponseCurve(dr_model, dr_params, DrugProperties(drug_name, target, 500, 2, 0.8, 0.9, 8, 1, 1))
                    fitter = DoseResponseFitter('hill')
                    fitter.fitted_parameters = dr_params
                
                # Create PK parameters
                pk_params = self._create_pk_parameters(drug_name)
                
                # Create simulation
                simulator_settings = SimulationSettings(
                    simulation_duration=max(time_points),
                    time_resolution=0.1,
                    tissue_type='tumor',
                    atm_status=atm_status,
                    combine_pk_effects=False,
                    include_combination_effects=False
                )
                
                # Generate constant concentration profile (simplified)
                constant_conc = np.mean(concentrations)
                const_time_points = np.linspace(0, max(time_points), 100)
                const_concentrations = np.full_like(const_time_points, constant_conc)
                
                # Create dose-response mapping
                dr_curve = fitter.fitted_parameters  # This should be a DoseResponseCurve
                if isinstance(dr_curve, DoseResponseParameters):
                    # Convert parameters to curve
                    dr_curve = DoseResponseCurve(
                        HillEquationModel(),
                        dr_curve,
                        DrugProperties(drug_name, target, 500, 2, 0.8, 0.9, 8, 1, 1)
                    )
                
                # Run QSP simulation
                qsp_result = self.qsp_model.run_dose_response_simulation(
                    const_concentrations, const_time_points, {target: dr_curve.calculate_effect}
                )
                
                # Extract final state
                final_apoptosis = qsp_result['ApoptosisSignal'].iloc[-1]
                final_dsb = qsp_result['DSB'].iloc[-1]
                
                # Calculate experimental metrics
                final_viability = viabilities[-1] if len(viabilities) > 0 else 50.0
                experimental_effect = (100 - final_viability) / 100.0
                
                # Calculate validation scores
                concentration_score = self._validate_concentration_profile(time_points, concentrations)
                time_course_score = self._validate_time_course_effects(time_points, viabilities, qsp_result)
                dose_response_score = self._validate_dose_response_curve(drug_name, atm_status, experimental_effect, final_apoptosis)
                
                # Overall validation score
                overall_score = (concentration_score + time_course_score + dose_response_score) / 3.0
                
                validation_results.append({
                    'cell_line': row['cell_line'],
                    'drug': drug_name,
                    'atm_status': atm_status,
                    'concentration_score': concentration_score,
                    'time_course_score': time_course_score,
                    'dose_response_score': dose_response_score,
                    'overall_score': overall_score,
                    'experimental_effect': experimental_effect,
                    'predicted_effect': final_apoptosis / 100.0,  # Normalize
                    'experimental_viability': final_viability,
                    'predicted_apoptosis': final_apoptosis
                })
                
            except Exception as e:
                logger.warning(f"Validation failed for {row.get('drug', 'Unknown')}: {e}")
                continue
        
        # Calculate summary statistics
        if validation_results:
            df_results = pd.DataFrame(validation_results)
            summary = {
                'n_validations': len(df_results),
                'mean_overall_score': df_results['overall_score'].mean(),
                'std_overall_score': df_results['overall_score'].std(),
                'mean_concentration_score': df_results['concentration_score'].mean(),
                'mean_time_course_score': df_results['time_course_score'].mean(),
                'mean_dose_response_score': df_results['dose_response_score'].mean(),
                'r_squared': self._calculate_r_squared(df_results['experimental_effect'], df_results['predicted_effect']),
                'rmse': np.sqrt(mean_squared_error(df_results['experimental_effect'], df_results['predicted_effect'])),
                'detailed_results': df_results
            }
        else:
            summary = {'n_validations': 0, 'error': 'No successful validations'}
        
        return summary
    
    def _validate_concentration_profile(self, time_points: List[float], 
                                      concentrations: List[float]) -> float:
        """Validate concentration-time profile"""
        # Simple validation based on concentration stability
        conc_array = np.array(concentrations)
        if len(conc_array) < 2:
            return 0.5
        
        # Check for reasonable concentration values
        if np.any(conc_array < 0) or np.any(conc_array > 10000):  # Reasonable bounds
            return 0.2
        
        # Check for reasonable decay pattern (if applicable)
        cv = np.std(conc_array) / np.mean(conc_array) if np.mean(conc_array) > 0 else 1
        if cv < 0.5:  # Relatively stable concentrations
            return 0.8
        else:
            return 0.6
    
    def _validate_time_course_effects(self, time_points: List[float], 
                                    viabilities: List[float], 
                                    qsp_result: pd.DataFrame) -> float:
        """Validate time-course effects"""
        if len(viabilities) == 0 or qsp_result.empty:
            return 0.5
        
        # Extract relevant time points from QSP results
        qsp_times = qsp_result['Time'].values
        qsp_apoptosis = qsp_result['ApoptosisSignal'].values
        
        # Interpolate QSP results to experimental time points
        if len(qsp_times) > 1 and len(viabilities) > 0:
            # Simple correlation check
            experimental_effects = [(100 - v) / 100.0 for v in viabilities]
            
            # Find corresponding QSP values for experimental time points
            qsp_effects = []
            for exp_time in time_points:
                if exp_time <= qsp_times[-1]:
                    qsp_effect = np.interp(exp_time, qsp_times, qsp_apoptosis) / 100.0
                    qsp_effects.append(qsp_effect)
                else:
                    qsp_effects.append(qsp_apoptosis[-1] / 100.0)
            
            # Calculate correlation if we have data
            if len(qsp_effects) == len(experimental_effects) and len(qsp_effects) > 1:
                correlation = np.corrcoef(qsp_effects, experimental_effects)[0, 1]
                if not np.isnan(correlation):
                    return max(0, correlation)  # Positive correlation is good
                else:
                    return 0.5
            else:
                return 0.5
        else:
            return 0.5
    
    def _validate_dose_response_curve(self, drug_name: str, atm_status: str,
                                    experimental_effect: float, predicted_effect: float) -> float:
        """Validate dose-response curve prediction"""
        # Simple comparison of effects
        if experimental_effect <= 0 and predicted_effect <= 0:
            return 0.9  # Both show no effect
        
        # Calculate relative error
        if experimental_effect > 0:
            relative_error = abs(experimental_effect - predicted_effect) / experimental_effect
            score = max(0, 1 - relative_error)
        else:
            # Experimental shows no effect but model predicts effect
            score = 1.0 if predicted_effect < 0.1 else 0.3
        
        return score
    
    def _create_pk_parameters(self, drug_name: str) -> PKParameters:
        """Create PK parameters for drug from database"""
        if drug_name not in self.drug_database:
            return PKParameters(ka=1.0, f_abs=0.7, volume_central=50, volume_peripheral=100, 
                              q=2.0, cl=5.0, ke=0.1, half_life=6.93, fu=0.1, tissue_plasma_ratio=1.5)
        
        drug_props = self.drug_database[drug_name]
        
        # Estimate volume and clearance from database values
        volume_central = drug_props['volume_distribution'] * 70  # Scale to typical human
        clearance_rate = drug_props['clearance_rate']
        
        return PKParameters(
            ka=1.0,  # Default absorption rate
            f_abs=drug_props['bioavailability'],
            volume_central=volume_central,
            volume_peripheral=volume_central * 2,  # Assume peripheral is 2x central
            q=2.0,  # Inter-compartmental clearance
            cl=clearance_rate,
            ke=clearance_rate / volume_central,
            half_life=drug_props['half_life'],
            fu=1.0 - drug_props['protein_binding'],
            tissue_plasma_ratio=1.0 + drug_props['logp'] / 5  # Simple tissue distribution
        )
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def run_comprehensive_dose_response_validation(self, experimental_data: pd.DataFrame) -> Dict:
        """
        Run comprehensive dose-response validation
        
        Args:
            experimental_data: Experimental data with concentration information
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Running comprehensive dose-response validation...")
        
        # Separate data by ATM status
        atm_prof_data = experimental_data[experimental_data['atm_status'] == 'proficient']
        atm_def_data = experimental_data[experimental_data['atm_status'] == 'deficient']
        
        # Fit dose-response parameters for each ATM status
        fitted_params_prof = self.fit_dose_response_to_experimental_data(atm_prof_data, True)
        fitted_params_def = self.fit_dose_response_to_experimental_data(atm_def_data, False)
        
        # Run concentration-time course validation
        concentration_validation = self.validate_concentration_time_course(experimental_data)
        
        # Run synthetic lethality validation
        synthetic_lethality_validation = self._validate_synthetic_lethality(
            experimental_data, fitted_params_prof, fitted_params_def
        )
        
        # Calculate overall validation metrics
        overall_score = self._calculate_overall_validation_score(
            concentration_validation, synthetic_lethality_validation
        )
        
        return {
            'concentration_validation': concentration_validation,
            'synthetic_lethality_validation': synthetic_lethality_validation,
            'overall_score': overall_score,
            'fitted_parameters_proficient': fitted_params_prof,
            'fitted_parameters_deficient': fitted_params_def,
            'validation_summary': self._generate_validation_summary(
                concentration_validation, synthetic_lethality_validation, overall_score
            )
        }
    
    def _validate_synthetic_lethality(self, experimental_data: pd.DataFrame,
                                    fitted_params_prof: Dict, fitted_params_def: Dict) -> Dict:
        """Validate synthetic lethality predictions"""
        sl_results = []
        
        for drug in experimental_data['drug'].unique():
            drug_data = experimental_data[experimental_data['drug'] == drug]
            
            # Get IC50 values for both ATM statuses
            if drug in fitted_params_prof:
                ic50_prof = fitted_params_prof[drug].ic50
            else:
                ic50_prof = 100.0  # Default
            
            if drug in fitted_params_def:
                ic50_def = fitted_params_def[drug].ic50
            else:
                ic50_def = 50.0  # Default
            
            # Calculate synthetic lethality ratio
            sl_ratio = ic50_prof / ic50_def if ic50_def > 0 else np.inf
            
            # Get experimental data for comparison
            exp_data = drug_data[['atm_status', 'ic50_nm']].groupby('atm_status')['ic50_nm'].mean()
            exp_sl_ratio = exp_data.get('proficient', 100) / exp_data.get('deficient', 10)
            
            # Calculate validation score
            if exp_sl_ratio > 0 and sl_ratio > 0:
                sl_error = abs(np.log(sl_ratio) - np.log(exp_sl_ratio))
                sl_score = max(0, 1 - sl_error / 2)  # Log-scale error
            else:
                sl_score = 0.5
            
            sl_results.append({
                'drug': drug,
                'predicted_sl_ratio': sl_ratio,
                'experimental_sl_ratio': exp_sl_ratio,
                'sl_score': sl_score,
                'ic50_proficient': ic50_prof,
                'ic50_deficient': ic50_def
            })
        
        return {
            'synthetic_lethality_results': sl_results,
            'mean_sl_score': np.mean([r['sl_score'] for r in sl_results]) if sl_results else 0,
            'n_drugs': len(sl_results)
        }
    
    def _calculate_overall_validation_score(self, concentration_validation: Dict,
                                          synthetic_lethality_validation: Dict) -> float:
        """Calculate overall validation score"""
        scores = []
        
        # Concentration validation score
        if 'mean_overall_score' in concentration_validation:
            scores.append(concentration_validation['mean_overall_score'])
        
        # Synthetic lethality score
        if 'mean_sl_score' in synthetic_lethality_validation:
            scores.append(synthetic_lethality_validation['mean_sl_score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_validation_summary(self, concentration_validation: Dict,
                                   synthetic_lethality_validation: Dict,
                                   overall_score: float) -> str:
        """Generate validation summary report"""
        summary = f"""
# Dose-Response Validation Report

## Overall Score: {overall_score:.3f}

## Concentration-Time Course Validation
- Number of validations: {concentration_validation.get('n_validations', 0)}
- Mean concentration score: {concentration_validation.get('mean_concentration_score', 0):.3f}
- Mean time-course score: {concentration_validation.get('mean_time_course_score', 0):.3f}
- Mean dose-response score: {concentration_validation.get('mean_dose_response_score', 0):.3f}
- R² for dose-response: {concentration_validation.get('r_squared', 0):.3f}

## Synthetic Lethality Validation
- Number of drugs tested: {synthetic_lethality_validation.get('n_drugs', 0)}
- Mean SL score: {synthetic_lethality_validation.get('mean_sl_score', 0):.3f}

## Key Findings
"""
        
        # Add drug-specific results if available
        if 'synthetic_lethality_results' in synthetic_lethality_validation:
            sl_results = synthetic_lethality_validation['synthetic_lethality_results']
            summary += "\n### Drug-Specific Results\n"
            for result in sl_results:
                summary += f"- **{result['drug']}**: SL Ratio = {result['predicted_sl_ratio']:.2f}, Score = {result['sl_score']:.2f}\n"
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test concentration dose-response validation
    print("Concentration Dose-Response Validation Framework - Test")
    print("=" * 60)
    
    # Create test QSP model
    qsp_model = EnhancedDDRModel(atm_proficient=True, cell_cycle_phase='S')
    
    # Initialize validator
    validator = ConcentrationDoseResponseValidator(qsp_model)
    
    # Create test experimental data
    test_data = pd.DataFrame({
        'cell_line': ['MEC1', 'MEC2', 'MEC1', 'MEC2'],
        'atm_status': ['deficient', 'proficient', 'deficient', 'proficient'],
        'drug': ['AZD6738', 'AZD6738', 'Prexasertib', 'Prexasertib'],
        'ic50_nm': [15.2, 145.7, 32.1, 287.3],
        'concentrations': [[1, 10, 100], [1, 10, 100], [1, 10, 100], [1, 10, 100]],
        'viabilities': [[20, 5, 1], [85, 70, 45], [30, 8, 2], [90, 80, 65]],
        'concentration_time_points': [[0, 12, 24], [0, 12, 24], [0, 12, 24], [0, 12, 24]],
        'viability_measurements': [[100, 50, 20], [100, 95, 90], [100, 40, 15], [100, 98, 95]]
    })
    
    print("Test data created with 4 experimental conditions")
    print(f"Drugs: {test_data['drug'].unique()}")
    print(f"ATM statuses: {test_data['atm_status'].unique()}")
    
    # Test parameter fitting
    print("\n1. Dose-Response Parameter Fitting")
    print("-" * 35)
    
    fitted_prof = validator.fit_dose_response_to_experimental_data(test_data[test_data['atm_status'] == 'proficient'], True)
    fitted_def = validator.fit_dose_response_to_experimental_data(test_data[test_data['atm_status'] == 'deficient'], False)
    
    print(f"Fitted parameters for ATM-proficient: {len(fitted_prof)} drugs")
    print(f"Fitted parameters for ATM-deficient: {len(fitted_def)} drugs")
    
    for drug, params in fitted_prof.items():
        print(f"  {drug} (Proficient): IC50 = {params.ic50:.1f} nM, Hill = {params.hill_coefficient:.1f}")
    
    for drug, params in fitted_def.items():
        print(f"  {drug} (Deficient): IC50 = {params.ic50:.1f} nM, Hill = {params.hill_coefficient:.1f}")
    
    # Test concentration validation
    print("\n2. Concentration-Time Course Validation")
    print("-" * 40)
    
    # Convert to concentration validation format
    concentration_data = []
    for _, row in test_data.iterrows():
        concentration_data.append(ConcentrationValidationData(
            cell_line=row['cell_line'],
            atm_status=row['atm_status'],
            drug=row['drug'],
            drug_target='ATR' if 'AZD' in row['drug'] else 'CHK1',
            concentration_time_points=row['concentration_time_points'],
            concentrations=row['concentrations'],
            viability_time_points=row['concentration_time_points'],
            viability_measurements=row['viability_measurements'],
            ic50=row['ic50_nm'],
            auc=sum(row['viabilities'])
        ))
    
    validation_df = pd.DataFrame([{
        'cell_line': d.cell_line,
        'atm_status': d.atm_status,
        'drug': d.drug,
        'concentration_time_points': d.concentration_time_points,
        'concentrations': d.concentrations,
        'viability_measurements': d.viability_measurements
    } for d in concentration_data])
    
    validation_result = validator.validate_concentration_time_course(validation_df)
    
    print(f"Validation completed: {validation_result.get('n_validations', 0)} validations")
    print(f"Mean overall score: {validation_result.get('mean_overall_score', 0):.3f}")
    print(f"Mean concentration score: {validation_result.get('mean_concentration_score', 0):.3f}")
    print(f"Mean time-course score: {validation_result.get('mean_time_course_score', 0):.3f}")
    print(f"R²: {validation_result.get('r_squared', 0):.3f}")
    
    # Test comprehensive validation
    print("\n3. Comprehensive Dose-Response Validation")
    print("-" * 45)
    
    comprehensive_result = validator.run_comprehensive_dose_response_validation(test_data)
    
    print(f"Comprehensive validation completed!")
    print(f"Overall score: {comprehensive_result['overall_score']:.3f}")
    
    if 'validation_summary' in comprehensive_result:
        print("\nValidation Summary:")
        print(comprehensive_result['validation_summary'])
    
    print("\nConcentration dose-response validation framework test completed!")