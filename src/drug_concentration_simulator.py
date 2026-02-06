"""
Drug Concentration Simulator for Synthetic Lethality QSP Model
==============================================================

This module provides integration between dose-response and pharmacokinetic models
to simulate concentration-dependent drug effects in the QSP model, including:
- Integration of PK models with dose-response curves
- Time-dependent drug concentration effects
- Tissue-specific concentration modeling
- Multi-drug combination effects
- Concentration-dependent synthetic lethality calculations

Author: Kilo Code
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings

# Import our modeling frameworks
from dose_response_modeling import (
    DoseResponseCurve, DrugProperties, DoseResponseParameters,
    DoseResponseAnalyzer, CombinationDoseResponse
)
from pharmacokinetic_modeling import (
    PKParameters, DosingRegimen, PharmacokineticModeler, ADMEModeler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DrugProfile:
    """Complete drug profile combining PK, dose-response, and drug properties"""
    name: str
    target: str
    drug_properties: DrugProperties
    dose_response_curve: DoseResponseCurve
    pk_parameters: PKParameters
    
    def __post_init__(self):
        """Validate drug profile consistency"""
        if self.dose_response_curve.drug_properties.name != self.name:
            logger.warning(f"Drug name mismatch: {self.name} vs {self.dose_response_curve.drug_properties.name}")
        if self.dose_response_curve.drug_properties.target != self.target:
            logger.warning(f"Target mismatch: {self.target} vs {self.dose_response_curve.drug_properties.target}")

@dataclass
class SimulationSettings:
    """Settings for concentration-effect simulation"""
    simulation_duration: float  # hours
    time_resolution: float      # hours (time step)
    tissue_type: str           # 'plasma', 'liver', 'brain', etc.
    atm_status: str            # 'proficient', 'deficient'
    combine_pk_effects: bool   # Whether to simulate PK over time
    include_combination_effects: bool  # For multi-drug simulations

class DrugConcentrationSimulator:
    """Main simulator for concentration-dependent drug effects"""
    
    def __init__(self, simulation_settings: SimulationSettings):
        """
        Initialize drug concentration simulator
        
        Args:
            simulation_settings: Simulation configuration
        """
        self.settings = simulation_settings
        self.pk_modeler = PharmacokineticModeler('2-compartment')  # Default to 2-compartment
        self.dr_analyzer = DoseResponseAnalyzer()
        self.combination_model = CombinationDoseResponse()
        
        # Storage for simulation results
        self.drug_profiles = {}
        self.simulation_results = {}
    
    def add_drug(self, drug_profile: DrugProfile):
        """
        Add drug profile to simulator
        
        Args:
            drug_profile: Complete drug profile
        """
        self.drug_profiles[drug_profile.name] = drug_profile
        logger.info(f"Added drug: {drug_profile.name} (target: {drug_profile.target})")
    
    def simulate_single_drug_effect(self, drug_name: str, dosing_regimen: DosingRegimen,
                                  initial_conditions: Optional[Dict] = None) -> Dict:
        """
        Simulate time-dependent effect of single drug
        
        Args:
            drug_name: Name of drug to simulate
            dosing_regimen: Dosing regimen
            initial_conditions: Initial model state (if using QSP integration)
            
        Returns:
            Dictionary with simulation results
        """
        if drug_name not in self.drug_profiles:
            raise ValueError(f"Drug {drug_name} not found in simulator")
        
        drug_profile = self.drug_profiles[drug_name]
        time_points = np.arange(0, self.settings.simulation_duration, self.settings.time_resolution)
        
        # Simulate pharmacokinetics if requested
        if self.settings.combine_pk_effects:
            pk_results = self.pk_modeler.simulate_multiple_doses(
                drug_profile.pk_parameters, dosing_regimen
            )
            plasma_concentrations = pk_results['concentrations']
        else:
            # Use constant concentration equivalent to steady-state
            total_dose = dosing_regimen.dose * dosing_regimen.n_doses
            avg_concentration = total_dose / (drug_profile.pk_parameters.volume_central * 24)  # Simplified
            plasma_concentrations = np.full_like(time_points, avg_concentration)
        
        # Interpolate PK results to simulation time points
        if self.settings.combine_pk_effects:
            interpolated_conc = np.interp(time_points, pk_results['time'], plasma_concentrations)
        else:
            interpolated_conc = plasma_concentrations
        
        # Calculate tissue concentrations
        tissue_concentrations = self._calculate_tissue_concentrations(
            interpolated_conc, drug_profile
        )
        
        # Calculate dose-response effects
        effects = self.dose_response_effects([drug_name], tissue_concentrations, time_points)
        
        # Compile results
        results = {
            'time': time_points,
            'plasma_concentration': interpolated_conc,
            'tissue_concentration': tissue_concentrations,
            'drug_effect': effects[drug_name],
            'pk_results': pk_results if self.settings.combine_pk_effects else None,
            'drug_profile': drug_profile,
            'dosing_regimen': dosing_regimen,
            'settings': self.settings
        }
        
        return results
    
    def simulate_drug_combination(self, drug_names: List[str], dosing_regimens: Dict[str, DosingRegimen],
                                combination_model: str = 'bliss') -> Dict:
        """
        Simulate time-dependent effects of drug combination
        
        Args:
            drug_names: List of drug names
            dosing_regimens: Dictionary of drug names to dosing regimens
            combination_model: Combination model ('bliss', 'loewe', 'hsa')
            
        Returns:
            Dictionary with combination simulation results
        """
        # Check all drugs are available
        for drug_name in drug_names:
            if drug_name not in self.drug_profiles:
                raise ValueError(f"Drug {drug_name} not found in simulator")
        
        time_points = np.arange(0, self.settings.simulation_duration, self.settings.time_resolution)
        
        # Simulate individual PK profiles
        individual_results = {}
        for drug_name in drug_names:
            individual_results[drug_name] = self.simulate_single_drug_effect(
                drug_name, dosing_regimens[drug_name]
            )
        
        # Calculate combination effects at each time point
        combination_effects = np.zeros_like(time_points)
        individual_effects = {drug: np.zeros_like(time_points) for drug in drug_names}
        
        for i, t in enumerate(time_points):
            # Get concentrations at this time point
            concentrations = []
            curves = []
            for drug_name in drug_names:
                tissue_conc = individual_results[drug_name]['tissue_concentration'][i]
                concentrations.append(tissue_conc)
                curves.append(self.drug_profiles[drug_name].dose_response_curve)
            
            # Calculate individual effects
            for j, drug_name in enumerate(drug_names):
                effect = curves[j].calculate_effect(np.array([concentrations[j]]))[0]
                individual_effects[drug_name][i] = effect
            
            # Calculate combination effect
            if combination_model == 'bliss':
                combination_effects[i] = self._calculate_bliss_combination(concentrations, curves)
            elif combination_model == 'loewe':
                combination_effects[i] = self._calculate_loewe_combination(concentrations, curves)
            elif combination_model == 'hsa':
                combination_effects[i] = self._calculate_hsa_combination(concentrations, curves)
            else:
                raise ValueError(f"Unknown combination model: {combination_model}")
        
        # Compile results
        results = {
            'time': time_points,
            'individual_effects': individual_effects,
            'combination_effect': combination_effects,
            'individual_results': individual_results,
            'combination_model': combination_model,
            'drug_names': drug_names,
            'settings': self.settings
        }
        
        return results
    
    def simulate_synthetic_lethality_dose_response(self, target_drug: str, reference_drug: str,
                                                 concentration_range: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate concentration-dependent synthetic lethality between two drugs
        
        Args:
            target_drug: Drug to test (ATM-deficient sensitivity)
            reference_drug: Reference drug for comparison
            concentration_range: Concentration range to test
            
        Returns:
            Dictionary with synthetic lethality analysis
        """
        if target_drug not in self.drug_profiles or reference_drug not in self.drug_profiles:
            raise ValueError("Both drugs must be added to simulator")
        
        # Set up concentration range if not provided
        if concentration_range is None:
            concentration_range = np.logspace(-2, 2, 20)  # 0.01 to 100 μM
        
        # Get dose-response curves
        target_curve = self.drug_profiles[target_drug].dose_response_curve
        reference_curve = self.drug_profiles[reference_drug].dose_response_curve
        
        # Calculate effects at each concentration
        target_effects = target_curve.calculate_effect(concentration_range)
        reference_effects = reference_curve.calculate_effect(concentration_range)
        
        # Calculate synthetic lethality metrics
        sl_analysis = self.dr_analyzer.analyze_synthetic_lethality(
            reference_curve, target_curve
        )
        
        # Create dose-response comparison
        comparison_results = {
            'concentrations': concentration_range,
            'target_effects': target_effects,
            'reference_effects': reference_effects,
            'synthetic_lethality_analysis': sl_analysis,
            'target_drug': target_drug,
            'reference_drug': reference_drug,
            'dose_response_curves': {
                'target': target_curve,
                'reference': reference_curve
            }
        }
        
        return comparison_results
    
    def optimize_therapeutic_window(self, target_drug: str, reference_drug: str,
                                  target_effect: float = 0.8) -> Dict:
        """
        Optimize dosing for maximum therapeutic window
        
        Args:
            target_drug: Drug targeting ATM-deficient cells
            reference_drug: Reference drug for ATM-proficient cells
            target_effect: Desired effect level in target cells
            
        Returns:
            Dictionary with optimization results
        """
        if target_drug not in self.drug_profiles or reference_drug not in self.drug_profiles:
            raise ValueError("Both drugs must be added to simulator")
        
        # Get dose-response curves
        target_curve = self.drug_profiles[target_drug].dose_response_curve
        reference_curve = self.drug_profiles[reference_drug].dose_response_curve
        
        # Perform dose optimization
        optimization = self.dr_analyzer.dose_optimization(
            target_effect, reference_curve, target_curve
        )
        
        # Calculate PK parameters for recommended dose
        target_pk = self.drug_profiles[target_drug].pk_parameters
        
        # Estimate exposure metrics
        recommended_dose = optimization['optimal_concentration'] * target_pk.volume_central / 1000  # mg
        
        # Create dosing regimen
        dosing_regimen = DosingRegimen(
            dose=recommended_dose,
            dosing_interval=24,  # daily dosing
            n_doses=7,  # one week
            route='oral',
            start_time=0
        )
        
        # Simulate exposure
        if self.settings.combine_pk_effects:
            exposure = self.pk_modeler.predict_exposure_metrics(target_pk, dosing_regimen)
        else:
            exposure = {'c_max': optimization['optimal_concentration'], 'avg_concentration': optimization['optimal_concentration']}
        
        optimization_results = {
            'dosing_optimization': optimization,
            'recommended_dose_mg': recommended_dose,
            'dosing_regimen': dosing_regimen,
            'predicted_exposure': exposure,
            'therapeutic_assessment': {
                'expected_target_effect': target_effect,
                'predicted_reference_toxicity': reference_curve.calculate_effect(np.array([optimization['optimal_concentration']]))[0],
                'safety_margin': optimization['safety_margin'],
                'recommendation': optimization['dose_recommendation']
            }
        }
        
        return optimization_results
    
    def run_time_course_simulation(self, drug_name: str, dosing_regimen: DosingRegimen,
                                 qsp_model_func: Optional[callable] = None) -> Dict:
        """
        Run time-course simulation integrating PK/PD with QSP model
        
        Args:
            drug_name: Drug to simulate
            dosing_regimen: Dosing regimen
            qsp_model_func: Optional function to run QSP model at each time point
            
        Returns:
            Dictionary with integrated simulation results
        """
        if drug_name not in self.drug_profiles:
            raise ValueError(f"Drug {drug_name} not found in simulator")
        
        # Get single drug simulation
        pd_results = self.simulate_single_drug_effect(drug_name, dosing_regimen)
        
        # Initialize arrays for time-course results
        time_points = pd_results['time']
        concentrations = pd_results['tissue_concentration']
        effects = pd_results['drug_effect']
        
        if qsp_model_func is not None:
            # Initialize QSP results storage
            qsp_results = {
                'dsb_levels': np.zeros_like(time_points),
                'apoptosis_signals': np.zeros_like(time_points),
                'pathway_activities': {name: np.zeros_like(time_points) 
                                     for name in ['ATM_active', 'ATR_active', 'HR_Activity', 'PARP_Activity']}
            }
            
            # Run QSP model at each time point
            for i, (t, conc, effect) in enumerate(zip(time_points, concentrations, effects)):
                try:
                    # Create time-dependent drug effects
                    drug_effects = {self.drug_profiles[drug_name].target: effect}
                    
                    # Run QSP simulation for this time point
                    qsp_result = qsp_model_func(drug_effects)
                    
                    # Store results
                    if hasattr(qsp_result, 'iloc'):
                        # DataFrame result
                        final_state = qsp_result.iloc[-1]
                        qsp_results['dsb_levels'][i] = final_state.get('DSB', 0)
                        qsp_results['apoptosis_signals'][i] = final_state.get('ApoptosisSignal', 0)
                        for pathway in qsp_results['pathway_activities']:
                            if pathway in final_state:
                                qsp_results['pathway_activities'][pathway][i] = final_state[pathway]
                    else:
                        # Dictionary result
                        qsp_results['dsb_levels'][i] = qsp_result.get('DSB', 0)
                        qsp_results['apoptosis_signals'][i] = qsp_result.get('ApoptosisSignal', 0)
                        for pathway in qsp_results['pathway_activities']:
                            if pathway in qsp_result:
                                qsp_results['pathway_activities'][pathway][i] = qsp_result[pathway]
                
                except Exception as e:
                    logger.warning(f"QSP simulation failed at time {t}: {e}")
                    # Use previous values or default
                    if i > 0:
                        for key in qsp_results:
                            if key != 'pathway_activities':
                                qsp_results[key][i] = qsp_results[key][i-1]
                            else:
                                for pathway in qsp_results[key]:
                                    qsp_results[key][pathway][i] = qsp_results[key][pathway][i-1]
        else:
            qsp_results = None
        
        # Combine results
        integrated_results = {
            'pharmacodynamics': pd_results,
            'qsp_results': qsp_results,
            'time_points': time_points,
            'drug_name': drug_name,
            'dosing_regimen': dosing_regimen,
            'simulation_settings': self.settings
        }
        
        return integrated_results
    
    def _calculate_tissue_concentrations(self, plasma_conc: np.ndarray, 
                                       drug_profile: DrugProfile) -> np.ndarray:
        """Calculate tissue concentrations from plasma concentrations"""
        tissue_ratio = drug_profile.pk_parameters.tissue_plasma_ratio
        
        # Apply tissue-specific corrections
        if self.settings.tissue_type == 'brain':
            # Blood-brain barrier penetration
            brain_factor = 0.1 * (1 + drug_profile.drug_properties.logp / 3)
            tissue_ratio *= brain_factor
        elif self.settings.tissue_type == 'liver':
            # Enhanced liver exposure
            liver_factor = 2.0
            tissue_ratio *= liver_factor
        elif self.settings.tissue_type == 'tumor':
            # Enhanced tumor penetration (active transport, EPR effect)
            tumor_factor = 1.5
            tissue_ratio *= tumor_factor
        
        return plasma_conc * tissue_ratio
    
    def _calculate_bliss_combination(self, concentrations: List[float], 
                                   curves: List[DoseResponseCurve]) -> float:
        """Calculate Bliss independence combination effect"""
        individual_effects = []
        for conc, curve in zip(concentrations, curves):
            effect = curve.calculate_effect(np.array([conc]))[0]
            individual_effects.append(effect)
        
        # Bliss independence: E_combined = E1 + E2 - (E1 * E2)
        combined = individual_effects[0]
        for effect in individual_effects[1:]:
            combined = combined + effect - (combined * effect)
        
        return min(1.0, combined)
    
    def _calculate_loewe_combination(self, concentrations: List[float],
                                   curves: List[DoseResponseCurve]) -> float:
        """Calculate Loewe additivity combination effect"""
        # Simplified Loewe calculation
        combination_index = 0
        for conc, curve in zip(concentrations, curves):
            ic50 = curve.calculate_ic50()
            ci = conc / ic50 if ic50 > 0 else 0
            combination_index += ci
        
        # Convert combination index to effect
        if combination_index <= 1:
            effect = 0.5 * combination_index
        else:
            effect = 0.5 + 0.3 * (combination_index - 1)
        
        return min(1.0, effect)
    
    def _calculate_hsa_combination(self, concentrations: List[float],
                                 curves: List[DoseResponseCurve]) -> float:
        """Calculate Highest Single Agent combination effect"""
        individual_effects = []
        for conc, curve in zip(concentrations, curves):
            effect = curve.calculate_effect(np.array([conc]))[0]
            individual_effects.append(effect)
        
        return max(individual_effects)
    
    def dose_response_effects(self, drug_names: List[str], concentrations: np.ndarray,
                            time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate dose-response effects for multiple drugs over time"""
        effects = {}
        
        for drug_name in drug_names:
            if drug_name in self.drug_profiles:
                curve = self.drug_profiles[drug_name].dose_response_curve
                drug_effects = curve.calculate_effect(concentrations)
                effects[drug_name] = drug_effects
            else:
                logger.warning(f"Drug {drug_name} not found, using zero effect")
                effects[drug_name] = np.zeros_like(concentrations)
        
        return effects

class VirtualPatientSimulator:
    """Simulator for virtual patient studies with population PK/PD"""
    
    def __init__(self, population_size: int = 100):
        """
        Initialize virtual patient simulator
        
        Args:
            population_size: Number of virtual patients
        """
        self.population_size = population_size
        self.concentration_simulator = None
        self.variability_models = {}
        
        # Default variability parameters
        self.default_variability = {
            'clearance_cv': 0.3,     # 30% CV for clearance
            'volume_cv': 0.25,       # 25% CV for volume
            'absorption_cv': 0.4,    # 40% CV for absorption
            'response_cv': 0.2       # 20% CV for drug response
        }
    
    def set_population_variability(self, variability_params: Dict):
        """Set population variability parameters"""
        self.default_variability.update(variability_params)
    
    def generate_virtual_population(self, drug_name: str, base_drug_profile: DrugProfile) -> List[DrugProfile]:
        """Generate virtual patient population for a drug"""
        population = []
        
        np.random.seed(42)  # For reproducible results
        
        for i in range(self.population_size):
            # Generate random factors (log-normal distribution)
            clearance_factor = np.random.lognormal(0, self.default_variability['clearance_cv'])
            volume_factor = np.random.lognormal(0, self.default_variability['volume_cv'])
            absorption_factor = np.random.lognormal(0, self.default_variability['absorption_cv'])
            response_factor = np.random.lognormal(0, self.default_variability['response_cv'])
            
            # Modify PK parameters
            modified_pk = PKParameters(
                ka=base_drug_profile.pk_parameters.ka * absorption_factor,
                f_abs=base_drug_profile.pk_parameters.f_abs,
                volume_central=base_drug_profile.pk_parameters.volume_central * volume_factor,
                volume_peripheral=base_drug_profile.pk_parameters.volume_peripheral * volume_factor,
                q=base_drug_profile.pk_parameters.q,
                cl=base_drug_profile.pk_parameters.cl * clearance_factor,
                ke=base_drug_profile.pk_parameters.ke,
                half_life=base_drug_profile.pk_parameters.half_life,
                fu=base_drug_profile.pk_parameters.fu,
                tissue_plasma_ratio=base_drug_profile.pk_parameters.tissue_plasma_ratio
            )
            
            # Modify dose-response parameters
            original_params = base_drug_profile.dose_response_curve.parameters
            modified_dr_params = DoseResponseParameters(
                ic50=original_params.ic50 * response_factor,
                hill_coefficient=original_params.hill_coefficient,
                emax=original_params.emax,
                baseline=original_params.baseline,
                ec50=original_params.ec50 * response_factor
            )
            
            # Create new dose-response curve
            modified_curve = DoseResponseCurve(
                base_drug_profile.dose_response_curve.model,
                modified_dr_params,
                base_drug_profile.drug_properties
            )
            
            # Create virtual patient drug profile
            virtual_profile = DrugProfile(
                name=f"{drug_name}_Patient_{i+1}",
                target=base_drug_profile.target,
                drug_properties=base_drug_profile.drug_properties,
                dose_response_curve=modified_curve,
                pk_parameters=modified_pk
            )
            
            population.append(virtual_profile)
        
        logger.info(f"Generated virtual population of {len(population)} patients for {drug_name}")
        return population
    
    def run_population_study(self, drug_name: str, dosing_regimen: DosingRegimen,
                           concentration_simulator: DrugConcentrationSimulator) -> Dict:
        """
        Run population PK/PD study
        
        Args:
            drug_name: Drug to study
            dosing_regimen: Dosing regimen
            concentration_simulator: Concentration simulator with base drug profile
            
        Returns:
            Dictionary with population study results
        """
        if drug_name not in concentration_simulator.drug_profiles:
            raise ValueError(f"Drug {drug_name} not found in simulator")
        
        # Generate virtual population
        base_profile = concentration_simulator.drug_profiles[drug_name]
        population = self.generate_virtual_population(drug_name, base_profile)
        
        # Simulate each patient
        population_results = []
        for i, patient_profile in enumerate(population):
            # Temporarily replace drug profile in simulator
            original_profile = concentration_simulator.drug_profiles[drug_name]
            concentration_simulator.drug_profiles[drug_name] = patient_profile
            
            try:
                # Run simulation for this patient
                patient_result = concentration_simulator.simulate_single_drug_effect(
                    drug_name, dosing_regimen
                )
                patient_result['patient_id'] = i + 1
                population_results.append(patient_result)
                
            except Exception as e:
                logger.warning(f"Simulation failed for patient {i+1}: {e}")
                continue
            
            finally:
                # Restore original profile
                concentration_simulator.drug_profiles[drug_name] = original_profile
        
        # Calculate population statistics
        if population_results:
            # Extract key metrics
            c_max_values = [r['pk_results']['peak_concentration'] if r.get('pk_results') else 0 
                          for r in population_results if r.get('pk_results')]
            auc_values = [r['pk_results']['auc'] if r.get('pk_results') else 0 
                         for r in population_results if r.get('pk_results')]
            effect_max = [np.max(r['drug_effect']) for r in population_results]
            
            # Statistical summary
            population_summary = {
                'n_patients': len(population_results),
                'c_max_mean': np.mean(c_max_values) if c_max_values else 0,
                'c_max_std': np.std(c_max_values) if c_max_values else 0,
                'c_max_cv': np.std(c_max_values)/np.mean(c_max_values) if c_max_values and np.mean(c_max_values) > 0 else 0,
                'auc_mean': np.mean(auc_values) if auc_values else 0,
                'auc_std': np.std(auc_values) if auc_values else 0,
                'effect_mean': np.mean(effect_max),
                'effect_std': np.std(effect_max),
                'effect_cv': np.std(effect_max)/np.mean(effect_max) if np.mean(effect_max) > 0 else 0
            }
        else:
            population_summary = {'n_patients': 0, 'error': 'No successful simulations'}
        
        return {
            'population_results': population_results,
            'population_summary': population_summary,
            'base_drug_profile': base_profile,
            'dosing_regimen': dosing_regimen
        }

# Example usage and testing
if __name__ == "__main__":
    # Test drug concentration simulator
    print("Drug Concentration Simulator - Test")
    print("=" * 50)
    
    # Create test drug profile
    from dose_response_modeling import HillEquationModel, DoseResponseParameters, DrugProperties
    from pharmacokinetic_modeling import PKParameters, DosingRegimen
    
    # Test drug properties
    drug_props = DrugProperties("Test ATR Inhibitor", "ATR", 450.0, 2.5, 0.8, 0.9, 8.0, 1.2, 1.0)
    
    # Test dose-response parameters
    dr_params = DoseResponseParameters(
        ic50=10.0,  # nM
        hill_coefficient=2.0,
        emax=0.9,
        baseline=0.1,
        ec50=10.0
    )
    
    # Create dose-response curve
    dr_model = HillEquationModel()
    dr_curve = DoseResponseCurve(dr_model, dr_params, drug_props)
    
    # Test PK parameters
    pk_params = PKParameters(
        ka=1.2,          # 1/h
        f_abs=0.8,       # Bioavailability
        volume_central=50,   # L
        volume_peripheral=100,  # L
        q=2.0,           # L/h
        cl=5.0,          # L/h
        ke=0.1,          # 1/h
        half_life=6.93,  # h
        fu=0.1,          # Fraction unbound
        tissue_plasma_ratio=1.5  # Tissue partitioning
    )
    
    # Create drug profile
    drug_profile = DrugProfile(
        name="Test ATR Inhibitor",
        target="ATR",
        drug_properties=drug_props,
        dose_response_curve=dr_curve,
        pk_parameters=pk_params
    )
    
    # Test simulation settings
    sim_settings = SimulationSettings(
        simulation_duration=72.0,  # 72 hours
        time_resolution=1.0,       # 1 hour
        tissue_type='liver',       # Liver tissue
        atm_status='deficient',    # ATM-deficient
        combine_pk_effects=True,   # Include PK
        include_combination_effects=False
    )
    
    # Test concentration simulator
    print("\n1. Single Drug Simulation")
    print("-" * 30)
    
    simulator = DrugConcentrationSimulator(sim_settings)
    simulator.add_drug(drug_profile)
    
    # Test dosing regimen
    regimen = DosingRegimen(
        dose=50,            # mg
        dosing_interval=24, # hours (daily)
        n_doses=3,         # 3 doses
        route='oral',
        start_time=0
    )
    
    # Run simulation
    result = simulator.simulate_single_drug_effect("Test ATR Inhibitor", regimen)
    
    print(f"Simulation duration: {sim_settings.simulation_duration} hours")
    print(f"Peak concentration: {np.max(result['plasma_concentration']):.2f} mg/L")
    print(f"Peak tissue concentration: {np.max(result['tissue_concentration']):.2f} mg/L")
    print(f"Maximum drug effect: {np.max(result['drug_effect']):.3f}")
    
    # Test synthetic lethality
    print("\n2. Synthetic Lethality Analysis")
    print("-" * 35)
    
    # Create reference drug (less sensitive)
    ref_drug_props = DrugProperties("Reference Drug", "Target", 400.0, 2.0, 0.7, 0.8, 6.0, 0.8, 0.8)
    ref_dr_params = DoseResponseParameters(ic50=50.0, hill_coefficient=1.8, emax=0.8, baseline=0.1, ec50=50.0)
    ref_dr_curve = DoseResponseCurve(dr_model, ref_dr_params, ref_drug_props)
    ref_pk_params = PKParameters(ka=0.8, f_abs=0.7, volume_central=40, volume_peripheral=80, q=1.5, 
                                cl=3.0, ke=0.075, half_life=9.24, fu=0.15, tissue_plasma_ratio=1.2)
    
    ref_drug_profile = DrugProfile("Reference Drug", "Target", ref_drug_props, ref_dr_curve, ref_pk_params)
    simulator.add_drug(ref_drug_profile)
    
    # Run synthetic lethality analysis
    sl_result = simulator.simulate_synthetic_lethality_dose_response(
        "Test ATR Inhibitor", "Reference Drug"
    )
    
    sl_analysis = sl_result['synthetic_lethality_analysis']
    print(f"IC50 (Target): {sl_analysis['ic50_mutant']:.1f} nM")
    print(f"IC50 (Reference): {sl_analysis['ic50_wildtype']:.1f} nM")
    print(f"Synthetic lethality ratio: {sl_analysis['synthetic_lethality_ratio']:.1f}")
    print(f"Classification: {sl_analysis['classification']}")
    
    # Test dose optimization
    print("\n3. Dose Optimization")
    print("-" * 20)
    
    optimization = simulator.optimize_therapeutic_window("Test ATR Inhibitor", "Reference Drug")
    print(f"Optimal concentration: {optimization['dosing_optimization']['optimal_concentration']:.1f} nM")
    print(f"Recommended dose: {optimization['recommended_dose_mg']:.1f} mg")
    print(f"Safety margin: {optimization['dosing_optimization']['safety_margin']:.1f}")
    print(f"Recommendation: {optimization['dosing_optimization']['dose_recommendation']}")
    
    # Test virtual patient simulation
    print("\n4. Virtual Patient Study")
    print("-" * 25)
    
    patient_sim = VirtualPatientSimulator(population_size=10)
    population_study = patient_sim.run_population_study("Test ATR Inhibitor", regimen, simulator)
    
    summary = population_study['population_summary']
    print(f"Population size: {summary['n_patients']}")
    print(f"C_max mean ± std: {summary['c_max_mean']:.2f} ± {summary['c_max_std']:.2f} mg/L")
    print(f"C_max CV: {summary['c_max_cv']:.1%}")
    print(f"Effect mean ± std: {summary['effect_mean']:.3f} ± {summary['effect_std']:.3f}")
    print(f"Effect CV: {summary['effect_cv']:.1%}")
    
    print("\nDrug concentration simulator test completed!")