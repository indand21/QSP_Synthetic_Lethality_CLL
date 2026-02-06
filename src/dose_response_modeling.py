"""
Dose-Response Modeling Framework for Synthetic Lethality QSP Model
==================================================================

This module provides comprehensive dose-response modeling capabilities including:
- Hill equation-based dose-response modeling
- Drug concentration effect relationships (E_max, IC50, Hill coefficient)
- Concentration-time profile modeling
- Emax and sigmoidal dose-response models
- Integration with QSP model for concentration-dependent effects

Author: Kilo Code
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DrugProperties:
    """Container for drug properties affecting dose-response"""
    name: str
    target: str
    molecular_weight: float  # Da
    logp: float  # Partition coefficient
    bioavailability: float  # Fraction (0-1)
    protein_binding: float  # Fraction (0-1)
    half_life: float  # hours
    clearance_rate: float  # L/h
    volume_distribution: float  # L/kg
    
@dataclass
class DoseResponseParameters:
    """Parameters for dose-response models"""
    ic50: float  # Concentration for 50% inhibition (nM)
    hill_coefficient: float  # Steepness of dose-response curve
    emax: float  # Maximum effect (0-1)
    baseline: float  # Baseline effect (0-1)
    ec50: float  # Concentration for 50% effect (nM)
    
class DoseResponseModel(ABC):
    """Abstract base class for dose-response models"""
    
    @abstractmethod
    def effect(self, concentration: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """Calculate effect at given concentration(s)"""
        pass
    
    @abstractmethod
    def inverse_effect(self, effect: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """Calculate concentration(s) for given effect(s)"""
        pass

class HillEquationModel(DoseResponseModel):
    """
    Hill equation dose-response model
    
    Effect = E_min + (E_max - E_min) * (C^n) / (EC50^n + C^n)
    where C is concentration, n is Hill coefficient
    """
    
    def effect(self, concentration: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate effect using Hill equation
        
        Args:
            concentration: Drug concentration(s) in nM
            parameters: Dose-response parameters
            
        Returns:
            Effect value(s) (0-1 scale)
        """
        C = np.asarray(concentration, dtype=float)
        ec50 = parameters.ec50
        n = parameters.hill_coefficient
        emax = parameters.emax
        baseline = parameters.baseline
        
        # Avoid division by zero
        ec50_powered = np.power(ec50, n)
        c_powered = np.power(np.maximum(C, 1e-12), n)  # Ensure positive values
        
        # Hill equation
        effect = baseline + (emax - baseline) * (c_powered / (ec50_powered + c_powered))
        
        return np.clip(effect, 0, 1)  # Ensure effect is between 0 and 1
    
    def inverse_effect(self, effect: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate concentration from effect using Hill equation
        
        Args:
            effect: Effect value(s) (0-1 scale)
            parameters: Dose-response parameters
            
        Returns:
            Concentration(s) in nM
        """
        E = np.asarray(effect, dtype=float)
        ec50 = parameters.ec50
        n = parameters.hill_coefficient
        emax = parameters.emax
        baseline = parameters.baseline
        
        # Inverse Hill equation
        # E = baseline + (emax - baseline) * (C^n) / (EC50^n + C^n)
        # Solve for C
        numerator = E - baseline
        denominator = emax - baseline
        
        # Avoid division by zero
        ratio = np.divide(numerator, denominator, 
                         out=np.zeros_like(numerator), where=denominator!=0)
        
        # Calculate concentration
        c_powered = ec50**n * ratio / (1 - ratio)
        concentration = np.power(np.maximum(c_powered, 1e-12), 1/n)
        
        return np.clip(concentration, 0, np.inf)

class SigmoidalModel(DoseResponseModel):
    """
    Sigmoidal dose-response model using logistic function
    
    Effect = baseline + (emax - baseline) / (1 + exp(-k*(log(C) - log(EC50))))
    """
    
    def effect(self, concentration: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate effect using sigmoidal model
        
        Args:
            concentration: Drug concentration(s) in nM
            parameters: Dose-response parameters
            
        Returns:
            Effect value(s) (0-1 scale)
        """
        C = np.asarray(concentration, dtype=float)
        ec50 = parameters.ec50
        k = parameters.hill_coefficient
        emax = parameters.emax
        baseline = parameters.baseline
        
        # Sigmoidal function
        log_ratio = np.log(np.maximum(C, 1e-12) / ec50)
        sigmoid = expit(k * log_ratio)  # More numerically stable than sigmoid function
        
        effect = baseline + (emax - baseline) * sigmoid
        
        return np.clip(effect, 0, 1)
    
    def inverse_effect(self, effect: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate concentration from effect using sigmoidal model
        
        Args:
            effect: Effect value(s) (0-1 scale)
            parameters: Dose-response parameters
            
        Returns:
            Concentration(s) in nM
        """
        E = np.asarray(effect, dtype=float)
        ec50 = parameters.ec50
        k = parameters.hill_coefficient
        emax = parameters.emax
        baseline = parameters.baseline
        
        # Inverse sigmoidal function
        numerator = E - baseline
        denominator = emax - baseline
        
        ratio = np.divide(numerator, denominator,
                         out=np.zeros_like(numerator), where=denominator!=0)
        
        # Calculate log concentration
        log_concentration = np.log(ec50) + np.log(ratio / (1 - ratio)) / k
        concentration = np.exp(log_concentration)
        
        return np.clip(concentration, 0, np.inf)

class EmaxModel(DoseResponseModel):
    """
    E_max dose-response model
    
    Effect = (E_max * C) / (EC50 + C)
    """
    
    def effect(self, concentration: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate effect using E_max model
        
        Args:
            concentration: Drug concentration(s) in nM
            parameters: Dose-response parameters
            
        Returns:
            Effect value(s) (0-1 scale)
        """
        C = np.asarray(concentration, dtype=float)
        emax = parameters.emax
        ec50 = parameters.ec50
        baseline = parameters.baseline
        
        # E_max equation
        effect = baseline + emax * (C / (ec50 + C))
        
        return np.clip(effect, 0, 1)
    
    def inverse_effect(self, effect: np.ndarray, parameters: DoseResponseParameters) -> np.ndarray:
        """
        Calculate concentration from effect using E_max model
        
        Args:
            effect: Effect value(s) (0-1 scale)
            parameters: Dose-response parameters
            
        Returns:
            Concentration(s) in nM
        """
        E = np.asarray(effect, dtype=float)
        emax = parameters.emax
        ec50 = parameters.ec50
        baseline = parameters.baseline
        
        # Inverse E_max equation
        numerator = E - baseline
        denominator = emax - (E - baseline)
        
        concentration = ec50 * numerator / np.maximum(denominator, 1e-12)
        
        return np.clip(concentration, 0, np.inf)

class DoseResponseFitter:
    """Fitter for dose-response model parameters from experimental data"""
    
    def __init__(self, model_type: str = 'hill'):
        """
        Initialize dose-response fitter
        
        Args:
            model_type: Type of model ('hill', 'sigmoid', 'emax')
        """
        if model_type == 'hill':
            self.model = HillEquationModel()
        elif model_type == 'sigmoid':
            self.model = SigmoidalModel()
        elif model_type == 'emax':
            self.model = EmaxModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.fitted_parameters = None
        self.fit_quality = None
    
    def fit(self, concentrations: np.ndarray, effects: np.ndarray, 
            initial_guess: Optional[DoseResponseParameters] = None) -> DoseResponseParameters:
        """
        Fit dose-response model to experimental data
        
        Args:
            concentrations: Experimental concentrations (nM)
            effects: Experimental effects (0-1 scale)
            initial_guess: Initial parameter guess
            
        Returns:
            Fitted dose-response parameters
        """
        # Sort data by concentration
        sorted_indices = np.argsort(concentrations)
        concentrations = concentrations[sorted_indices]
        effects = effects[sorted_indices]
        
        # Set initial guess if not provided
        if initial_guess is None:
            initial_guess = self._estimate_initial_guess(concentrations, effects)
        
        # Define objective function
        def objective(params):
            try:
                fitted_params = DoseResponseParameters(
                    ic50=params[0],
                    hill_coefficient=params[1],
                    emax=params[2],
                    baseline=params[3],
                    ec50=params[0]  # Use IC50 as EC50 for simplicity
                )
                
                predicted_effects = self.model.effect(concentrations, fitted_params)
                residuals = (predicted_effects - effects) ** 2
                return np.sum(residuals)
            except:
                return np.inf
        
        # Set parameter bounds
        if self.model_type == 'hill':
            bounds = ([1e-6, 0.1, 0.1, 0.0],  # lower bounds
                     [1e6, 10.0, 1.0, 0.5])  # upper bounds
        elif self.model_type == 'sigmoid':
            bounds = ([1e-6, 0.1, 0.1, 0.0],
                     [1e6, 5.0, 1.0, 0.5])
        else:  # emax
            bounds = ([1e-6, 0.1, 0.1, 0.0],
                     [1e6, 2.0, 1.0, 0.5])
        
        # Fit parameters
        initial_params = [
            initial_guess.ic50,
            initial_guess.hill_coefficient,
            initial_guess.emax,
            initial_guess.baseline
        ]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            fitted_params = DoseResponseParameters(
                ic50=result.x[0],
                hill_coefficient=result.x[1],
                emax=result.x[2],
                baseline=result.x[3],
                ec50=result.x[0]
            )
            
            # Calculate fit quality
            predicted_effects = self.model.effect(concentrations, fitted_params)
            r_squared = self._calculate_r_squared(effects, predicted_effects)
            rmse = np.sqrt(np.mean((predicted_effects - effects) ** 2))
            
            self.fitted_parameters = fitted_params
            self.fit_quality = {
                'r_squared': r_squared,
                'rmse': rmse,
                'success': True
            }
            
            logger.info(f"Model fitting successful. R² = {r_squared:.3f}, RMSE = {rmse:.4f}")
            return fitted_params
        else:
            logger.warning(f"Model fitting failed: {result.message}")
            self.fit_quality = {'success': False, 'message': result.message}
            return initial_guess
    
    def _estimate_initial_guess(self, concentrations: np.ndarray, 
                              effects: np.ndarray) -> DoseResponseParameters:
        """Estimate initial parameter guess from data"""
        # Estimate IC50 as concentration at 50% effect
        effect_50_idx = np.argmin(np.abs(effects - 0.5))
        ic50_guess = concentrations[effect_50_idx] if effect_50_idx < len(concentrations) else 100.0
        
        # Estimate Hill coefficient from curve steepness
        log_conc = np.log10(np.maximum(concentrations, 1e-12))
        # Simple estimation based on the range of effects
        effect_range = np.max(effects) - np.min(effects)
        hill_guess = max(1.0, min(4.0, 2.0 / effect_range))
        
        # Estimate E_max and baseline
        emax_guess = max(effects)
        baseline_guess = min(effects)
        
        return DoseResponseParameters(
            ic50=ic50_guess,
            hill_coefficient=hill_guess,
            emax=emax_guess,
            baseline=baseline_guess,
            ec50=ic50_guess
        )
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

class DoseResponseCurve:
    """Container for dose-response curve data and methods"""
    
    def __init__(self, model: DoseResponseModel, parameters: DoseResponseParameters,
                 drug_properties: DrugProperties):
        """
        Initialize dose-response curve
        
        Args:
            model: Dose-response model instance
            parameters: Model parameters
            drug_properties: Drug properties
        """
        self.model = model
        self.parameters = parameters
        self.drug_properties = drug_properties
    
    def calculate_effect(self, concentrations: np.ndarray) -> np.ndarray:
        """Calculate effect at given concentrations"""
        return self.model.effect(concentrations, self.parameters)
    
    def calculate_concentration(self, effects: np.ndarray) -> np.ndarray:
        """Calculate concentration for given effects"""
        return self.model.inverse_effect(effects, self.parameters)
    
    def calculate_ic50(self) -> float:
        """Calculate IC50 (concentration for 50% effect)"""
        return self.parameters.ic50
    
    def calculate_ec50(self) -> float:
        """Calculate EC50 (concentration for 50% effect)"""
        return self.parameters.ec50
    
    def calculate_hill_slope(self) -> float:
        """Calculate Hill slope"""
        return self.parameters.hill_coefficient
    
    def calculate_therapeutic_index(self, td50: float) -> float:
        """
        Calculate therapeutic index (TI = TD50/ED50)
        
        Args:
            td50: Toxic dose for 50% of population
            
        Returns:
            Therapeutic index
        """
        ed50 = self.calculate_ec50()
        return td50 / ed50 if ed50 > 0 else np.inf
    
    def calculate_selectivity_ratio(self, other_curve: 'DoseResponseCurve') -> float:
        """
        Calculate selectivity ratio between two dose-response curves
        
        Args:
            other_curve: Other dose-response curve for comparison
            
        Returns:
            Selectivity ratio (IC50_other / IC50_self)
        """
        return other_curve.calculate_ic50() / self.calculate_ic50()
    
    def generate_full_curve(self, min_conc: float = 1e-3, max_conc: float = 1e4, 
                          n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full dose-response curve
        
        Args:
            min_conc: Minimum concentration (nM)
            max_conc: Maximum concentration (nM)
            n_points: Number of points in curve
            
        Returns:
            Tuple of (concentrations, effects)
        """
        concentrations = np.logspace(np.log10(min_conc), np.log10(max_conc), n_points)
        effects = self.calculate_effect(concentrations)
        return concentrations, effects

class CombinationDoseResponse:
    """Dose-response modeling for drug combinations"""
    
    @staticmethod
    def bliss_independence(curve1: DoseResponseCurve, curve2: DoseResponseCurve,
                         concentrations1: np.ndarray, concentrations2: np.ndarray) -> np.ndarray:
        """
        Calculate combined effect using Bliss independence model
        
        E_combined = E1 + E2 - (E1 * E2)
        
        Args:
            curve1: First drug dose-response curve
            curve2: Second drug dose-response curve
            concentrations1: Concentrations of first drug
            concentrations2: Concentrations of second drug
            
        Returns:
            Combined effects
        """
        effects1 = curve1.calculate_effect(concentrations1[:, np.newaxis])
        effects2 = curve2.calculate_effect(concentrations2[np.newaxis, :])
        
        # Calculate combination matrix
        combined_effects = effects1 + effects2 - (effects1 * effects2)
        return np.clip(combined_effects, 0, 1)
    
    @staticmethod
    def loewe_additivity(curve1: DoseResponseCurve, curve2: DoseResponseCurve,
                        concentrations1: np.ndarray, concentrations2: np.ndarray) -> np.ndarray:
        """
        Calculate combined effect using Loewe additivity model
        
        C1/EC50_1 + C2/EC50_2 = 1 (for additive effect)
        
        Args:
            curve1: First drug dose-response curve
            curve2: Second drug dose-response curve
            concentrations1: Concentrations of first drug
            concentrations2: Concentrations of second drug
            
        Returns:
            Combined effects
        """
        # Create combination matrix
        C1, C2 = np.meshgrid(concentrations1, concentrations2, indexing='ij')
        
        # Calculate equivalent concentrations
        ic50_1 = curve1.calculate_ic50()
        ic50_2 = curve2.calculate_ic50()
        
        # Loewe additivity index
        loewe_index = C1/ic50_1 + C2/ic50_2
        
        # Calculate combined effect
        combined_effects = np.zeros_like(loewe_index)
        
        for i in range(len(concentrations1)):
            for j in range(len(concentrations2)):
                if loewe_index[i, j] <= 1:
                    # Additive effect
                    effect = 0.5 * loewe_index[i, j]
                else:
                    # Supra-additive effect
                    effect = 0.5 + 0.3 * (loewe_index[i, j] - 1)
                
                combined_effects[i, j] = min(effect, 1.0)
        
        return combined_effects
    
    @staticmethod
    def highest_single_agent(curve1: DoseResponseCurve, curve2: DoseResponseCurve,
                           concentrations1: np.ndarray, concentrations2: np.ndarray) -> np.ndarray:
        """
        Calculate combined effect using Highest Single Agent (HSA) model
        
        E_combined = max(E1, E2)
        
        Args:
            curve1: First drug dose-response curve
            curve2: Second drug dose-response curve
            concentrations1: Concentrations of first drug
            concentrations2: Concentrations of second drug
            
        Returns:
            Combined effects
        """
        effects1 = curve1.calculate_effect(concentrations1[:, np.newaxis])
        effects2 = curve2.calculate_effect(concentrations2[np.newaxis, :])
        
        # Calculate combination matrix
        combined_effects = np.maximum(effects1, effects2)
        return combined_effects

class DoseResponseAnalyzer:
    """Analyzer for dose-response relationships and synthetic lethality"""
    
    def __init__(self):
        """Initialize dose-response analyzer"""
        pass
    
    def analyze_synthetic_lethality(self, wt_curve: DoseResponseCurve, 
                                  mutant_curve: DoseResponseCurve) -> Dict:
        """
        Analyze synthetic lethality between two dose-response curves
        
        Args:
            wt_curve: Wild-type (ATM-proficient) dose-response curve
            mutant_curve: Mutant (ATM-deficient) dose-response curve
            
        Returns:
            Dictionary with synthetic lethality metrics
        """
        # Calculate key metrics
        ic50_wt = wt_curve.calculate_ic50()
        ic50_mutant = mutant_curve.calculate_ic50()
        
        # Synthetic lethality ratio
        sl_ratio = ic50_wt / ic50_mutant if ic50_mutant > 0 else np.inf
        
        # Selectivity index
        selectivity_index = sl_ratio
        
        # Calculate concentration for 10% and 90% effect
        effect_levels = [0.1, 0.5, 0.9]
        concentration_ratios = {}
        
        for effect in effect_levels:
            conc_wt = wt_curve.calculate_concentration(np.array([effect]))[0]
            conc_mutant = mutant_curve.calculate_concentration(np.array([effect]))[0]
            
            if conc_wt > 0 and conc_mutant > 0:
                concentration_ratios[f'{int(effect*100)}%_effect'] = conc_wt / conc_mutant
            else:
                concentration_ratios[f'{int(effect*100)}%_effect'] = np.inf
        
        # Therapeutic index calculation
        therapeutic_index = wt_curve.calculate_therapeutic_index(ic50_wt * 10)  # Assume TD50 = 10x IC50
        
        return {
            'ic50_wildtype': ic50_wt,
            'ic50_mutant': ic50_mutant,
            'synthetic_lethality_ratio': sl_ratio,
            'selectivity_index': selectivity_index,
            'concentration_ratios': concentration_ratios,
            'therapeutic_index': therapeutic_index,
            'fold_selectivity': sl_ratio,
            'classification': self._classify_synthetic_lethality(sl_ratio)
        }
    
    def _classify_synthetic_lethality(self, sl_ratio: float) -> str:
        """Classify synthetic lethality based on ratio"""
        if sl_ratio > 10:
            return "Strong synthetic lethality"
        elif sl_ratio > 3:
            return "Moderate synthetic lethality"
        elif sl_ratio > 1.5:
            return "Weak synthetic lethality"
        else:
            return "No synthetic lethality"
    
    def dose_optimization(self, target_effect: float, wt_curve: DoseResponseCurve,
                        mutant_curve: DoseResponseCurve, toxicity_limit: float = 0.3) -> Dict:
        """
        Optimize dose for maximum therapeutic window
        
        Args:
            target_effect: Desired effect level (0-1)
            wt_curve: Wild-type dose-response curve
            mutant_curve: Mutant dose-response curve
            toxicity_limit: Maximum acceptable toxicity in wild-type
            
        Returns:
            Dictionary with optimized dosing recommendations
        """
        # Calculate concentrations for target effect
        conc_mutant = mutant_curve.calculate_concentration(np.array([target_effect]))[0]
        conc_wt = wt_curve.calculate_concentration(np.array([toxicity_limit]))[0]
        
        # Optimization
        if conc_mutant > 0 and conc_wt > 0:
            # Check if therapeutic window exists
            therapeutic_window = conc_wt / conc_mutant
            
            if therapeutic_window > 1:
                # Optimal concentration is below toxicity limit
                optimal_concentration = conc_mutant
                safety_margin = therapeutic_window
            else:
                # No safe therapeutic window
                optimal_concentration = conc_mutant
                safety_margin = 1.0
        else:
            optimal_concentration = 100.0  # Default
            safety_margin = 0.0
        
        return {
            'optimal_concentration': optimal_concentration,
            'safety_margin': safety_margin,
            'therapeutic_window': safety_margin > 1,
            'expected_mutant_effect': target_effect,
            'predicted_wt_toxicity': wt_curve.calculate_effect(np.array([optimal_concentration]))[0],
            'dose_recommendation': self._generate_dose_recommendation(safety_margin, optimal_concentration)
        }
    
    def _generate_dose_recommendation(self, safety_margin: float, concentration: float) -> str:
        """Generate human-readable dose recommendation"""
        if safety_margin > 10:
            return f"Excellent therapeutic window. Recommended dose: {concentration:.1f} nM"
        elif safety_margin > 3:
            return f"Good therapeutic window. Recommended dose: {concentration:.1f} nM (monitor closely)"
        elif safety_margin > 1:
            return f"Narrow therapeutic window. Recommended dose: {concentration:.1f} nM (careful monitoring required)"
        else:
            return f"No safe therapeutic window identified. Consider combination therapy."

# Example usage and testing
if __name__ == "__main__":
    # Test dose-response modeling
    print("Dose-Response Modeling Framework - Test")
    print("=" * 50)
    
    # Create test data
    concentrations = np.logspace(-1, 2, 8)  # 0.1 to 100 nM
    true_params = DoseResponseParameters(
        ic50=10.0,
        hill_coefficient=2.0,
        emax=0.9,
        baseline=0.1,
        ec50=10.0
    )
    
    # Test Hill equation model
    model = HillEquationModel()
    true_effects = model.effect(concentrations, true_params)
    
    # Add some noise
    noisy_effects = true_effects + np.random.normal(0, 0.02, len(true_effects))
    noisy_effects = np.clip(noisy_effects, 0, 1)
    
    print(f"Test concentrations: {concentrations}")
    print(f"True effects: {true_effects}")
    print(f"Noisy effects: {noisy_effects}")
    
    # Fit model
    fitter = DoseResponseFitter('hill')
    fitted_params = fitter.fit(concentrations, noisy_effects)
    
    print(f"\nFitted parameters:")
    print(f"  IC50: {fitted_params.ic50:.2f} (true: {true_params.ic50:.2f})")
    print(f"  Hill coefficient: {fitted_params.hill_coefficient:.2f} (true: {true_params.hill_coefficient:.2f})")
    print(f"  E_max: {fitted_params.emax:.3f} (true: {true_params.emax:.3f})")
    print(f"  Baseline: {fitted_params.baseline:.3f} (true: {true_params.baseline:.3f})")
    
    if fitter.fit_quality:
        print(f"  Fit quality - R²: {fitter.fit_quality['r_squared']:.3f}, RMSE: {fitter.fit_quality['rmse']:.4f}")
    
    # Test synthetic lethality analysis
    print(f"\nSynthetic Lethality Analysis:")
    
    # Create wild-type and mutant curves
    wt_params = DoseResponseParameters(ic50=50.0, hill_coefficient=2.0, emax=0.9, baseline=0.1, ec50=50.0)
    mutant_params = DoseResponseParameters(ic50=5.0, hill_coefficient=2.0, emax=0.9, baseline=0.1, ec50=5.0)
    
    drug_props = DrugProperties("Test Drug", "Target", 500.0, 2.0, 0.8, 0.9, 8.0, 1.2, 1.0)
    
    wt_curve = DoseResponseCurve(model, wt_params, drug_props)
    mutant_curve = DoseResponseCurve(model, mutant_params, drug_props)
    
    analyzer = DoseResponseAnalyzer()
    sl_analysis = analyzer.analyze_synthetic_lethality(wt_curve, mutant_curve)
    
    print(f"  IC50 (WT): {sl_analysis['ic50_wildtype']:.1f} nM")
    print(f"  IC50 (Mutant): {sl_analysis['ic50_mutant']:.1f} nM")
    print(f"  Synthetic lethality ratio: {sl_analysis['synthetic_lethality_ratio']:.1f}")
    print(f"  Classification: {sl_analysis['classification']}")
    
    # Test dose optimization
    optimization = analyzer.dose_optimization(0.8, wt_curve, mutant_curve)
    print(f"\nDose Optimization:")
    print(f"  Optimal concentration: {optimization['optimal_concentration']:.1f} nM")
    print(f"  Safety margin: {optimization['safety_margin']:.1f}")
    print(f"  Recommendation: {optimization['dose_recommendation']}")
    
    print("\nDose-response modeling framework test completed!")