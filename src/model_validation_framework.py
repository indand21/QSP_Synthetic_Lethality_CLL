"""
Model Validation Framework for Enhanced DDR QSP Model
====================================================

This module provides comprehensive validation tools for the DDR QSP model including:
- Experimental data integration
- Statistical validation metrics
- Uncertainty quantification
- Model comparison frameworks
- Cross-validation strategies
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

@dataclass
class ValidationMetrics:
    """Container for model validation metrics"""
    rmse: float
    mae: float
    r_squared: float
    spearman_correlation: float
    pearson_correlation: float
    prediction_interval_coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'r_squared': self.r_squared,
            'spearman_correlation': self.spearman_correlation,
            'pearson_correlation': self.pearson_correlation,
            'prediction_interval_coverage': self.prediction_interval_coverage
        }

class ExperimentalData:
    """Container for experimental validation data"""
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize experimental data container
        
        Args:
            data_file: Path to experimental data file (CSV/JSON)
        """
        self.data = {}
        self.metadata = {}
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, data_file: str):
        """Load experimental data from file"""
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            df = pd.read_json(data_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Parse data structure
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            self.data[condition] = {
                'time': condition_data['time'].values,
                'apoptosis': condition_data['apoptosis'].values,
                'dsb_level': condition_data.get('dsb_level', None),
                'atm_activity': condition_data.get('atm_activity', None),
                'atr_activity': condition_data.get('atr_activity', None),
                'metadata': {
                    'cell_line': condition_data['cell_line'].iloc[0],
                    'drug_treatment': condition_data['drug_treatment'].iloc[0],
                    'concentration': condition_data.get('concentration', [None] * len(condition_data)).iloc[0]
                }
            }
    
    def get_condition_data(self, condition: str) -> Dict:
        """Get data for specific experimental condition"""
        return self.data.get(condition, {})
    
    def add_synthetic_data(self, model, conditions: List[Dict], noise_level: float = 0.1):
        """
        Generate synthetic experimental data from model predictions
        
        Args:
            model: DDR model instance
            conditions: List of experimental conditions
            noise_level: Fractional noise to add to data
        """
        for condition in conditions:
            sim_result = model.run_simulation(
                condition['duration'], 
                condition['drug_effects']
            )
            
            # Add experimental noise
            n_points = len(sim_result)
            noise_factor = noise_level * np.random.normal(0, 1, n_points)
            
            self.data[condition['name']] = {
                'time': sim_result['Time'].values,
                'apoptosis': sim_result['ApoptosisSignal'].values * (1 + noise_factor),
                'dsb_level': sim_result['DSB'].values * (1 + noise_factor),
                'metadata': {
                    'cell_line': condition.get('cell_line', 'synthetic'),
                    'drug_treatment': str(condition['drug_effects']),
                    'synthetic': True
                }
            }

class ModelValidator:
    """Comprehensive model validation framework"""
    
    def __init__(self, model, experimental_data: ExperimentalData):
        """
        Initialize model validator
        
        Args:
            model: DDR model instance to validate
            experimental_data: ExperimentalData instance
        """
        self.model = model
        self.experimental_data = experimental_data
        self.validation_results = {}
        self.uncertainty_results = {}
    
    def calculate_validation_metrics(self, simulated_data: pd.DataFrame, 
                                   experimental_condition: str) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics
        
        Args:
            simulated_data: Model simulation results
            experimental_condition: Experimental condition name
            
        Returns:
            ValidationMetrics object
        """
        exp_data = self.experimental_data.get_condition_data(experimental_condition)
        
        if not exp_data:
            raise ValueError(f"No experimental data found for condition: {experimental_condition}")
        
        # Convert to numpy arrays
        exp_time = np.array(exp_data['time'], dtype=float)
        exp_apoptosis = np.array(exp_data['apoptosis'], dtype=float)
        sim_time = np.array(simulated_data['Time'], dtype=float)
        sim_apoptosis = np.array(simulated_data['ApoptosisSignal'], dtype=float)
        
        # Interpolate experimental data
        exp_interp = np.interp(sim_time, exp_time, exp_apoptosis)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(exp_interp, sim_apoptosis))
        mae = mean_absolute_error(exp_interp, sim_apoptosis)
        r_squared = r2_score(exp_interp, sim_apoptosis)
        
        # Correlation metrics (simplified approach)
        spearman_corr = np.corrcoef(exp_interp, sim_apoptosis)[0, 1]
        pearson_corr = np.corrcoef(exp_interp, sim_apoptosis)[0, 1]
        
        # Prediction interval coverage (95% interval)
        residuals = exp_interp - sim_apoptosis
        residual_std = np.std(residuals)
        lower_bound = sim_apoptosis - 1.96 * residual_std
        upper_bound = sim_apoptosis + 1.96 * residual_std
        
        within_interval = np.sum((exp_interp >= lower_bound) & (exp_interp <= upper_bound))
        prediction_coverage = within_interval / len(exp_interp)
        
        return ValidationMetrics(
            rmse=float(rmse),
            mae=float(mae),
            r_squared=float(r_squared),
            spearman_correlation=spearman_corr,
            pearson_correlation=pearson_corr,
            prediction_interval_coverage=float(prediction_coverage)
        )
    
    def validate_all_conditions(self, conditions: List[str]) -> Dict:
        """
        Validate model against all experimental conditions
        
        Args:
            conditions: List of experimental condition names
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for condition in conditions:
            try:
                # Get experimental condition
                exp_data = self.experimental_data.get_condition_data(condition)
                if not exp_data:
                    continue
                
                # Run simulation with matching parameters
                drug_effects = self._extract_drug_effects(exp_data['metadata'])
                simulated = self.model.run_simulation(48, drug_effects)
                
                # Calculate metrics
                metrics = self.calculate_validation_metrics(simulated, condition)
                validation_results[condition] = metrics.to_dict()
                
            except Exception as e:
                print(f"Validation failed for condition {condition}: {e}")
                continue
        
        self.validation_results = validation_results
        return validation_results
    
    def _extract_drug_effects(self, metadata: Dict) -> Dict:
        """Extract drug effects from experimental metadata"""
        # This is a simplified extraction - implement based on your data structure
        drug_effects = {}
        drug_treatment = metadata.get('drug_treatment', '{}')
        
        # Parse drug treatment string (implement proper parsing)
        if 'ATR' in drug_treatment:
            drug_effects['ATR'] = 0.9
        if 'PARP' in drug_treatment:
            drug_effects['PARP'] = 0.9
        if 'CHK1' in drug_treatment:
            drug_effects['CHK1'] = 0.9
        
        return drug_effects
    
    def plot_validation_results(self, save_path: str = "validation_plots.png"):
        """
        Create comprehensive validation plots
        
        Args:
            save_path: Path to save validation plots
        """
        if not self.validation_results:
            print("No validation results to plot. Run validate_all_conditions() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Validation Results', fontsize=16)
        
        # Extract metrics for plotting
        conditions = list(self.validation_results.keys())
        rmse_values = [self.validation_results[c]['rmse'] for c in conditions]
        mae_values = [self.validation_results[c]['mae'] for c in conditions]
        r2_values = [self.validation_results[c]['r_squared'] for c in conditions]
        correlation_values = [self.validation_results[c]['pearson_correlation'] for c in conditions]
        
        # RMSE plot
        axes[0, 0].bar(conditions, rmse_values, color='skyblue')
        axes[0, 0].set_title('Root Mean Square Error')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE plot
        axes[0, 1].bar(conditions, mae_values, color='lightcoral')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² plot
        axes[1, 0].bar(conditions, r2_values, color='lightgreen')
        axes[1, 0].set_title('R-squared Score')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good fit threshold')
        axes[1, 0].legend()
        
        # Correlation plot
        axes[1, 1].bar(conditions, correlation_values, color='gold')
        axes[1, 1].set_title('Pearson Correlation')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Strong correlation')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validate_model(self, n_splits: int = 5) -> Dict:
        """
        Perform cross-validation using model predictions
        
        Args:
            n_splits: Number of CV splits
            
        Returns:
            Cross-validation results
        """
        # This is a simplified cross-validation approach
        # In practice, you'd need a more sophisticated method for ODE models
        
        conditions = list(self.experimental_data.data.keys())
        if len(conditions) < n_splits:
            print(f"Not enough conditions ({len(conditions)}) for {n_splits}-fold CV")
            return {}
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(conditions):
            train_conditions = [conditions[i] for i in train_idx]
            test_conditions = [conditions[i] for i in test_idx]
            
            # Train on subset (simplified)
            fold_scores = []
            for condition in test_conditions:
                exp_data = self.experimental_data.get_condition_data(condition)
                if not exp_data:
                    continue
                
                # Simulate (in real CV, you'd retrain parameters)
                drug_effects = self._extract_drug_effects(exp_data['metadata'])
                simulated = self.model.run_simulation(48, drug_effects)
                
                # Calculate validation metrics
                metrics = self.calculate_validation_metrics(simulated, condition)
                fold_scores.append(metrics.r_squared)
            
            if fold_scores:
                cv_scores.append(np.mean(fold_scores))
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
        
        return cv_results

class UncertaintyQuantifier:
    """Uncertainty quantification for DDR model predictions"""
    
    def __init__(self, model, n_bootstrap: int = 1000):
        """
        Initialize uncertainty quantifier
        
        Args:
            model: DDR model instance
            n_bootstrap: Number of bootstrap samples
        """
        self.model = model
        self.n_bootstrap = n_bootstrap
        self.uncertainty_results = {}
    
    def parameter_uncertainty_analysis(self, parameter_bounds: Dict, 
                                     experimental_data: Dict) -> Dict:
        """
        Perform parameter uncertainty analysis using bootstrap
        
        Args:
            parameter_bounds: Dictionary of parameter bounds
            experimental_data: Experimental validation data
            
        Returns:
            Parameter uncertainty results
        """
        bootstrap_results = []
        
        print(f"Running {self.n_bootstrap} bootstrap samples...")
        for i in range(self.n_bootstrap):
            if i % 100 == 0:
                print(f"Bootstrap sample {i}/{self.n_bootstrap}")
            
            # Sample parameters
            sampled_params = {}
            for param_name, (lower, upper) in parameter_bounds.items():
                sampled_params[param_name] = np.random.uniform(lower, upper)
            
            # Update model parameters
            self.model.update_parameters(sampled_params)
            
            # Run simulation
            try:
                simulation = self.model.run_simulation(48)
                final_apoptosis = simulation['ApoptosisSignal'].iloc[-1]
                bootstrap_results.append({
                    'sample': i,
                    'parameters': sampled_params,
                    'final_apoptosis': final_apoptosis
                })
            except:
                continue
        
        # Analyze results
        final_apoptoses = [result['final_apoptosis'] for result in bootstrap_results]
        
        uncertainty_results = {
            'bootstrap_samples': len(bootstrap_results),
            'mean_apoptosis': np.mean(final_apoptoses),
            'std_apoptosis': np.std(final_apoptoses),
            'ci_95_lower': np.percentile(final_apoptoses, 2.5),
            'ci_95_upper': np.percentile(final_apoptoses, 97.5),
            'parameter_sensitivity': self._calculate_parameter_sensitivity(bootstrap_results),
            'all_samples': bootstrap_results
        }
        
        self.uncertainty_results = uncertainty_results
        return uncertainty_results
    
    def _calculate_parameter_sensitivity(self, bootstrap_results: List[Dict]) -> Dict:
        """Calculate parameter sensitivity from bootstrap results"""
        if not bootstrap_results:
            return {}
        
        # Extract parameter correlations with outcome
        param_names = list(bootstrap_results[0]['parameters'].keys())
        param_values = {name: [] for name in param_names}
        outcomes = []
        
        for result in bootstrap_results:
            outcomes.append(result['final_apoptosis'])
            for param_name in param_names:
                param_values[param_name].append(result['parameters'][param_name])
        
        # Calculate correlations
        sensitivity = {}
        for param_name in param_names:
            correlation = np.corrcoef(param_values[param_name], outcomes)[0, 1]
            sensitivity[param_name] = {
                'correlation': correlation,
                'importance': abs(correlation)
            }
        
        return sensitivity
    
    def plot_uncertainty_results(self, save_path: str = "uncertainty_analysis.png"):
        """
        Plot uncertainty analysis results
        
        Args:
            save_path: Path to save uncertainty plots
        """
        if not self.uncertainty_results:
            print("No uncertainty results to plot. Run parameter_uncertainty_analysis() first.")
            return
        
        bootstrap_results = self.uncertainty_results['all_samples']
        final_apoptoses = [result['final_apoptosis'] for result in bootstrap_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Uncertainty Analysis', fontsize=16)
        
        # Histogram of final apoptosis values
        axes[0, 0].hist(final_apoptoses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.uncertainty_results['mean_apoptosis'], 
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(self.uncertainty_results['ci_95_lower'], 
                          color='orange', linestyle='--', label='95% CI')
        axes[0, 0].axvline(self.uncertainty_results['ci_95_upper'], 
                          color='orange', linestyle='--')
        axes[0, 0].set_title('Distribution of Final Apoptosis')
        axes[0, 0].set_xlabel('Final Apoptosis Signal')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Parameter sensitivity
        sensitivity = self.uncertainty_results['parameter_sensitivity']
        param_names = list(sensitivity.keys())
        importances = [sensitivity[p]['importance'] for p in param_names]
        
        axes[0, 1].barh(param_names, importances, color='lightcoral')
        axes[0, 1].set_title('Parameter Importance')
        axes[0, 1].set_xlabel('Absolute Correlation with Outcome')
        
        # Time series with uncertainty bands
        # This would require re-running simulations for selected parameter sets
        axes[1, 0].set_title('Prediction Uncertainty Over Time')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Apoptosis Signal')
        
        # Correlation matrix heatmap
        # Extract parameter correlations
        param_matrix = []
        param_names = list(sensitivity.keys())
        
        for result in bootstrap_results[:min(100, len(bootstrap_results))]:  # Sample for performance
            param_row = [result['parameters'][name] for name in param_names]
            param_matrix.append(param_row)
        
        param_matrix = np.array(param_matrix)
        correlation_matrix = np.corrcoef(param_matrix.T)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Parameter Correlation Matrix')
        axes[1, 1].set_xticks(range(len(param_names)))
        axes[1, 1].set_yticks(range(len(param_names)))
        axes[1, 1].set_xticklabels(param_names, rotation=45)
        axes[1, 1].set_yticklabels(param_names)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == '__main__':
    # Import the enhanced model
    from enhanced_ddr_qsp_model import EnhancedDDRModel
    
    print("DDR QSP Model Validation Framework - Example Usage")
    print("=" * 60)
    
    # Initialize model and experimental data
    model = EnhancedDDRModel(atm_proficient=True)
    exp_data = ExperimentalData()
    
    # Create synthetic experimental data
    synthetic_conditions = [
        {
            'name': 'control',
            'duration': 48,
            'drug_effects': {},
            'cell_line': 'MEC1'
        },
        {
            'name': 'atr_inhibitor',
            'duration': 48,
            'drug_effects': {'ATR': 0.9},
            'cell_line': 'MEC1'
        },
        {
            'name': 'parp_inhibitor',
            'duration': 48,
            'drug_effects': {'PARP': 0.9},
            'cell_line': 'MEC1'
        }
    ]
    
    exp_data.add_synthetic_data(model, synthetic_conditions, noise_level=0.1)
    
    # Initialize validator
    validator = ModelValidator(model, exp_data)
    
    # Run validation
    print("Running model validation...")
    validation_results = validator.validate_all_conditions(['control', 'atr_inhibitor', 'parp_inhibitor'])
    
    # Print validation summary
    print("\nValidation Results Summary:")
    print("-" * 40)
    for condition, metrics in validation_results.items():
        print(f"{condition}: R² = {metrics['r_squared']:.3f}, RMSE = {metrics['rmse']:.3f}")
    
    # Plot validation results
    validator.plot_validation_results("validation_results.png")
    
    # Cross-validation
    print("\nRunning cross-validation...")
    cv_results = validator.cross_validate_model(n_splits=3)
    if cv_results:
        print(f"Cross-validation R²: {cv_results['mean_cv_score']:.3f} ± {cv_results['std_cv_score']:.3f}")
    
    # Uncertainty quantification
    print("\nRunning uncertainty analysis...")
    uncertainty = UncertaintyQuantifier(model, n_bootstrap=100)
    
    # Define parameter bounds for uncertainty analysis
    param_bounds = {
        'k_atr_act': (0.4, 1.2),
        'k_chk1_act_by_atr': (0.6, 1.8),
        'k_apoptosis_damage': (0.05, 0.15),
        'k_dsb_repair_hr': (0.4, 1.2)
    }
    
    uncertainty_results = uncertainty.parameter_uncertainty_analysis(param_bounds, {})
    
    print(f"Parameter Uncertainty Results:")
    print(f"Mean Apoptosis: {uncertainty_results['mean_apoptosis']:.3f}")
    print(f"95% CI: [{uncertainty_results['ci_95_lower']:.3f}, {uncertainty_results['ci_95_upper']:.3f}]")
    
    # Plot uncertainty results
    uncertainty.plot_uncertainty_results("uncertainty_analysis.png")
    
    print("\nValidation and uncertainty analysis complete!")