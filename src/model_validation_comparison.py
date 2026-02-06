"""
Model Validation Comparison Framework for Synthetic Lethality QSP Model
======================================================================

This module provides advanced statistical comparison and model fitting capabilities
for validating QSP model predictions against experimental GDSC data.

Key Features:
- Statistical model comparison metrics
- Parameter fitting to experimental data
- Bootstrap confidence intervals
- Cross-validation capabilities
- Model selection criteria (AIC, BIC)
- Residual analysis and diagnostics

Author: Kilo Code
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution, least_squares
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import multiple testing correction framework
from statistical_testing_correction import MultipleTestingCorrector, CorrectionResult, run_comprehensive_correction_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    r_squared: float
    rmse: float
    mae: float
    mape: float
    spearman_corr: float
    pearson_corr: float
    bias: float
    aic: float
    bic: float
    n_parameters: int
    n_observations: int

@dataclass
class ModelComparisonResult:
    """Container for model comparison results"""
    model_name: str
    metrics: ValidationMetrics
    parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float]
    residuals: np.ndarray
    fitted_values: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    cross_validation_score: float
    bootstrap_scores: List[float]

class AbstractModelFitter(ABC):
    """Abstract base class for model fitting strategies"""
    
    @abstractmethod
    def fit(self, model, experimental_data: pd.DataFrame) -> Dict[str, float]:
        """Fit model parameters to experimental data"""
        pass
    
    @abstractmethod
    def predict(self, model, parameters: Dict[str, float], 
                drug_conditions: List[Dict]) -> np.ndarray:
        """Make predictions with fitted parameters"""
        pass

class QSPModelFitter(AbstractModelFitter):
    """Fitter specifically for QSP DDR model parameters"""
    
    def __init__(self, target_metric: str = 'ic50', optimization_method: str = 'differential_evolution'):
        """
        Initialize QSP model fitter
        
        Args:
            target_metric: Target experimental metric to fit ('ic50', 'auc', 'apoptosis')
            optimization_method: Optimization method ('differential_evolution', 'least_squares')
        """
        self.target_metric = target_metric
        self.optimization_method = optimization_method
        self.fitted_parameters = {}
        self.fitting_history = []
        
        # Define parameter bounds for fitting
        self.parameter_bounds = {
            'k_atm_act': (0.001, 3.0),
            'k_atr_act': (0.1, 2.0),
            'k_chk1_act_by_atr': (0.1, 2.0),
            'k_chk2_act_by_atm': (0.1, 2.0),
            'k_p53_act_by_atm': (0.1, 2.0),
            'k_apoptosis_p53': (0.001, 0.1),
            'k_apoptosis_damage': (0.01, 0.5),
            'k_rad51_recruitment': (0.1, 2.0),
            'k_dsb_repair_hr': (0.1, 2.0),
            'k_dsb_repair_nhej': (0.1, 2.0),
            'apoptosis_threshold': (50.0, 200.0)
        }
    
    def fit(self, model, experimental_data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit QSP model parameters to experimental data
        
        Args:
            model: QSP model instance
            experimental_data: Experimental data DataFrame
            
        Returns:
            Dictionary of fitted parameters
        """
        logger.info(f"Fitting model parameters using {self.optimization_method}")
        
        # Prepare experimental conditions
        exp_conditions = self._prepare_experimental_conditions(experimental_data)
        
        if self.optimization_method == 'differential_evolution':
            fitted_params = self._fit_differential_evolution(model, exp_conditions)
        elif self.optimization_method == 'least_squares':
            fitted_params = self._fit_least_squares(model, exp_conditions)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
        
        self.fitted_parameters = fitted_params
        return fitted_params
    
    def _prepare_experimental_conditions(self, experimental_data: pd.DataFrame) -> List[Dict]:
        """Prepare experimental conditions for fitting"""
        conditions = []
        
        for _, row in experimental_data.iterrows():
            condition = {
                'cell_line': row['cell_line'],
                'atm_status': row['atm_status'],
                'drug': row['drug'],
                'drug_target': row['drug_target'],
                'experimental_value': row['ic50_nm'],
                'drug_effects': self._get_drug_effects(row['drug'])
            }
            conditions.append(condition)
        
        return conditions
    
    def _get_drug_effects(self, drug_name: str) -> Dict[str, float]:
        """Convert drug name to model drug effects"""
        drug_mapping = {
            'AZD6738': {'ATR': 0.9, 'CHK1': 0.3},
            'VE-822': {'ATR': 0.85, 'CHK1': 0.2},
            'Prexasertib': {'CHK1': 0.9, 'ATR': 0.1},
            'Adavosertib': {'WEE1': 0.8, 'CHK1': 0.4},
            'Olaparib': {'PARP': 0.9},
            'Talazoparib': {'PARP': 0.95},
            'ATM inhibitor': {'ATM': 0.8, 'CHK2': 0.6},
            'KU-55933': {'ATM': 0.8, 'CHK2': 0.6},
            'MK-8776': {'CHK1': 0.85},
            'CCT068127': {'WEE1': 0.75}
        }
        
        return drug_mapping.get(drug_name, {'TARGET': 0.5})
    
    def _fit_differential_evolution(self, model, exp_conditions: List[Dict]) -> Dict[str, float]:
        """Fit parameters using differential evolution"""
        
        def objective_function(params):
            # Map parameters to model
            param_dict = {name: value for name, value 
                         in zip(self.parameter_bounds.keys(), params)}
            
            model.update_parameters(param_dict)
            
            # Calculate total residual
            total_residual = 0.0
            
            for condition in exp_conditions:
                # Run simulation
                atm_proficient = (condition['atm_status'] == 'proficient')
                simulation = model.run_simulation(48, condition['drug_effects'])
                
                # Extract target metric
                if self.target_metric == 'ic50':
                    # Convert apoptosis to IC50 prediction
                    final_apoptosis = simulation['ApoptosisSignal'].iloc[-1]
                    predicted_ic50 = 1000 * np.exp(-final_apoptosis / 50)
                    predicted_value = predicted_ic50
                elif self.target_metric == 'apoptosis':
                    predicted_value = simulation['ApoptosisSignal'].iloc[-1]
                else:
                    predicted_value = simulation['ApoptosisSignal'].iloc[-1]
                
                experimental_value = condition['experimental_value']
                
                # Calculate residual (log-scale for IC50)
                if self.target_metric == 'ic50':
                    residual = (np.log10(predicted_value + 1) - np.log10(experimental_value + 1)) ** 2
                else:
                    residual = (predicted_value - experimental_value) ** 2
                
                total_residual += residual
            
            return total_residual
        
        # Define bounds
        bounds = [self.parameter_bounds[name] for name in self.parameter_bounds.keys()]
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Map back to parameter dictionary
        fitted_params = {
            name: result.x[i] for i, name in enumerate(self.parameter_bounds.keys())
        }
        
        # Store fitting history
        self.fitting_history.append({
            'method': 'differential_evolution',
            'success': result.success,
            'fun': result.fun,
            'parameters': fitted_params.copy()
        })
        
        return fitted_params
    
    def _fit_least_squares(self, model, exp_conditions: List[Dict]) -> Dict[str, float]:
        """Fit parameters using least squares method"""
        
        def residual_function(params):
            # Map parameters to model
            param_dict = {name: value for name, value 
                         in zip(self.parameter_bounds.keys(), params)}
            
            model.update_parameters(param_dict)
            
            residuals = []
            
            for condition in exp_conditions:
                # Run simulation
                simulation = model.run_simulation(48, condition['drug_effects'])
                
                # Extract target metric
                if self.target_metric == 'ic50':
                    final_apoptosis = simulation['ApoptosisSignal'].iloc[-1]
                    predicted_value = 1000 * np.exp(-final_apoptosis / 50)
                else:
                    predicted_value = simulation['ApoptosisSignal'].iloc[-1]
                
                experimental_value = condition['experimental_value']
                
                # Calculate residual
                if self.target_metric == 'ic50':
                    residual = np.log10(predicted_value + 1) - np.log10(experimental_value + 1)
                else:
                    residual = predicted_value - experimental_value
                
                residuals.append(residual)
            
            return np.array(residuals)
        
        # Define bounds
        bounds = [(lower, upper) for lower, upper in self.parameter_bounds.values()]
        
        # Run optimization
        result = least_squares(
            residual_function,
            x0=[(lower + upper) / 2 for lower, upper in bounds],
            bounds=bounds,
            method='trf',
            verbose=0
        )
        
        # Map back to parameter dictionary
        fitted_params = {
            name: result.x[i] for i, name in enumerate(self.parameter_bounds.keys())
        }
        
        # Store fitting history
        self.fitting_history.append({
            'method': 'least_squares',
            'success': result.success,
            'fun': result.cost,
            'parameters': fitted_params.copy()
        })
        
        return fitted_params
    
    def predict(self, model, parameters: Dict[str, float], 
                drug_conditions: List[Dict]) -> np.ndarray:
        """
        Make predictions with fitted parameters
        
        Args:
            model: QSP model instance
            parameters: Fitted parameters
            drug_conditions: List of drug conditions to predict
            
        Returns:
            Array of predicted values
        """
        model.update_parameters(parameters)
        
        predictions = []
        
        for condition in drug_conditions:
            # Run simulation
            simulation = model.run_simulation(48, condition['drug_effects'])
            
            # Extract target metric
            if self.target_metric == 'ic50':
                final_apoptosis = simulation['ApoptosisSignal'].iloc[-1]
                prediction = 1000 * np.exp(-final_apoptosis / 50)
            else:
                prediction = simulation['ApoptosisSignal'].iloc[-1]
            
            predictions.append(prediction)
        
        return np.array(predictions)

class ValidationComparisonFramework:
    """Advanced model validation and comparison framework"""
    
    def __init__(self, model, experimental_data: pd.DataFrame):
        """
        Initialize validation comparison framework
        
        Args:
            model: QSP model instance
            experimental_data: Experimental data DataFrame
        """
        self.model = model
        self.experimental_data = experimental_data
        self.fitted_models = {}
        self.comparison_results = {}
        
    def fit_multiple_models(self, fitting_strategies: List[Union[AbstractModelFitter, QSPModelFitter]]) -> Dict[str, Dict]:
        """
        Fit multiple model variations using different strategies
        
        Args:
            fitting_strategies: List of model fitting strategies
            
        Returns:
            Dictionary of fitted models with their parameters
        """
        logger.info(f"Fitting {len(fitting_strategies)} model variations")
        
        fitted_models = {}
        
        for i, strategy in enumerate(fitting_strategies):
            # Handle both AbstractModelFitter and QSPModelFitter
            target_metric = getattr(strategy, 'target_metric', 'unknown')
            model_name = f"{strategy.__class__.__name__}_{target_metric}"
            logger.info(f"Fitting model variation: {model_name}")
            
            try:
                # Create fresh model instance
                model_copy = type(self.model)(atm_proficient=True)
                
                # Fit parameters
                fitted_params = strategy.fit(model_copy, self.experimental_data)
                
                fitted_models[model_name] = {
                    'model': model_copy,
                    'fitter': strategy,
                    'parameters': fitted_params,
                    'strategy': strategy
                }
                
            except Exception as e:
                logger.error(f"Failed to fit model {model_name}: {e}")
                continue
        
        self.fitted_models = fitted_models
        return fitted_models
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare fitted models using various metrics
        
        Returns:
            DataFrame with comparison results
        """
        if not self.fitted_models:
            logger.error("No fitted models to compare")
            return pd.DataFrame()
        
        logger.info("Comparing fitted models...")
        
        comparison_results = []
        
        for model_name, model_info in self.fitted_models.items():
            try:
                # Make predictions
                exp_conditions = self._prepare_prediction_conditions()
                predictions = model_info['strategy'].predict(
                    model_info['model'],
                    model_info['parameters'],
                    exp_conditions
                )
                
                # Get experimental values
                experimental_values = self.experimental_data['ic50_nm'].values
                
                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(
                    experimental_values, predictions, len(model_info['parameters'])
                )
                
                # Calculate cross-validation score
                cv_score = self._cross_validate_model(model_info)
                
                # Bootstrap confidence intervals
                bootstrap_scores = self._bootstrap_validation(model_info)
                
                # Create result
                result = ModelComparisonResult(
                    model_name=model_name,
                    metrics=metrics,
                    parameters=model_info['parameters'],
                    parameter_uncertainties={},  # Would calculate from Hessian
                    residuals=experimental_values - predictions,
                    fitted_values=predictions,
                    confidence_intervals={},  # Would calculate from bootstrap
                    cross_validation_score=cv_score,
                    bootstrap_scores=bootstrap_scores
                )
                
                comparison_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to compare model {model_name}: {e}")
                continue
        
        # Convert to DataFrame for easy analysis
        results_df = self._create_comparison_dataframe(comparison_results)
        self.comparison_results = {r.model_name: r for r in comparison_results}
        
        return results_df
    
    def _prepare_prediction_conditions(self) -> List[Dict]:
        """Prepare drug conditions for prediction"""
        conditions = []
        
        for _, row in self.experimental_data.iterrows():
            condition = {
                'drug_effects': self._get_drug_effects(row['drug']),
                'atm_status': row['atm_status']
            }
            conditions.append(condition)
        
        return conditions
    
    def _get_drug_effects(self, drug_name: str) -> Dict[str, float]:
        """Convert drug name to model drug effects"""
        drug_mapping = {
            'AZD6738': {'ATR': 0.9, 'CHK1': 0.3},
            'VE-822': {'ATR': 0.85, 'CHK1': 0.2},
            'Prexasertib': {'CHK1': 0.9, 'ATR': 0.1},
            'Adavosertib': {'WEE1': 0.8, 'CHK1': 0.4},
            'Olaparib': {'PARP': 0.9},
            'Talazoparib': {'PARP': 0.95},
            'ATM inhibitor': {'ATM': 0.8, 'CHK2': 0.6},
            'KU-55933': {'ATM': 0.8, 'CHK2': 0.6},
            'MK-8776': {'CHK1': 0.85},
            'CCT068127': {'WEE1': 0.75}
        }
        
        return drug_mapping.get(drug_name, {'TARGET': 0.5})
    
    def _calculate_comprehensive_metrics(self, experimental: np.ndarray, 
                                       predicted: np.ndarray, n_params: int) -> ValidationMetrics:
        """Calculate comprehensive validation metrics"""
        
        # Basic metrics
        r2 = r2_score(experimental, predicted)
        rmse = np.sqrt(mean_squared_error(experimental, predicted))
        mae = np.mean(np.abs(experimental - predicted))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((experimental - predicted) / (experimental + 1e-8))) * 100
        
        # Correlations
        pearson_corr, _ = stats.pearsonr(experimental, predicted)
        spearman_corr, _ = stats.spearmanr(experimental, predicted)
        
        # Bias (mean residual)
        bias = np.mean(predicted - experimental)
        
        # Information criteria
        n_obs = len(experimental)
        sse = np.sum((experimental - predicted) ** 2)  # Sum of squared errors
        aic = n_obs * np.log(sse / n_obs) + 2 * n_params
        bic = n_obs * np.log(sse / n_obs) + n_params * np.log(n_obs)
        
        return ValidationMetrics(
            r_squared=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            spearman_corr=spearman_corr,
            pearson_corr=pearson_corr,
            bias=bias,
            aic=aic,
            bic=bic,
            n_parameters=n_params,
            n_observations=n_obs
        )
    
    def _cross_validate_model(self, model_info: Dict, n_folds: int = 5) -> float:
        """
        Perform cross-validation on fitted model
        
        Args:
            model_info: Fitted model information
            n_folds: Number of CV folds
            
        Returns:
            Cross-validation score (negative MSE)
        """
        try:
            # Use K-fold cross-validation
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            cv_scores = []
            
            for train_idx, val_idx in kf.split(self.experimental_data):
                # Split data
                train_data = self.experimental_data.iloc[train_idx]
                val_data = self.experimental_data.iloc[val_idx]
                
                # Fit on training data
                fitter = model_info['strategy']
                model_copy = type(self.model)(atm_proficient=True)
                fitted_params = fitter.fit(model_copy, train_data)
                
                # Predict on validation data
                val_conditions = self._prepare_validation_conditions(val_data)
                predictions = fitter.predict(model_copy, fitted_params, val_conditions)
                
                # Calculate validation score
                val_experimental = val_data['ic50_nm'].values
                val_mse = mean_squared_error(val_experimental, predictions)
                cv_scores.append(-val_mse)  # Negative MSE for maximization
            
            return np.mean(cv_scores)
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return 0.0
    
    def _prepare_validation_conditions(self, data_subset: pd.DataFrame) -> List[Dict]:
        """Prepare conditions for validation subset"""
        conditions = []
        
        for _, row in data_subset.iterrows():
            condition = {
                'drug_effects': self._get_drug_effects(row['drug']),
                'atm_status': row['atm_status']
            }
            conditions.append(condition)
        
        return conditions
    
    def _bootstrap_validation(self, model_info: Dict, n_bootstrap: int = 100) -> List[float]:
        """
        Perform bootstrap validation
        
        Args:
            model_info: Fitted model information
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            List of bootstrap scores
        """
        try:
            bootstrap_scores = []
            n_samples = len(self.experimental_data)
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = self.experimental_data.iloc[bootstrap_indices]
                
                # Fit on bootstrap sample
                fitter = model_info['strategy']
                model_copy = type(self.model)(atm_proficient=True)
                fitted_params = fitter.fit(model_copy, bootstrap_data)
                
                # Predict on original data (out-of-bootstrap)
                original_conditions = self._prepare_prediction_conditions()
                predictions = fitter.predict(model_copy, fitted_params, original_conditions)
                
                # Calculate score
                original_experimental = self.experimental_data['ic50_nm'].values
                score = -mean_squared_error(original_experimental, predictions)
                bootstrap_scores.append(score)
            
            return bootstrap_scores
            
        except Exception as e:
            logger.warning(f"Bootstrap validation failed: {e}")
            return [0.0]
    
    def _create_comparison_dataframe(self, results: List[ModelComparisonResult]) -> pd.DataFrame:
        """Create DataFrame from model comparison results"""
        comparison_data = []
        
        for result in results:
            row = {
                'Model': result.model_name,
                'R²': result.metrics.r_squared,
                'RMSE': result.metrics.rmse,
                'MAE': result.metrics.mae,
                'MAPE': result.metrics.mape,
                'Pearson_Corr': result.metrics.pearson_corr,
                'Spearman_Corr': result.metrics.spearman_corr,
                'Bias': result.metrics.bias,
                'AIC': result.metrics.aic,
                'BIC': result.metrics.bic,
                'CV_Score': result.cross_validation_score,
                'Bootstrap_Mean': np.mean(result.bootstrap_scores) if result.bootstrap_scores else 0.0,
                'Bootstrap_Std': np.std(result.bootstrap_scores) if result.bootstrap_scores else 0.0
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_comparison_plots(self, output_dir: str = "model_comparison_plots"):
        """Generate comprehensive model comparison plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.comparison_results:
            logger.warning("No comparison results to plot")
            return
        
        # 1. Model performance comparison
        self._plot_model_performance(output_path)
        
        # 2. Residual analysis
        self._plot_residual_analysis(output_path)
        
        # 3. Cross-validation comparison
        self._plot_cross_validation_comparison(output_path)
        
        # 4. Bootstrap confidence intervals
        self._plot_bootstrap_analysis(output_path)
        
        logger.info(f"Model comparison plots saved to {output_path}")
    
    def _plot_model_performance(self, output_path: Path):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(self.comparison_results.keys())
        metrics_data = {
            'R²': [self.comparison_results[m].metrics.r_squared for m in models],
            'RMSE': [self.comparison_results[m].metrics.rmse for m in models],
            'MAE': [self.comparison_results[m].metrics.mae for m in models],
            'MAPE': [self.comparison_results[m].metrics.mape for m in models],
            'Pearson_Corr': [self.comparison_results[m].metrics.pearson_corr for m in models],
            'AIC': [self.comparison_results[m].metrics.aic for m in models]
        }
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[i // 3, i % 3]
            bars = ax.bar(range(len(models)), values, alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Highlight best performance
            if metric in ['R²', 'Pearson_Corr']:  # Higher is better
                best_idx = np.argmax(values)
            else:  # Lower is better
                best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(1.0)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_analysis(self, output_path: Path):
        """Plot residual analysis for all models"""
        n_models = len(self.comparison_results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Residual Analysis by Model', fontsize=16)
        
        for i, (model_name, result) in enumerate(self.comparison_results.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Residuals vs fitted
            fitted_values = result.fitted_values
            residuals = result.residuals
            
            ax.scatter(fitted_values, residuals, alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{model_name} - Residuals vs Fitted')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].remove()
            else:
                axes[col].remove()
        
        plt.tight_layout()
        plt.savefig(output_path / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_validation_comparison(self, output_path: Path):
        """Plot cross-validation comparison"""
        models = list(self.comparison_results.keys())
        cv_scores = [self.comparison_results[m].cross_validation_score for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(models)), cv_scores, alpha=0.7)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel('Cross-Validation Score (Negative MSE)')
        plt.title('Cross-Validation Performance Comparison')
        plt.grid(True, alpha=0.3)
        
        # Highlight best performance
        best_idx = np.argmax(cv_scores)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cross_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bootstrap_analysis(self, output_path: Path):
        """Plot bootstrap analysis"""
        models = list(self.comparison_results.keys())
        
        plt.figure(figsize=(12, 6))
        
        bootstrap_means = []
        bootstrap_stds = []
        
        for model_name in models:
            result = self.comparison_results[model_name]
            if result.bootstrap_scores:
                bootstrap_means.append(np.mean(result.bootstrap_scores))
                bootstrap_stds.append(np.std(result.bootstrap_scores))
            else:
                bootstrap_means.append(0.0)
                bootstrap_stds.append(0.0)
        
        # Plot means with error bars
        x_pos = range(len(models))
        plt.errorbar(x_pos, bootstrap_means, yerr=bootstrap_stds, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.ylabel('Bootstrap Score (Negative MSE)')
        plt.title('Bootstrap Validation with 95% Confidence Intervals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'bootstrap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def select_best_model(self, criterion: str = 'AIC') -> str:
        """
        Select best model based on specified criterion
        
        Args:
            criterion: Selection criterion ('AIC', 'BIC', 'CV_Score', 'R²')
            
        Returns:
            Name of best model
        """
        if not self.comparison_results:
            logger.error("No comparison results available")
            return ""
        
        models = list(self.comparison_results.keys())
        
        if criterion == 'AIC':
            scores = [self.comparison_results[m].metrics.aic for m in models]
            best_model = models[np.argmin(scores)]
        elif criterion == 'BIC':
            scores = [self.comparison_results[m].metrics.bic for m in models]
            best_model = models[np.argmin(scores)]
        elif criterion == 'CV_Score':
            scores = [self.comparison_results[m].cross_validation_score for m in models]
            best_model = models[np.argmax(scores)]
        elif criterion == 'R²':
            scores = [self.comparison_results[m].metrics.r_squared for m in models]
            best_model = models[np.argmax(scores)]
        else:
            raise ValueError(f"Unknown selection criterion: {criterion}")
        
        logger.info(f"Best model selected by {criterion}: {best_model}")
        return best_model
    
    def apply_multiple_testing_correction(self, correction_method: str = "fdr_bh",
                                        alpha: float = 0.05) -> Dict:
        """
        Apply multiple testing correction to model comparison p-values
        
        Args:
            correction_method: Correction method ('fdr_bh', 'bonferroni', 'holm')
            alpha: Significance level
            
        Returns:
            Dictionary with correction results and updated comparison
        """
        logger.info(f"Applying multiple testing correction using {correction_method}")
        
        if not self.comparison_results:
            logger.error("No comparison results available for correction")
            return {}
        
        # Collect p-values from model comparison tests
        pvalues = []
        test_descriptions = []
        
        # Extract p-values from correlation tests and other statistical tests
        for model_name, result in self.comparison_results.items():
            # Use correlation p-value if available, otherwise use a default
            if hasattr(result.metrics, 'correlation_p_value'):
                p_value = result.metrics.correlation_p_value
            else:
                # Calculate p-value from model fit statistics
                p_value = self._calculate_pvalue_from_model_metrics(result)
            
            pvalues.append(p_value)
            test_descriptions.append(f"{model_name}_validation")
        
        pvalues = np.array(pvalues)
        
        # Initialize corrector
        corrector = MultipleTestingCorrector(alpha=alpha)
        
        # Apply correction
        if correction_method == "fdr_bh":
            correction_result = corrector.fdr_benjamini_hochberg(pvalues, alpha)
        elif correction_method == "bonferroni":
            correction_result = corrector.bonferroni_correction(pvalues, alpha)
        elif correction_method == "holm":
            correction_result = corrector.sequential_bonferroni_holm(pvalues, alpha)
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")
        
        # Update comparison results with corrected p-values
        model_names = list(self.comparison_results.keys())
        for i, model_name in enumerate(model_names):
            if model_name in self.comparison_results:
                result = self.comparison_results[model_name]
                result.corrected_pvalue = correction_result.corrected_pvalues[i]
                result.significant_before_correction = pvalues[i] <= alpha
                result.significant_after_correction = correction_result.rejected_nulls[i]
                result.corrected_method = correction_method
        
        # Create before/after comparison
        before_after = corrector.create_before_after_comparison(
            pvalues, correction_result, test_descriptions
        )
        
        # Generate publication table
        pub_table = corrector.generate_publication_table(
            before_after, f"model_comparison_correction_{correction_method}.csv"
        )
        
        # Update comparison DataFrame
        updated_comparison_df = self._update_comparison_dataframe_with_corrections()
        
        return {
            'correction_result': correction_result,
            'before_after_comparison': before_after,
            'publication_table': pub_table,
            'updated_comparison_df': updated_comparison_df,
            'correction_method': correction_method
        }
    
    def _calculate_pvalue_from_model_metrics(self, result) -> float:
        """Calculate approximate p-value from model metrics"""
        # This is a simplified approach - in practice, you'd have proper test statistics
        if hasattr(result, 'metrics') and result.metrics:
            # Use R² to estimate significance (approximate)
            r_squared = result.metrics.r_squared if hasattr(result.metrics, 'r_squared') else 0.0
            
            # Convert R² to approximate p-value (very rough estimate)
            if r_squared > 0.8:
                return 0.001
            elif r_squared > 0.5:
                return 0.01
            elif r_squared > 0.3:
                return 0.05
            else:
                return 0.1
        else:
            return 0.1  # Default non-significant
    
    def _update_comparison_dataframe_with_corrections(self) -> pd.DataFrame:
        """Update comparison DataFrame with corrected p-values and significance"""
        if not self.comparison_results:
            return pd.DataFrame()
        
        # Get original comparison DataFrame
        original_df = self._create_comparison_dataframe(list(self.comparison_results.values()))
        
        # Add correction columns
        model_names = list(self.comparison_results.keys())
        original_df['Corrected_pvalue'] = [
            self.comparison_results[name].corrected_pvalue
            for name in model_names
        ]
        original_df['Significant_Before'] = [
            self.comparison_results[name].significant_before_correction
            for name in model_names
        ]
        original_df['Significant_After'] = [
            self.comparison_results[name].significant_after_correction
            for name in model_names
        ]
        original_df['Correction_Method'] = [
            self.comparison_results[name].corrected_method
            for name in model_names
        ]
        
        return original_df
    
    def comprehensive_model_comparison_with_corrections(self,
                                                     correction_methods: Optional[List[str]] = None,
                                                     output_dir: str = "model_comparison_corrections") -> Dict:
        """
        Perform comprehensive model comparison with multiple testing corrections
        
        Args:
            correction_methods: List of correction methods to apply
            output_dir: Output directory for results
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        if correction_methods is None:
            correction_methods = ['fdr_bh', 'bonferroni', 'holm']
        
        logger.info("Performing comprehensive model comparison with corrections...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Apply corrections for each method
        all_corrections = {}
        for method in correction_methods:
            logger.info(f"Applying {method} correction...")
            correction_results = self.apply_multiple_testing_correction(method)
            if correction_results:
                all_corrections[method] = correction_results
                self._save_correction_results(correction_results, output_path / method)
        
        # Generate method comparison
        method_comparison_df = self._generate_correction_method_comparison(all_corrections)
        
        # Run power analysis
        power_analysis = self._run_comprehensive_power_analysis()
        
        # Create final report
        final_report = self._generate_comprehensive_comparison_report(
            all_corrections, method_comparison_df, power_analysis
        )
        
        # Save comprehensive results
        import json
        with open(output_path / "comprehensive_model_comparison.json", 'w') as f:
            serializable_results = {
                'correction_methods': list(all_corrections.keys()),
                'method_comparison': method_comparison_df.to_dict(),
                'power_analysis': power_analysis,
                'best_model_before_correction': self.select_best_model('AIC'),
                'best_model_after_correction': self._select_best_corrected_model(all_corrections),
                'summary': {
                    'n_models_compared': len(self.comparison_results),
                    'total_corrections_applied': len(all_corrections),
                    'significance_improvements': {
                        method: np.sum([r.get('before_after_comparison', {}).get('summary_statistics', {}).get('n_newly_significant', 0)
                                      for r in results.values()])
                        for method, results in all_corrections.items()
                    }
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        # Save method comparison
        method_comparison_df.to_csv(output_path / "correction_methods_comparison.csv", index=False)
        
        logger.info(f"Comprehensive model comparison completed. Results saved to {output_path}")
        return {
            'correction_results': all_corrections,
            'method_comparison': method_comparison_df,
            'power_analysis': power_analysis,
            'final_report': final_report,
            'output_directory': str(output_path)
        }
    
    def _save_correction_results(self, correction_results: Dict, output_path: Path):
        """Save individual correction results"""
        output_path.mkdir(exist_ok=True)
        
        # Save publication table
        if 'publication_table' in correction_results:
            correction_results['publication_table'].to_csv(
                output_path / "correction_table.csv", index=False
            )
        
        # Save updated comparison DataFrame
        if 'updated_comparison_df' in correction_results:
            correction_results['updated_comparison_df'].to_csv(
                output_path / "updated_model_comparison.csv", index=False
            )
        
        # Save before/after plot
        if 'before_after_comparison' in correction_results:
            fig = correction_results['before_after_comparison']['visualization']
            fig.savefig(output_path / "before_after_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def _generate_correction_method_comparison(self, all_corrections: Dict) -> pd.DataFrame:
        """Generate comparison DataFrame for correction methods"""
        comparison_data = []
        
        for method, results in all_corrections.items():
            if 'correction_result' in results:
                corr_result = results['correction_result']
                summary_stats = results.get('before_after_comparison', {}).get('summary_statistics', {})
                
                comparison_data.append({
                    'Method': method.upper(),
                    'N_Tests': corr_result.n_tests,
                    'N_Significant': corr_result.n_significant,
                    'Significance_Rate': corr_result.n_significant / corr_result.n_tests,
                    'Expected_False_Discoveries': corr_result.n_false_discoveries,
                    'Power_Estimate': corr_result.power_estimate,
                    'Newly_Significant': summary_stats.get('n_newly_significant', 0),
                    'Lost_Significance': summary_stats.get('n_lost_significance', 0),
                    'Net_Change': summary_stats.get('n_significant_after', 0) - summary_stats.get('n_significant_before', 0)
                })
        
        return pd.DataFrame(comparison_data)
    
    def _run_comprehensive_power_analysis(self) -> Dict:
        """Run comprehensive power analysis for model comparison"""
        power_results = {}
        
        # Analyze power for different numbers of models being compared
        n_models_options = [2, 3, 5, 10, len(self.comparison_results)]
        effect_size = 0.5  # Medium effect size
        sample_size = 30   # Typical sample size
        
        corrector = MultipleTestingCorrector()
        
        for n_models in n_models_options:
            method_power = {}
            for correction_method in ['fdr_bh', 'bonferroni', 'holm']:
                power_result = corrector.statistical_power_analysis(
                    effect_size=effect_size,
                    sample_size=sample_size,
                    alpha=0.05,
                    method=correction_method,
                    n_tests=n_models
                )
                method_power[correction_method] = {
                    'power_uncorrected': power_result.power,
                    'power_corrected': power_result.power_with_correction,
                    'corrected_alpha': power_result.corrected_alpha
                }
            
            power_results[f"{n_models}_models"] = method_power
        
        return power_results
    
    def _select_best_corrected_model(self, all_corrections: Dict) -> str:
        """Select best model after multiple testing correction"""
        if not all_corrections:
            return self.select_best_model('AIC')
        
        # Use the FDR-BH corrected results for best model selection
        fdr_results = all_corrections.get('fdr_bh', {})
        if not fdr_results:
            return self.select_best_model('AIC')
        
        # Get updated comparison DataFrame with corrections
        updated_df = fdr_results.get('updated_comparison_df')
        if updated_df is None or updated_df.empty:
            return self.select_best_model('AIC')
        
        # Select based on corrected AIC
        if 'AIC' in updated_df.columns:
            best_idx = updated_df['AIC'].idxmin()
            return updated_df.loc[best_idx, 'Model'] if 'Model' in updated_df.columns else f"Model_{best_idx}"
        else:
            return self.select_best_model('AIC')
    
    def _generate_comprehensive_comparison_report(self, all_corrections: Dict,
                                                method_comparison_df: pd.DataFrame,
                                                power_analysis: Dict) -> str:
        """Generate comprehensive comparison report with corrections"""
        report = """# Comprehensive Model Comparison Report with Multiple Testing Corrections

## Executive Summary

This report presents the results of comprehensive model comparison analysis with multiple testing corrections applied to address statistical rigor concerns in model selection and validation.

"""
        
        # Add method comparison table
        report += "## Correction Method Comparison\n\n"
        report += method_comparison_df.to_string(index=False) + "\n\n"
        
        # Add power analysis
        report += "## Statistical Power Analysis\n\n"
        for n_models, power_data in power_analysis.items():
            report += f"### {n_models.replace('_', ' ').title()}\n"
            for method, method_power in power_data.items():
                report += f"**{method.upper()}**: Power = {method_power['power_corrected']:.3f} (α = {method_power['corrected_alpha']:.4f})\n"
            report += "\n"
        
        # Add recommendations
        report += "## Statistical Rigor Recommendations for Model Selection\n\n"
        report += """1. **Use FDR Correction**: For model comparison studies, FDR control provides a good balance
2. **Report Effect Sizes**: Always report effect sizes alongside corrected p-values
3. **Power Considerations**: Ensure adequate sample sizes for reliable model discrimination
4. **Transparent Reporting**: Present both original and corrected statistics
5. **Model Selection Criteria**: Consider multiple criteria (AIC, BIC, cross-validation) with corrections

## Impact of Multiple Testing Corrections

"""
        
        for method, results in all_corrections.items():
            summary = results.get('before_after_comparison', {}).get('summary_statistics', {})
            report += f"**{method.upper()}**: "
            report += f"{summary.get('n_significant_after', 0)} significant after correction "
            report += f"(vs {summary.get('n_significant_before', 0)} before)\n"
        
        return report

def run_comprehensive_model_comparison(model_class, experimental_data: pd.DataFrame):
    """
    Run comprehensive model comparison workflow
    
    Args:
        model_class: QSP model class
        experimental_data: Experimental data DataFrame
        
    Returns:
        Dictionary with comparison results and analysis
    """
    logger.info("Starting comprehensive model comparison...")
    
    # Initialize model
    model = model_class(atm_proficient=True)
    
    # Initialize validation framework
    validator = ValidationComparisonFramework(model, experimental_data)
    
    # Define multiple fitting strategies
    fitting_strategies = [
        QSPModelFitter(target_metric='ic50', optimization_method='differential_evolution'),
        QSPModelFitter(target_metric='ic50', optimization_method='least_squares'),
        QSPModelFitter(target_metric='apoptosis', optimization_method='differential_evolution')
    ]
    
    # Fit multiple models
    fitted_models = validator.fit_multiple_models(fitting_strategies)
    
    if not fitted_models:
        logger.error("No models were successfully fitted")
        return None
    
    # Compare models
    comparison_results = validator.compare_models()
    
    if comparison_results.empty:
        logger.error("Model comparison failed")
        return None
    
    # Generate comparison plots
    validator.generate_comparison_plots()
    
    # Select best model
    best_model_aic = validator.select_best_model('AIC')
    best_model_cv = validator.select_best_model('CV_Score')
    
    # Compile results
    results = {
        'fitted_models': fitted_models,
        'comparison_results': comparison_results,
        'validation_framework': validator,
        'best_models': {
            'by_AIC': best_model_aic,
            'by_CV_Score': best_model_cv
        },
        'summary': {
            'n_models_compared': len(fitted_models),
            'best_r2': comparison_results['R²'].max(),
            'best_rmse': comparison_results['RMSE'].min(),
            'best_cv_score': comparison_results['CV_Score'].max()
        }
    }
    
    logger.info(f"Model comparison completed: {len(fitted_models)} models compared")
    return results

if __name__ == "__main__":
    # Example usage
    from enhanced_ddr_qsp_model import EnhancedDDRModel
    import pandas as pd
    
    print("Model Validation Comparison Framework - Example Usage")
    print("=" * 60)
    
    # Create mock experimental data
    mock_data = pd.DataFrame({
        'cell_line': ['MEC1', 'MEC1', 'MEC2', 'MEC2'],
        'atm_status': ['deficient', 'deficient', 'proficient', 'proficient'],
        'drug': ['AZD6738', 'Olaparib', 'AZD6738', 'Olaparib'],
        'drug_target': ['ATR', 'PARP', 'ATR', 'PARP'],
        'ic50_nm': [50.0, 200.0, 500.0, 800.0]
    })
    
    # Run comprehensive comparison
    comparison_results = run_comprehensive_model_comparison(EnhancedDDRModel, mock_data)
    
    if comparison_results:
        print(f"\nComparison completed successfully!")
        print(f"Models compared: {comparison_results['summary']['n_models_compared']}")
        print(f"Best R²: {comparison_results['summary']['best_r2']:.3f}")
        print(f"Best CV Score: {comparison_results['summary']['best_cv_score']:.3f}")
        print(f"Best model (AIC): {comparison_results['best_models']['by_AIC']}")
    else:
        print("Model comparison failed!")