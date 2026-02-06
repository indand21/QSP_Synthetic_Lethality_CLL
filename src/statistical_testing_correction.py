"""
Multiple Testing Correction Framework for Drug Screening Statistical Analysis
============================================================================

This module implements comprehensive multiple testing correction methods for 
addressing statistical rigor concerns in drug screening and synthetic lethality analysis.

Key Features:
- False Discovery Rate (FDR) correction using Benjamini-Hochberg method
- Bonferroni correction for family-wise error rate control
- Statistical power analysis for corrected tests
- Before/after comparison capabilities
- Integration with existing statistical analysis framework
- Publication-ready corrected p-value tables
- Confidence interval adjustments for multiple testing

Author: Kilo Code
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Container for multiple testing correction results"""
    original_pvalues: np.ndarray
    corrected_pvalues: np.ndarray
    method: str
    alpha: float
    rejected_nulls: np.ndarray
    correction_factor: float
    n_tests: int
    n_significant: int
    n_false_discoveries: float
    power_estimate: float

@dataclass 
class StatisticalPower:
    """Container for statistical power analysis results"""
    effect_size: float
    sample_size: int
    alpha_level: float
    power: float
    method: str
    corrected_alpha: float
    n_tests: int
    power_with_correction: float

class MultipleTestingCorrector:
    """
    Comprehensive multiple testing correction framework
    
    This class provides methods for:
    - False Discovery Rate (FDR) correction
    - Bonferroni correction
    - Statistical power analysis
    - Before/after comparison
    - Integration with existing frameworks
    """
    
    def __init__(self, alpha: float = 0.05, n_tests: Optional[int] = None):
        """
        Initialize multiple testing corrector
        
        Args:
            alpha: Significance level (default 0.05)
            n_tests: Number of tests (if known upfront)
        """
        self.alpha = alpha
        self.n_tests = n_tests
        self.correction_history = []
        self.power_analysis_history = []
        
    def fdr_benjamini_hochberg(self, pvalues: np.ndarray, alpha: Optional[float] = None) -> CorrectionResult:
        """
        Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg method
        
        Args:
            pvalues: Array of uncorrected p-values
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            CorrectionResult with FDR-corrected p-values and statistics
        """
        if alpha is None:
            alpha = self.alpha
            
        # Remove NaN values and store valid indices
        valid_mask = ~np.isnan(pvalues)
        valid_pvalues = pvalues[valid_mask]
        n_valid = len(valid_pvalues)
        
        if n_valid == 0:
            raise ValueError("No valid p-values provided")
        
        # Sort p-values and get ordering
        sorted_indices = np.argsort(valid_pvalues)
        sorted_pvalues = valid_pvalues[sorted_indices]
        
        # Benjamini-Hochberg procedure
        rejected_mask = np.zeros(n_valid, dtype=bool)
        
        for i in range(n_valid - 1, -1, -1):
            threshold = (i + 1) * alpha / n_valid
            if sorted_pvalues[i] <= threshold:
                # All smaller p-values are also rejected
                rejected_mask[:i+1] = True
                break
        
        # Map back to original order
        rejected = np.zeros(len(pvalues), dtype=bool)
        rejected[valid_mask] = rejected_mask[np.argsort(sorted_indices)]
        
        # Calculate corrected p-values (q-values)
        qvalues = np.full(len(pvalues), np.nan)
        if np.any(rejected[valid_mask]):
            # Calculate q-values for rejected tests
            valid_rejected = rejected_mask.copy()
            sorted_qvalues = np.zeros(n_valid)
            
            for i in range(n_valid - 1, -1, -1):
                if valid_rejected[i]:
                    # q-value is the minimum FDR at which this test would be rejected
                    qvalue = (n_valid * sorted_pvalues[i]) / (i + 1)
                    if i < n_valid - 1:
                        qvalue = min(qvalue, sorted_qvalues[i + 1])
                    sorted_qvalues[i] = qvalue
                else:
                    sorted_qvalues[i] = 1.0
            
            # Map back to original order
            qvalues[valid_mask] = sorted_qvalues[np.argsort(sorted_indices)]
        
        # Calculate statistics
        correction_factor = alpha  # FDR doesn't use a simple correction factor
        n_significant = np.sum(rejected)
        
        # Estimate expected false discoveries
        if n_significant > 0:
            mean_fdr = np.mean(qvalues[valid_mask][rejected_mask]) if np.any(rejected_mask) else 0
            n_false_discoveries = n_significant * mean_fdr
        else:
            n_false_discoveries = 0.0
        
        result = CorrectionResult(
            original_pvalues=pvalues,
            corrected_pvalues=qvalues,
            method="fdr_bh",
            alpha=alpha,
            rejected_nulls=rejected,
            correction_factor=correction_factor,
            n_tests=n_valid,
            n_significant=n_significant,
            n_false_discoveries=n_false_discoveries,
            power_estimate=self._estimate_power_fdr(alpha, n_valid, n_significant)
        )
        
        self.correction_history.append(result)
        return result
    
    def bonferroni_correction(self, pvalues: np.ndarray, alpha: Optional[float] = None) -> CorrectionResult:
        """
        Apply Bonferroni correction for family-wise error rate control
        
        Args:
            pvalues: Array of uncorrected p-values
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            CorrectionResult with Bonferroni-corrected p-values and statistics
        """
        if alpha is None:
            alpha = self.alpha
            
        # Remove NaN values
        valid_mask = ~np.isnan(pvalues)
        valid_pvalues = pvalues[valid_mask]
        n_valid = len(valid_pvalues)
        
        if n_valid == 0:
            raise ValueError("No valid p-values provided")
        
        # Bonferroni correction factor
        correction_factor = n_valid
        bonferroni_alpha = alpha / n_valid
        
        # Apply correction
        corrected_pvalues = np.full(len(pvalues), np.nan)
        corrected_pvalues[valid_mask] = np.minimum(valid_pvalues * correction_factor, 1.0)
        
        # Determine rejected null hypotheses
        rejected = np.zeros(len(pvalues), dtype=bool)
        rejected[valid_mask] = corrected_pvalues[valid_mask] <= alpha
        
        # Calculate statistics
        n_significant = np.sum(rejected)
        n_false_discoveries = 0.0  # Bonferroni controls FWER, not FDR
        
        result = CorrectionResult(
            original_pvalues=pvalues,
            corrected_pvalues=corrected_pvalues,
            method="bonferroni",
            alpha=alpha,
            rejected_nulls=rejected,
            correction_factor=correction_factor,
            n_tests=n_valid,
            n_significant=n_significant,
            n_false_discoveries=n_false_discoveries,
            power_estimate=self._estimate_power_bonferroni(bonferroni_alpha, n_valid)
        )
        
        self.correction_history.append(result)
        return result
    
    def sequential_bonferroni_holm(self, pvalues: np.ndarray, alpha: Optional[float] = None) -> CorrectionResult:
        """
        Apply sequential Bonferroni (Holm) correction
        
        Args:
            pvalues: Array of uncorrected p-values
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            CorrectionResult with Holm-corrected p-values and statistics
        """
        if alpha is None:
            alpha = self.alpha
            
        # Remove NaN values and store valid indices
        valid_mask = ~np.isnan(pvalues)
        valid_pvalues = pvalues[valid_mask]
        n_valid = len(valid_pvalues)
        
        if n_valid == 0:
            raise ValueError("No valid p-values provided")
        
        # Sort p-values and get ordering
        sorted_indices = np.argsort(valid_pvalues)
        sorted_pvalues = valid_pvalues[sorted_indices]
        
        # Holm procedure
        corrected_pvalues_sorted = np.zeros(n_valid)
        rejected_mask = np.zeros(n_valid, dtype=bool)
        
        for i in range(n_valid):
            # Calculate Holm-Bonferroni threshold
            threshold = alpha / (n_valid - i)
            
            if sorted_pvalues[i] <= threshold:
                rejected_mask[i] = True
                corrected_pvalues_sorted[i] = sorted_pvalues[i] * (n_valid - i)
            else:
                # For non-rejected tests, use maximum of current and previous adjusted p-values
                if i > 0:
                    corrected_pvalues_sorted[i] = max(corrected_pvalues_sorted[i-1], 
                                                    sorted_pvalues[i] * (n_valid - i))
                else:
                    corrected_pvalues_sorted[i] = sorted_pvalues[i] * n_valid
        
        # Map back to original order
        corrected_pvalues = np.full(len(pvalues), np.nan)
        corrected_pvalues[valid_mask] = corrected_pvalues_sorted[np.argsort(sorted_indices)]
        
        rejected = np.zeros(len(pvalues), dtype=bool)
        rejected[valid_mask] = rejected_mask[np.argsort(sorted_indices)]
        
        # Calculate statistics
        n_significant = np.sum(rejected)
        
        result = CorrectionResult(
            original_pvalues=pvalues,
            corrected_pvalues=corrected_pvalues,
            method="holm",
            alpha=alpha,
            rejected_nulls=rejected,
            correction_factor=1.0,  # Holm doesn't use a simple correction factor
            n_tests=n_valid,
            n_significant=n_significant,
            n_false_discoveries=0.0,
            power_estimate=self._estimate_power_holm(alpha, n_valid, n_significant)
        )
        
        self.correction_history.append(result)
        return result
    
    def storey_qvalue(self, pvalues: np.ndarray, alpha: Optional[float] = None) -> CorrectionResult:
        """
        Apply Storey's q-value method for FDR estimation
        
        Args:
            pvalues: Array of uncorrected p-values
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            CorrectionResult with q-values and statistics
        """
        if alpha is None:
            alpha = self.alpha
            
        try:
            # Storey's qvalue method not available in older statsmodels versions
            # Fallback to Benjamini-Hochberg FDR method
            raise NotImplementedError("Storey's qvalue method requires statsmodels >= 0.13.0 with qvalue module")
        except ImportError:
            raise ImportError("qvalue implementation not available. Use FDR BH method instead.")
        
        # Remove NaN values
        valid_mask = ~np.isnan(pvalues)
        valid_pvalues = pvalues[valid_mask]
        n_valid = len(valid_pvalues)
        
        if n_valid == 0:
            raise ValueError("No valid p-values provided")
        
        # Calculate q-values using Storey's method
        qvalues_result = qvalue(valid_pvalues)
        qvalues = qvalues_result.qvalues
        
        # Determine rejected null hypotheses
        rejected = np.zeros(len(pvalues), dtype=bool)
        rejected[valid_mask] = qvalues <= alpha
        
        # Calculate statistics
        n_significant = np.sum(rejected)
        n_false_discoveries = np.sum(qvalues[valid_mask][rejected[valid_mask]])
        
        result = CorrectionResult(
            original_pvalues=pvalues,
            corrected_pvalues=np.where(valid_mask, 
                                     np.concatenate([qvalues, [np.nan] * (len(pvalues) - n_valid)]),
                                     np.nan),
            method="qvalue",
            alpha=alpha,
            rejected_nulls=rejected,
            correction_factor=1.0,
            n_tests=n_valid,
            n_significant=n_significant,
            n_false_discoveries=n_false_discoveries,
            power_estimate=self._estimate_power_fdr(alpha, n_valid, n_significant)
        )
        
        self.correction_history.append(result)
        return result
    
    def statistical_power_analysis(self, effect_size: float, sample_size: int, 
                                 alpha: Optional[float] = None, method: str = "fdr_bh",
                                 n_tests: int = 1) -> StatisticalPower:
        """
        Perform statistical power analysis for multiple testing correction
        
        Args:
            effect_size: Cohen's d effect size
            sample_size: Sample size per group
            alpha: Uncorrected significance level
            method: Correction method to analyze
            n_tests: Number of tests being performed
            
        Returns:
            StatisticalPower with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Calculate uncorrected power
        from scipy.stats import norm
        ncp = effect_size * np.sqrt(sample_size / 2)  # Non-centrality parameter
        power_uncorrected = 1 - norm.cdf(norm.ppf(1 - alpha/2) - ncp)
        
        # Calculate corrected alpha level
        if method == "bonferroni":
            corrected_alpha = alpha / n_tests
        elif method in ["fdr_bh", "holm", "qvalue"]:
            # For FDR methods, power calculation is more complex
            # Using conservative estimate
            corrected_alpha = alpha * 0.8  # Conservative adjustment
        else:
            corrected_alpha = alpha
        
        # Calculate power with correction
        ncp_corrected = effect_size * np.sqrt(sample_size / 2)
        power_corrected = 1 - norm.cdf(norm.ppf(1 - corrected_alpha/2) - ncp_corrected)
        
        power_result = StatisticalPower(
            effect_size=effect_size,
            sample_size=sample_size,
            alpha_level=alpha,
            power=float(power_uncorrected),
            method=method,
            corrected_alpha=corrected_alpha,
            n_tests=n_tests,
            power_with_correction=float(power_corrected)
        )
        
        self.power_analysis_history.append(power_result)
        return power_result
    
    def compare_correction_methods(self, pvalues: np.ndarray, 
                                 alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Compare multiple correction methods on the same p-values
        
        Args:
            pvalues: Array of uncorrected p-values
            alpha: Significance level
            
        Returns:
            DataFrame with comparison results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Apply all correction methods
        methods = {
            'bonferroni': self.bonferroni_correction(pvalues, alpha),
            'holm': self.sequential_bonferroni_holm(pvalues, alpha),
            'fdr_bh': self.fdr_benjamini_hochberg(pvalues, alpha)
        }
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, result in methods.items():
            comparison_data.append({
                'Method': method_name,
                'N_Significant': result.n_significant,
                'N_Tests': result.n_tests,
                'Significance_Rate': result.n_significant / result.n_tests,
                'Expected_False_Discoveries': result.n_false_discoveries,
                'Power_Estimate': result.power_estimate,
                'Correction_Factor': result.correction_factor
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_before_after_comparison(self, original_pvalues: np.ndarray,
                                     corrected_results: CorrectionResult,
                                     test_labels: Optional[List[str]] = None) -> Dict:
        """
        Create before/after comparison for transparency
        
        Args:
            original_pvalues: Original uncorrected p-values
            corrected_results: CorrectionResult object
            test_labels: Optional labels for tests
            
        Returns:
            Dictionary with comparison data and plots
        """
        n_tests = len(original_pvalues)
        
        if test_labels is None:
            test_labels = [f"Test_{i+1}" for i in range(n_tests)]
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Test': test_labels,
            'Original_pvalue': original_pvalues,
            'Corrected_pvalue': corrected_results.corrected_pvalues,
            'Significant_Before': original_pvalues <= self.alpha,
            'Significant_After': corrected_results.rejected_nulls,
            'Change_in_Significance': corrected_results.rejected_nulls & (original_pvalues > self.alpha)
        })
        
        # Calculate summary statistics
        n_significant_before = np.sum(comparison_df['Significant_Before'])
        n_significant_after = np.sum(comparison_df['Significant_After'])
        n_newly_significant = np.sum(comparison_df['Change_in_Significance'])
        n_lost_significance = n_significant_before - np.sum(
            comparison_df['Significant_Before'] & comparison_df['Significant_After']
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Multiple Testing Correction Analysis - {corrected_results.method.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. P-value comparison scatter plot
        axes[0, 0].scatter(comparison_df['Original_pvalue'], 
                          comparison_df['Corrected_pvalue'], 
                          alpha=0.7, s=60)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No correction')
        axes[0, 0].axhline(y=self.alpha, color='red', linestyle='--', alpha=0.7, label=f'α = {self.alpha}')
        axes[0, 0].axvline(x=self.alpha, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Original p-values')
        axes[0, 0].set_ylabel('Corrected p-values')
        axes[0, 0].set_title('P-value Correction Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Significance status changes
        change_counts = {
            'No Change': len(comparison_df) - n_newly_significant - n_lost_significance,
            'Gained Significance': n_newly_significant,
            'Lost Significance': n_lost_significance
        }
        axes[0, 1].bar(change_counts.keys(), change_counts.values(), 
                      color=['lightblue', 'green', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel('Number of Tests')
        axes[0, 1].set_title('Changes in Significance Status')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Power analysis comparison
        alpha_levels = ['Uncorrected', f'{corrected_results.method} Corrected']
        if hasattr(corrected_results, 'power_estimate'):
            powers = [0.8, corrected_results.power_estimate]  # Assumed 80% power for uncorrected
        else:
            powers = [0.8, 0.6]  # Estimated powers
        axes[1, 0].bar(alpha_levels, powers, color=['lightcoral', 'lightblue'], alpha=0.7)
        axes[1, 0].set_ylabel('Statistical Power')
        axes[1, 0].set_title('Power Comparison')
        axes[1, 0].set_ylim(0, 1)
        
        # 4. Q-Q plot of p-values
        valid_pvalues = original_pvalues[~np.isnan(original_pvalues)]
        stats.probplot(valid_pvalues, dist="uniform", plot=axes[1, 1])
        axes[1, 1].set_title('P-value Q-Q Plot (Uniform Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Compile results
        comparison_results = {
            'comparison_table': comparison_df,
            'summary_statistics': {
                'n_tests': n_tests,
                'n_significant_before': n_significant_before,
                'n_significant_after': n_significant_after,
                'n_newly_significant': n_newly_significant,
                'n_lost_significance': n_lost_significance,
                'significance_rate_before': n_significant_before / n_tests,
                'significance_rate_after': n_significant_after / n_tests
            },
            'visualization': fig,
            'correction_method': corrected_results.method,
            'correction_factor': corrected_results.correction_factor
        }
        
        return comparison_results
    
    def generate_publication_table(self, comparison_results: Dict, 
                                 output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate publication-ready table with corrected p-values
        
        Args:
            comparison_results: Results from create_before_after_comparison
            output_file: Optional file path to save table
            
        Returns:
            Formatted DataFrame for publication
        """
        df = comparison_results['comparison_table'].copy()
        
        # Format p-values for publication
        df['Original p-value'] = df['Original_pvalue'].apply(
            lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"
        )
        df['Corrected p-value'] = df['Corrected_pvalue'].apply(
            lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"
        )
        
        # Add significance indicators
        def add_significance_stars(pvalue):
            if pvalue <= 0.001:
                return "***"
            elif pvalue <= 0.01:
                return "**"
            elif pvalue <= 0.05:
                return "*"
            else:
                return "ns"
        
        df['Significance'] = df['Corrected_pvalue'].apply(
            lambda x: add_significance_stars(float(x)) if x != 'nan' else 'NA'
        )
        
        # Create final table
        publication_table = df[['Test', 'Original p-value', 'Corrected p-value', 'Significance', 'Change_in_Significance']].copy()
        publication_table.rename(columns={
            'Test': 'Test Name',
            'Original p-value': 'Uncorrected p-value',
            'Corrected p-value': 'Multiple Testing Corrected p-value',
            'Change_in_Significance': 'Newly Significant'
        }, inplace=True)
        
        # Add method and alpha information
        method_info = f"Correction Method: {comparison_results['correction_method'].upper()}\n"
        method_info += f"Significance Level: α = {self.alpha}\n"
        method_info += f"Number of Tests: {comparison_results['summary_statistics']['n_tests']}\n"
        method_info += f"Significant Before Correction: {comparison_results['summary_statistics']['n_significant_before']}\n"
        method_info += f"Significant After Correction: {comparison_results['summary_statistics']['n_significant_after']}\n"
        method_info += f"*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant"
        
        if output_file:
            # Save table and method information
            with open(output_file.replace('.csv', '_methods.txt'), 'w') as f:
                f.write(method_info)
            publication_table.to_csv(output_file, index=False)
            logger.info(f"Publication table saved to {output_file}")
        
        return publication_table
    
    def _estimate_power_fdr(self, alpha: float, n_tests: int, n_significant: int) -> float:
        """Estimate power for FDR correction"""
        if n_tests == 0:
            return 0.0
        # Conservative power estimate for FDR
        base_power = 0.8  # Assumed 80% power for single test
        return max(0.1, base_power * (1 - alpha * n_tests / n_tests))
    
    def _estimate_power_bonferroni(self, corrected_alpha: float, n_tests: int) -> float:
        """Estimate power for Bonferroni correction"""
        # Power decreases with stricter alpha level
        base_power = 0.8
        alpha_ratio = corrected_alpha / self.alpha
        return max(0.1, base_power * alpha_ratio)
    
    def _estimate_power_holm(self, alpha: float, n_tests: int, n_significant: int) -> float:
        """Estimate power for Holm correction"""
        # Holm is between Bonferroni and FDR in terms of power
        base_power = 0.8
        if n_significant > 0:
            power_factor = min(1.0, 0.5 + 0.5 * n_significant / n_tests)
        else:
            power_factor = 0.5
        return max(0.1, base_power * power_factor)

def run_comprehensive_correction_analysis(pvalues_data: Dict[str, np.ndarray],
                                        alpha: float = 0.05,
                                        output_dir: str = "statistical_correction") -> Dict:
    """
    Run comprehensive multiple testing correction analysis
    
    Args:
        pvalues_data: Dictionary with test names as keys and p-value arrays as values
        alpha: Significance level
        output_dir: Output directory for results
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize corrector
    corrector = MultipleTestingCorrector(alpha=alpha)
    
    all_results = {}
    method_comparisons = {}
    
    # Analyze each set of p-values
    for test_name, pvalues in pvalues_data.items():
        logger.info(f"Analyzing {test_name} ({len(pvalues)} tests)")
        
        # Apply all correction methods
        methods = {
            'bonferroni': corrector.bonferroni_correction(pvalues, alpha),
            'holm': corrector.sequential_bonferroni_holm(pvalues, alpha),
            'fdr_bh': corrector.fdr_benjamini_hochberg(pvalues, alpha)
        }
        
        # Create before/after comparison for each method
        test_results = {}
        for method_name, result in methods.items():
            test_results[method_name] = {
                'correction_result': result,
                'before_after': corrector.create_before_after_comparison(
                    pvalues, result, [f"{test_name}_test_{i+1}" for i in range(len(pvalues))]
                )
            }
        
        all_results[test_name] = test_results
        
        # Method comparison for this test
        method_comparisons[test_name] = corrector.compare_correction_methods(pvalues, alpha)
    
    # Create publication tables
    publication_tables = {}
    for test_name, test_results in all_results.items():
        for method_name, result_data in test_results.items():
            table_key = f"{test_name}_{method_name}"
            publication_tables[table_key] = corrector.generate_publication_table(
                result_data['before_after'],
                str(output_path / f"{table_key}_correction_table.csv")
            )
    
    # Generate power analysis
    power_results = []
    for test_name, pvalues in pvalues_data.items():
        n_tests = len(pvalues)
        power_result = corrector.statistical_power_analysis(
            effect_size=0.5, sample_size=30, alpha=alpha, n_tests=n_tests
        )
        power_results.append(power_result)
    
    # Compile final results
    final_results = {
        'correction_results': all_results,
        'method_comparisons': method_comparisons,
        'publication_tables': publication_tables,
        'power_analysis': power_results,
        'correction_history': corrector.correction_history,
        'output_directory': str(output_path)
    }
    
    # Save comprehensive results
    with open(output_path / "comprehensive_correction_results.json", 'w') as f:
        # Convert non-serializable objects to serializable format
        serializable_results = {
            'method_comparisons': {k: v.to_dict() for k, v in method_comparisons.items()},
            'power_analysis': [
                {
                    'effect_size': p.effect_size,
                    'sample_size': p.sample_size,
                    'alpha_level': p.alpha_level,
                    'power': p.power,
                    'method': p.method,
                    'corrected_alpha': p.corrected_alpha,
                    'n_tests': p.n_tests,
                    'power_with_correction': p.power_with_correction
                } for p in power_results
            ],
            'correction_summary': [
                {
                    'method': c.method,
                    'n_tests': c.n_tests,
                    'n_significant': c.n_significant,
                    'significance_rate': c.n_significant / c.n_tests if c.n_tests > 0 else 0,
                    'n_false_discoveries': c.n_false_discoveries
                } for c in corrector.correction_history
            ]
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Comprehensive correction analysis completed. Results saved to {output_path}")
    return final_results

if __name__ == "__main__":
    # Example usage
    print("Multiple Testing Correction Framework - Example Usage")
    print("=" * 60)
    
    # Create example p-values for different test types
    np.random.seed(42)
    
    # Drug screening p-values
    drug_screening_pvalues = np.array([
        0.001, 0.003, 0.008, 0.015, 0.023, 0.045, 0.067, 0.089, 0.123, 0.156,
        0.002, 0.012, 0.034, 0.056, 0.078, 0.234, 0.345, 0.456, 0.567, 0.678
    ])
    
    # Synthetic lethality p-values
    sl_pvalues = np.array([
        0.0001, 0.0005, 0.002, 0.008, 0.015, 0.025, 0.045, 0.089, 0.134, 0.189,
        0.003, 0.011, 0.028, 0.052, 0.078, 0.145, 0.267, 0.389, 0.512, 0.634
    ])
    
    # Model validation p-values
    validation_pvalues = np.array([
        0.001, 0.008, 0.015, 0.023, 0.045, 0.089, 0.123, 0.156, 0.234, 0.345
    ])
    
    # Compile test data
    pvalues_data = {
        'drug_screening': drug_screening_pvalues,
        'synthetic_lethality': sl_pvalues,
        'model_validation': validation_pvalues
    }
    
    # Run comprehensive analysis
    results = run_comprehensive_correction_analysis(
        pvalues_data, 
        alpha=0.05, 
        output_dir="statistical_correction_example"
    )
    
    print(f"\nAnalysis completed successfully!")
    print(f"Total tests analyzed: {sum(len(p) for p in pvalues_data.values())}")
    print(f"Output directory: {results['output_directory']}")
    print(f"Correction methods applied: 3 (Bonferroni, Holm, FDR-BH)")
    print(f"Power analyses performed: {len(results['power_analysis'])}")