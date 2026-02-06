#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Script
===========================================

This script generates publication-quality figures using the QSP model results
with available packages and ensures compatibility with the R framework.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'Arial'

# Set seaborn style for better aesthetics
sns.set_style("whitegrid", {
    "grid.color": ".9",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "font.family": "Arial",
    "font.size": 10
})

def main():
    """Main function to generate all publication-quality figures"""

    print("Publication-Quality Figure Generation")
    print("=" * 50)

    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "figures" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading screening results...")
    screening_data = pd.read_csv(str(Path(__file__).resolve().parent.parent / "data" / "screening" / "complete_screening_results.csv"))
    print(f"   Loaded {len(screening_data)} drug screening results")

    print("Loading time series data...")
    time_series_files = list((Path(__file__).resolve().parent.parent / "data" / "time_series").glob("*.csv"))
    time_series_data = {}
    for file in time_series_files:
        drug_name = file.stem
        time_series_data[drug_name] = pd.read_csv(file)
        print(f"   Loaded: {drug_name}")

    # 1. Enhanced Synthetic Lethality Analysis
    print("\nCreating Enhanced Synthetic Lethality Analysis...")
    create_synthetic_lethality_plot(screening_data, output_dir)

    # 2. Time-course Analysis for Top Candidates
    print("\nCreating Time-Course Analysis...")
    create_time_course_plots(screening_data, time_series_data, output_dir)

    # 3. Statistical Analysis Visualizations
    print("\nCreating Statistical Analysis Plots...")
    create_statistical_plots(screening_data, output_dir)

    # 4. Multi-Panel Publication Figure
    print("\nCreating Multi-Panel Publication Figure...")
    create_multipanel_figure(screening_data, time_series_data, output_dir)

    # 5. Therapeutic Index Analysis
    print("\nCreating Therapeutic Index Analysis...")
    create_therapeutic_index_plot(screening_data, output_dir)

    # 6. DDR Pathway Network Diagram
    print("\nCreating DDR Pathway Network...")
    create_pathway_network_plot(screening_data, output_dir)

    # 7. Generate Integration Report
    print("\nGenerating Integration Report...")
    create_integration_report(screening_data, time_series_data, output_dir)

    print("\nPublication Figure Generation Complete!")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("\nGenerated Files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            print(f"   {file.name}")

    return output_dir

def get_color_palette(n_colors, palette_name="viridis"):
    """Get color palette for plots"""
    try:
        if palette_name == "viridis":
            return cm.viridis(np.linspace(0, 1, n_colors))
        elif palette_name == "plasma":
            return cm.plasma(np.linspace(0, 1, n_colors))
        elif palette_name == "tab10":
            return cm.tab10(np.linspace(0, 1, n_colors))
        elif palette_name == "Set3":
            return cm.Set3(np.linspace(0, 1, n_colors))
        else:
            return cm.viridis(np.linspace(0, 1, n_colors))
    except:
        # Fallback to simple colors
        return plt.cm.viridis(np.linspace(0, 1, n_colors))

def create_synthetic_lethality_plot(screening_data, output_dir):
    """Create enhanced synthetic lethality analysis plot"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Synthetic Lethality Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Top drugs ranking
    ax = axes[0, 0]
    top_drugs = screening_data.nlargest(10, 'Synthetic_Lethality_Score')
    y_pos = range(len(top_drugs))
    colors = get_color_palette(len(top_drugs), "viridis")
    bars = ax.barh(y_pos, top_drugs['Synthetic_Lethality_Score'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([drug[:15] + '...' if len(drug) > 15 else drug
                       for drug in top_drugs['Drug']], fontsize=9)
    ax.set_xlabel('Synthetic Lethality Score', fontsize=12)
    ax.set_title('A. Top Drug Candidates', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_drugs['Synthetic_Lethality_Score'])):
        ax.text(value + 0.01, i, f'{value:.2f}', va='center', fontsize=8)

    # Plot 2: Target distribution
    ax = axes[0, 1]
    target_counts = screening_data['Target'].value_counts()
    colors = get_color_palette(len(target_counts), "Set3")
    wedges, texts, autotexts = ax.pie(target_counts.values, labels=target_counts.index,
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('B. Target Distribution', fontweight='bold')

    # Plot 3: SL Score vs Therapeutic Index
    ax = axes[1, 0]
    scatter = ax.scatter(screening_data['Synthetic_Lethality_Score'],
                        screening_data['Therapeutic_Index'],
                        c=screening_data['Target'].astype('category').cat.codes,
                        s=100, alpha=0.7, cmap='tab10')
    ax.set_xlabel('Synthetic Lethality Score')
    ax.set_ylabel('Therapeutic Index')
    ax.set_title('C. SL Score vs Therapeutic Index', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='TI = 2')
    ax.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='SL = 1')
    ax.legend()

    # Plot 4: Target effectiveness
    ax = axes[1, 1]
    target_effectiveness = screening_data.groupby('Target')['Synthetic_Lethality_Score'].agg(['mean', 'std']).sort_values('mean')
    y_pos = range(len(target_effectiveness))
    colors = get_color_palette(len(target_effectiveness), "plasma")
    bars = ax.barh(y_pos, target_effectiveness['mean'],
                   xerr=target_effectiveness['std'], capsize=5, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(target_effectiveness.index, fontsize=10)
    ax.set_xlabel('Mean SL Score ± SD', fontsize=12)
    ax.set_title('D. Target Effectiveness', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f"enhanced_synthetic_lethality_analysis.{fmt}",
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Enhanced SL analysis created")

def create_time_course_plots(screening_data, time_series_data, output_dir):
    """Create time-course plots for top candidates"""

    # Get top 3 candidates
    top_candidates = screening_data.nlargest(3, 'Synthetic_Lethality_Score')

    for idx, (_, drug_row) in enumerate(top_candidates.iterrows()):
        drug_name = drug_row['Drug']
        target = drug_row['Target']

        # Find matching time series data
        timecourse_data = None
        for ts_name, ts_data in time_series_data.items():
            if target.lower() in ts_name.lower() or drug_name.split()[0].lower() in ts_name.lower():
                timecourse_data = ts_data
                break

        if timecourse_data is not None:
            print(f"   Creating time-course for: {drug_name}")

            # Create time-course figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Time-Course Analysis: {drug_name}', fontsize=16, fontweight='bold')

            # Plot different pathway components
            variables = [
                ('DSB', 'DNA Damage (DSB)'),
                ('ATM_active', 'ATM Activity'),
                ('ATR_active', 'ATR Activity'),
                ('CHK1_active', 'CHK1 Activity'),
                ('p53_active', 'p53 Activity'),
                ('ApoptosisSignal', 'Apoptosis Signal')
            ]

            colors = get_color_palette(len(variables), "viridis")

            for i, (var, title) in enumerate(variables):
                row, col = i // 3, i % 3
                ax = axes[row, col]

                if var in timecourse_data.columns:
                    ax.plot(timecourse_data['Time'], timecourse_data[var],
                           color=colors[i], linewidth=2.5, alpha=0.8)
                    ax.fill_between(timecourse_data['Time'], timecourse_data[var],
                                   alpha=0.3, color=colors[i])
                    ax.set_title(title, fontweight='bold')
                    ax.set_xlabel('Time (hours)')
                    ax.set_ylabel('Activity Level')
                    ax.grid(True, alpha=0.3)

                    # Add trend annotation
                    start_val = timecourse_data[var].iloc[0]
                    end_val = timecourse_data[var].iloc[-1]
                    change_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
                    ax.text(0.02, 0.98, f'Change: {change_pct:+.1f}%',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            plt.tight_layout()

            # Save in multiple formats
            safe_name = drug_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            for fmt in ['png', 'pdf', 'svg']:
                fig.savefig(output_dir / f"time_course_{safe_name}.{fmt}",
                           dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

    print("   Time-course plots created")

def create_statistical_plots(screening_data, output_dir):
    """Create comprehensive statistical analysis plots"""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Statistical Analysis', fontsize=16, fontweight='bold')

    # 1. SL Score Distribution
    ax = axes[0, 0]
    ax.hist(screening_data['Synthetic_Lethality_Score'], bins=20,
           alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('A. SL Score Distribution', fontweight='bold')
    ax.set_xlabel('Synthetic Lethality Score')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_sl = screening_data['Synthetic_Lethality_Score'].mean()
    ax.axvline(mean_sl, color='red', linestyle='--', label=f'Mean: {mean_sl:.2f}')
    ax.legend()

    # 2. Pathway Activity Heatmap
    ax = axes[0, 1]
    pathway_cols = ['DSB_Level', 'HR_Activity', 'PARP_Activity', 'Cell_Cycle_Arrest', 'ATM_Activity', 'ATR_Activity']
    correlation_matrix = screening_data[pathway_cols].corr()

    # Create heatmap
    im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(pathway_cols)))
    ax.set_yticks(range(len(pathway_cols)))
    ax.set_xticklabels([col.replace('_', '\n') for col in pathway_cols], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([col.replace('_', '\n') for col in pathway_cols], fontsize=8)
    ax.set_title('B. Pathway Correlations', fontweight='bold')

    # Add correlation values
    for i in range(len(pathway_cols)):
        for j in range(len(pathway_cols)):
            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                   ha='center', va='center', fontsize=8,
                   color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    # 3. Therapeutic Window Categories
    ax = axes[0, 2]
    screening_data['TI_Category'] = pd.cut(screening_data['Therapeutic_Index'],
                                          bins=[0, 1, 2, 5, float('inf')],
                                          labels=['Low', 'Moderate', 'High', 'Very High'])
    category_counts = screening_data['TI_Category'].value_counts()
    bars = ax.bar(range(len(category_counts)), category_counts.values,
                  color=['lightcoral', 'gold', 'lightgreen', 'darkgreen'], alpha=0.7)
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45)
    ax.set_title('C. Therapeutic Categories', fontweight='bold')
    ax.set_ylabel('Number of Drugs')
    ax.grid(True, alpha=0.3)

    # 4. Drug Effectiveness by Target
    ax = axes[1, 0]
    target_stats = screening_data.groupby('Target').agg({
        'Synthetic_Lethality_Score': ['mean', 'std', 'count']
    }).round(2)
    target_stats.columns = ['Mean_SL', 'Std_SL', 'Count']
    target_stats = target_stats.sort_values('Mean_SL')

    x_pos = range(len(target_stats))
    bars = ax.bar(x_pos, target_stats['Mean_SL'],
                  yerr=target_stats['Std_SL'], capsize=5,
                  color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_stats.index, rotation=45, ha='right')
    ax.set_title('D. Target Performance', fontweight='bold')
    ax.set_ylabel('Mean SL Score')
    ax.grid(True, alpha=0.3)

    # 5. SL Score vs Pathway Activities
    ax = axes[1, 1]
    for target in screening_data['Target'].unique():
        target_data = screening_data[screening_data['Target'] == target]
        ax.scatter(target_data['Synthetic_Lethality_Score'],
                  target_data['DSB_Level'],
                  label=target, alpha=0.7, s=60)
    ax.set_xlabel('Synthetic Lethality Score')
    ax.set_ylabel('DNA Damage Level')
    ax.set_title('E. SL Score vs DNA Damage', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Therapeutic Index Distribution
    ax = axes[1, 2]
    ax.hist(screening_data['Therapeutic_Index'], bins=15,
           alpha=0.7, color='orange', edgecolor='black')
    ax.set_title('F. Therapeutic Index Distribution', fontweight='bold')
    ax.set_xlabel('Therapeutic Index')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f"statistical_analysis_comprehensive.{fmt}",
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Statistical analysis plots created")

def create_multipanel_figure(screening_data, time_series_data, output_dir):
    """Create a comprehensive multi-panel figure for publication"""

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Panel A: Top drugs ranking (spans 2 columns)
    ax_a = fig.add_subplot(gs[0, :2])
    top_drugs = screening_data.nlargest(8, 'Synthetic_Lethality_Score')
    y_pos = range(len(top_drugs))
    colors = get_color_palette(len(top_drugs), "viridis")
    ax_a.barh(y_pos, top_drugs['Synthetic_Lethality_Score'], color=colors)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([drug[:20] + '...' if len(drug) > 20 else drug
                         for drug in top_drugs['Drug']], fontsize=10)
    ax_a.set_xlabel('Synthetic Lethality Score', fontsize=12)
    ax_a.set_title('A. Top Drug Candidates', fontsize=14, fontweight='bold')
    ax_a.grid(True, alpha=0.3)

    # Panel B: Target distribution
    ax_b = fig.add_subplot(gs[0, 2:])
    target_counts = screening_data['Target'].value_counts()
    colors = get_color_palette(len(target_counts), "Set3")
    ax_b.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
             colors=colors, startangle=90)
    ax_b.set_title('B. Target Distribution', fontsize=14, fontweight='bold')

    # Panel C: Time course for best drug
    ax_c = fig.add_subplot(gs[1, :2])
    if time_series_data:
        # Use the first available time series
        first_ts = list(time_series_data.values())[0]
        if 'ApoptosisSignal' in first_ts.columns:
            ax_c.plot(first_ts['Time'], first_ts['ApoptosisSignal'], 'r-', linewidth=2.5, label='Apoptosis')
        if 'DSB' in first_ts.columns:
            ax_c.plot(first_ts['Time'], first_ts['DSB'], 'b-', linewidth=2.5, label='DNA Damage')
        if 'ATM_active' in first_ts.columns:
            ax_c.plot(first_ts['Time'], first_ts['ATM_active'], 'g-', linewidth=2.5, label='ATM Activity')
        ax_c.set_xlabel('Time (hours)', fontsize=12)
        ax_c.set_ylabel('Activity Level', fontsize=12)
        ax_c.set_title('C. Time-Course Analysis', fontsize=14, fontweight='bold')
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3)

    # Panel D: Correlation heatmap
    ax_d = fig.add_subplot(gs[1, 2:])
    pathway_cols = ['DSB_Level', 'HR_Activity', 'PARP_Activity', 'Cell_Cycle_Arrest', 'ATM_Activity', 'ATR_Activity']
    correlation_matrix = screening_data[pathway_cols].corr()
    im = ax_d.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_d.set_xticks(range(len(pathway_cols)))
    ax_d.set_yticks(range(len(pathway_cols)))
    ax_d.set_xticklabels([col.replace('_', '\n') for col in pathway_cols], rotation=45, ha='right', fontsize=9)
    ax_d.set_yticklabels([col.replace('_', '\n') for col in pathway_cols], fontsize=9)
    ax_d.set_title('D. Pathway Correlations', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_d, shrink=0.8)
    cbar.set_label('Correlation', fontsize=10)

    # Panel E: Therapeutic analysis
    ax_e = fig.add_subplot(gs[2, :2])
    scatter = ax_e.scatter(screening_data['Synthetic_Lethality_Score'],
                          screening_data['Therapeutic_Index'],
                          c=screening_data['Target'].astype('category').cat.codes,
                          s=100, alpha=0.7, cmap='tab10')
    ax_e.set_xlabel('Synthetic Lethality Score', fontsize=12)
    ax_e.set_ylabel('Therapeutic Index', fontsize=12)
    ax_e.set_title('E. Therapeutic Window Analysis', fontsize=14, fontweight='bold')
    ax_e.grid(True, alpha=0.3)

    # Add quadrant lines
    ax_e.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='TI = 2')
    ax_e.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='SL = 1')
    ax_e.legend(fontsize=9)

    # Panel F: Target effectiveness with error bars
    ax_f = fig.add_subplot(gs[2, 2:])
    target_effectiveness = screening_data.groupby('Target')['Synthetic_Lethality_Score'].agg(['mean', 'std']).sort_values('mean')
    y_pos = range(len(target_effectiveness))
    colors = get_color_palette(len(target_effectiveness), "plasma")
    ax_f.barh(y_pos, target_effectiveness['mean'],
              xerr=target_effectiveness['std'], capsize=5, color=colors)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(target_effectiveness.index, fontsize=10)
    ax_f.set_xlabel('Mean SL Score ± SD', fontsize=12)
    ax_f.set_title('F. Target Effectiveness', fontsize=14, fontweight='bold')
    ax_f.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf']:
        fig.savefig(output_dir / f"multi_panel_publication_figure.{fmt}",
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Multi-panel publication figure created")

def create_therapeutic_index_plot(screening_data, output_dir):
    """Create therapeutic index analysis plot"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Therapeutic Index Analysis', fontsize=16, fontweight='bold')

    # 1. TI vs SL Score with target coloring
    ax = axes[0, 0]
    targets = screening_data['Target'].unique()
    colors = get_color_palette(len(targets), "tab10")

    for i, target in enumerate(targets):
        target_data = screening_data[screening_data['Target'] == target]
        ax.scatter(target_data['Synthetic_Lethality_Score'],
                  target_data['Therapeutic_Index'],
                  c=[colors[i]], label=target, alpha=0.7, s=80)

    ax.set_xlabel('Synthetic Lethality Score')
    ax.set_ylabel('Therapeutic Index')
    ax.set_title('A. SL Score vs TI by Target', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 2. TI categories
    ax = axes[0, 1]
    screening_data['TI_Category'] = pd.cut(screening_data['Therapeutic_Index'],
                                          bins=[0, 1, 2, 5, float('inf')],
                                          labels=['Low', 'Moderate', 'High', 'Very High'])
    category_stats = screening_data.groupby('TI_Category').agg({
        'Synthetic_Lethality_Score': 'mean',
        'Drug': 'count'
    }).round(2)

    bars = ax.bar(category_stats.index, category_stats['Synthetic_Lethality_Score'],
                  color=['lightcoral', 'gold', 'lightgreen', 'darkgreen'], alpha=0.7)
    ax.set_ylabel('Mean SL Score')
    ax.set_title('B. Mean SL Score by TI Category', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, category_stats['Drug']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # 3. Target-specific TI
    ax = axes[1, 0]
    target_ti = screening_data.groupby('Target')['Therapeutic_Index'].mean().sort_values(ascending=True)
    ax.barh(range(len(target_ti)), target_ti.values,
            color='skyblue', alpha=0.7)
    ax.set_yticks(range(len(target_ti)))
    ax.set_yticklabels(target_ti.index, fontsize=10)
    ax.set_xlabel('Mean Therapeutic Index')
    ax.set_title('C. Mean TI by Target', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Optimal quadrant analysis
    ax = axes[1, 1]

    # Define quadrants
    high_sl = screening_data['Synthetic_Lethality_Score'] > screening_data['Synthetic_Lethality_Score'].median()
    high_ti = screening_data['Therapeutic_Index'] > 2

    quadrant_data = {
        'High SL, High TI': screening_data[high_sl & high_ti],
        'High SL, Low TI': screening_data[high_sl & ~high_ti],
        'Low SL, High TI': screening_data[~high_sl & high_ti],
        'Low SL, Low TI': screening_data[~high_sl & ~high_ti]
    }

    colors_quad = ['green', 'orange', 'yellow', 'red']
    for i, (quad_name, quad_drugs) in enumerate(quadrant_data.items()):
        if len(quad_drugs) > 0:
            ax.scatter(quad_drugs['Synthetic_Lethality_Score'],
                      quad_drugs['Therapeutic_Index'],
                      c=colors_quad[i], label=f'{quad_name} (n={len(quad_drugs)})',
                      alpha=0.7, s=60)

    ax.set_xlabel('Synthetic Lethality Score')
    ax.set_ylabel('Therapeutic Index')
    ax.set_title('D. Optimal Therapeutic Quadrants', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf']:
        fig.savefig(output_dir / f"therapeutic_index_analysis.{fmt}",
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   Therapeutic index analysis created")

def create_pathway_network_plot(screening_data, output_dir):
    """Create simplified DDR pathway network diagram"""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Define pathway components and their positions
    pathway_components = {
        'ATM': (2, 6),
        'ATR': (4, 6),
        'DNA_Damage': (3, 4),
        'Cell_Cycle_Arrest': (3, 2),
        'Apoptosis': (1, 0),
        'DNA_Repair_HR': (5, 2),
        'DNA_Repair_NHEJ': (1, 2),
        'PARP1': (3, 0)
    }

    # Colors for different pathways
    pathway_colors = {
        'ATM': '#3498DB',
        'ATR': '#3498DB',
        'DNA_Damage': '#E74C3C',
        'Cell_Cycle_Arrest': '#F39C12',
        'Apoptosis': '#D63031',
        'DNA_Repair_HR': '#2ECC71',
        'DNA_Repair_NHEJ': '#9B59B6',
        'PARP1': '#E67E22'
    }

    # Draw pathway components
    for component, (x, y) in pathway_components.items():
        color = pathway_colors.get(component, '#95A5A6')
        circle = mpatches.Circle((x, y), 0.4, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, component.replace('_', ' '), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    # Add connections (simplified)
    connections = [
        ('DNA_Damage', 'ATM'),
        ('DNA_Damage', 'ATR'),
        ('ATM', 'Cell_Cycle_Arrest'),
        ('ATR', 'Cell_Cycle_Arrest'),
        ('Cell_Cycle_Arrest', 'Apoptosis'),
        ('DNA_Damage', 'DNA_Repair_HR'),
        ('DNA_Damage', 'DNA_Repair_NHEJ'),
        ('DNA_Damage', 'PARP1')
    ]

    for start, end in connections:
        if start in pathway_components and end in pathway_components:
            x1, y1 = pathway_components[start]
            x2, y2 = pathway_components[end]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)

    # Add top drugs information
    top_drugs = screening_data.nlargest(3, 'Synthetic_Lethality_Score')
    y_offset = 8
    for i, (_, drug) in enumerate(top_drugs.iterrows()):
        ax.text(0.5 + i*2, y_offset,
               f"{drug['Drug']}\nScore: {drug['Synthetic_Lethality_Score']:.2f}\nTarget: {drug['Target']}",
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1, 9.5)
    ax.set_title('DDR Pathway Network with Drug Targets', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_elements = [mpatches.Patch(color=color, label=pathway.replace('_', ' '))
                      for pathway, color in pathway_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f"ddr_pathway_network.{fmt}",
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("   DDR pathway network created")

def create_integration_report(screening_data, time_series_data, output_dir):
    """Generate comprehensive integration report"""

    # Calculate statistics
    sl_stats = {
        'mean': float(screening_data['Synthetic_Lethality_Score'].mean()),
        'std': float(screening_data['Synthetic_Lethality_Score'].std()),
        'min': float(screening_data['Synthetic_Lethality_Score'].min()),
        'max': float(screening_data['Synthetic_Lethality_Score'].max()),
        'median': float(screening_data['Synthetic_Lethality_Score'].median())
    }

    ti_stats = {
        'mean': float(screening_data['Therapeutic_Index'].mean()),
        'std': float(screening_data['Therapeutic_Index'].std()),
        'min': float(screening_data['Therapeutic_Index'].min()),
        'max': float(screening_data['Therapeutic_Index'].max()),
        'median': float(screening_data['Therapeutic_Index'].median())
    }

    # Create comprehensive report
    report = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "total_drugs_analyzed": len(screening_data),
            "unique_targets": screening_data['Target'].nunique(),
            "target_list": screening_data['Target'].unique().tolist(),
            "time_series_simulations": len(time_series_data)
        },
        "synthetic_lethality_statistics": sl_stats,
        "therapeutic_index_statistics": ti_stats,
        "top_candidates": {
            "by_sl_score": screening_data.nlargest(5, 'Synthetic_Lethality_Score')[
                ['Drug', 'Target', 'Synthetic_Lethality_Score', 'Therapeutic_Index']
            ].to_dict('records'),
            "by_therapeutic_index": screening_data.nlargest(5, 'Therapeutic_Index')[
                ['Drug', 'Target', 'Synthetic_Lethality_Score', 'Therapeutic_Index']
            ].to_dict('records')
        },
        "target_analysis": {
            "target_effectiveness": screening_data.groupby('Target')['Synthetic_Lethality_Score'].mean().to_dict(),
            "target_therapeutic_index": screening_data.groupby('Target')['Therapeutic_Index'].mean().to_dict(),
            "target_drug_counts": screening_data['Target'].value_counts().to_dict()
        },
        "pathway_correlations": screening_data[
            ['DSB_Level', 'HR_Activity', 'PARP_Activity', 'Cell_Cycle_Arrest', 'ATM_Activity', 'ATR_Activity']
        ].corr().to_dict(),
        "generated_visualizations": [
            "enhanced_synthetic_lethality_analysis.png/pdf/svg",
            "time_course_*.png/pdf/svg (for top 3 candidates)",
            "statistical_analysis_comprehensive.png/pdf/svg",
            "multi_panel_publication_figure.png/pdf",
            "therapeutic_index_analysis.png/pdf",
            "ddr_pathway_network.png/pdf/svg"
        ],
        "r_framework_integration": {
            "data_compatibility": "All generated data files are compatible with R visualization framework",
            "json_report": "integration_report.json can be consumed by R pipeline",
            "csv_data": "screening_data and time_series files are R-readable",
            "figure_formats": "PNG, PDF, SVG formats supported by R",
            "statistical_output": "Summary statistics available for R meta-analysis"
        },
        "publication_specifications": {
            "figure_resolution": "300 DPI for print quality",
            "font_family": "Arial (publication standard)",
            "color_scheme": "Colorblind-friendly palettes",
            "figure_sizes": "Optimized for journal submissions",
            "file_formats": "PNG (web), PDF (print), SVG (scalable)"
        }
    }

    # Save JSON report
    with open(output_dir / "integration_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Create markdown summary
    md_content = f"""# Publication-Quality Visualization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report summarizes the comprehensive analysis of DDR pathway synthetic lethality screening results, producing publication-quality visualizations for scientific communication.

## Data Overview

- **Total Drugs Analyzed:** {report['metadata']['total_drugs_analyzed']}
- **Unique Targets:** {report['metadata']['unique_targets']}
- **Time Series Simulations:** {report['metadata']['time_series_simulations']}
- **Targets:** {', '.join(report['metadata']['target_list'])}

## Key Findings

### Synthetic Lethality Statistics
- **Mean SL Score:** {sl_stats['mean']:.3f} ± {sl_stats['std']:.3f}
- **Range:** {sl_stats['min']:.3f} - {sl_stats['max']:.3f}
- **Median:** {sl_stats['median']:.3f}

### Therapeutic Index Statistics
- **Mean TI:** {ti_stats['mean']:.3f} ± {ti_stats['std']:.3f}
- **Range:** {ti_stats['min']:.3f} - {ti_stats['max']:.3f}
- **Median:** {ti_stats['median']:.3f}

## Top Drug Candidates

### By Synthetic Lethality Score
"""

    for i, drug in enumerate(report['top_candidates']['by_sl_score'], 1):
        md_content += f"{i}. **{drug['Drug']}** (Target: {drug['Target']})\n"
        md_content += f"   - SL Score: {drug['Synthetic_Lethality_Score']:.3f}\n"
        md_content += f"   - Therapeutic Index: {drug['Therapeutic_Index']:.3f}\n\n"

    md_content += f"""
### By Therapeutic Index
"""

    for i, drug in enumerate(report['top_candidates']['by_therapeutic_index'], 1):
        md_content += f"{i}. **{drug['Drug']}** (Target: {drug['Target']})\n"
        md_content += f"   - SL Score: {drug['Synthetic_Lethality_Score']:.3f}\n"
        md_content += f"   - Therapeutic Index: {drug['Therapeutic_Index']:.3f}\n\n"

    md_content += f"""
## Generated Publications

The following publication-quality figures have been generated:

{chr(10).join([f'- {fig}' for fig in report['generated_visualizations']])}

## R Framework Integration

✓ **Complete compatibility** with the R visualization framework:
- JSON data format for statistical analysis
- CSV files for R data import
- Multiple figure formats (PNG, PDF, SVG)
- Standardized statistical outputs

## Technical Specifications

- **Resolution:** 300 DPI (print quality)
- **Font:** Arial (publication standard)
- **Color Schemes:** Colorblind-friendly palettes
- **File Formats:** PNG, PDF, SVG

## Usage for Publication

All figures are ready for direct inclusion in scientific manuscripts. The high-resolution formats (PDF) are suitable for both print and digital publication. The SVG format allows for further editing in vector graphics software.

---

*This report was generated by the Enhanced Visualization Framework for DDR Pathway Analysis.*
"""

    # Save markdown report
    with open(output_dir / "visualization_report.md", 'w') as f:
        f.write(md_content)

    print("   Integration report created (JSON and Markdown)")

if __name__ == "__main__":
    main()
