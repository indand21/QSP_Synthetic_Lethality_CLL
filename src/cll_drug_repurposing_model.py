
# Simplified DDR Model for In Silico Drug Repurposing in ATM-deficient CLL

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

class SimplifiedDDRModel:
    """
    A simplified Quantitative Systems Pharmacology (QSP) model for the DNA Damage Response (DDR) 
    pathway, tailored to simulate synthetic lethality in ATM-deficient Chronic Lymphocytic Leukemia (CLL).
    
    This model focuses on the core interactions between ATM, ATR, and key downstream effectors 
    (CHK1, p53) to predict cell fate (survival vs. apoptosis) under different genetic backgrounds 
    (ATM-proficient vs. ATM-deficient) and drug interventions.
    """
    
    def __init__(self, atm_proficient=True):
        """
        Initializes the model with parameters for either ATM-proficient (WT) or ATM-deficient cells.
        
        Args:
            atm_proficient (bool): If True, models a cell with functional ATM. 
                                 If False, models an ATM-deficient cell, simulating a key characteristic of 
                                 aggressive CLL.
        """
        self.atm_proficient = atm_proficient
        self.species_names = ["DSB", "ATM_active", "ATR_active", "CHK1_active", "p53_active", "ApoptosisSignal"]
        self.params = self._get_default_params()
        self.initial_conditions = self._get_initial_conditions()

    def _get_default_params(self):
        """
        Defines the kinetic parameters of the model. 
        ATM activation rate is set to near zero if the cell is ATM-deficient.
        """
        params = {
            # Damage and Sensor Activation
            'k_dsb_gen': 0.05,       # Rate of double-strand break generation
            'k_atm_act': 1.5 if self.atm_proficient else 0.01, # ATM activation by DSBs
            'k_atr_act': 0.8,        # ATR activation (basal and in response to stalled forks, simplified)
            'k_atm_deact': 0.2,      # Deactivation of ATM
            'k_atr_deact': 0.15,     # Deactivation of ATR
            
            # Downstream Signaling
            'k_chk1_act_by_atr': 1.2, # CHK1 activation by ATR
            'k_p53_act_by_atm': 1.0,  # p53 activation by ATM
            'k_chk1_deact': 0.25,
            'k_p53_deact': 0.1,
            
            # Apoptosis Induction
            'k_apoptosis_p53': 0.03,  # Apoptosis driven by p53
            'k_apoptosis_unrepaired': 0.1, # Apoptosis from persistent DSBs (synthetic lethality)
            'apoptosis_threshold': 100.0 # Threshold for cell death
        }
        return params

    def _get_initial_conditions(self):
        """ Defines the initial concentrations of the molecular species. """
        return np.array([0.0, 0.0, 5.0, 5.0, 10.0, 0.0]) # Start with some basal ATR/CHK1/p53

    def _ode_system(self, y, t, drug_effects):
        """
        Defines the system of ordinary differential equations (ODEs) for the DDR pathway.
        
        Args:
            y (list): Array of current concentrations of the species.
            t (float): Current time point (for the ODE solver).
            drug_effects (dict): A dictionary specifying the fractional inhibition of drug targets.
        """
        DSB, ATM_active, ATR_active, CHK1_active, p53_active, ApoptosisSignal = y
        p = self.params
        
        # Drug inhibition factors (1.0 = no inhibition, 0.0 = full inhibition)
        inhibit_atr = 1.0 - drug_effects.get('ATR', 0.0)
        inhibit_chk1 = 1.0 - drug_effects.get('CHK1', 0.0)
        
        # Equations
        dDSB_dt = p['k_dsb_gen'] - (p['k_atm_act'] * DSB * ATM_active) - (p['k_atr_act'] * DSB) # Simplified repair
        dATM_active_dt = p['k_atm_act'] * DSB - p['k_atm_deact'] * ATM_active
        dATR_active_dt = p['k_atr_act'] * DSB - p['k_atr_deact'] * ATR_active
        dCHK1_active_dt = (p['k_chk1_act_by_atr'] * ATR_active * inhibit_atr) - (p['k_chk1_deact'] * CHK1_active * inhibit_chk1)
        dp53_active_dt = p['k_p53_act_by_atm'] * ATM_active - p['k_p53_deact'] * p53_active
        dApoptosisSignal_dt = (p['k_apoptosis_p53'] * p53_active) + (p['k_apoptosis_unrepaired'] * DSB * (1-inhibit_atr)) # Unrepaired damage is more lethal with ATRi
        
        return [dDSB_dt, dATM_active_dt, dATR_active_dt, dCHK1_active_dt, dp53_active_dt, dApoptosisSignal_dt]

    def run_simulation(self, duration, drug_effects={}):
        """
        Runs the ODE simulation for a given duration and drug intervention.
        
        Args:
            duration (int): The total time (in hours) for the simulation.
            drug_effects (dict): Dictionary of drug inhibitions.
            
        Returns:
            pandas.DataFrame: A DataFrame containing the time-course concentrations of all species.
        """
        time_points = np.linspace(0, duration, 100)
        solution = odeint(self._ode_system, self.initial_conditions, time_points, args=(drug_effects,))
        results_df = pd.DataFrame(solution, columns=self.species_names)
        results_df['Time'] = time_points
        return results_df

def run_virtual_screen(drugs_to_screen):
    """
    Performs a virtual screen to evaluate the synthetic lethal effect of different drugs.
    It compares the apoptosis level in ATM-deficient vs. ATM-proficient cells.
    
    Args:
        drugs_to_screen (dict): A dictionary where keys are drug names and values are dicts 
                                specifying the target and inhibition level.
                                
    Returns:
        pandas.DataFrame: A DataFrame summarizing the screening results, including the predicted
                          synthetic lethality score for each drug.
    """
    screen_results = []

    for drug_name, props in drugs_to_screen.items():
        # Simulate in ATM-proficient (WT) cells
        model_wt = SimplifiedDDRModel(atm_proficient=True)
        sim_wt = model_wt.run_simulation(duration=48, drug_effects=props['effects'])
        apoptosis_wt = sim_wt['ApoptosisSignal'].iloc[-1]
        
        # Simulate in ATM-deficient cells
        model_atm_def = SimplifiedDDRModel(atm_proficient=False)
        sim_atm_def = model_atm_def.run_simulation(duration=48, drug_effects=props['effects'])
        apoptosis_atm_def = sim_atm_def['ApoptosisSignal'].iloc[-1]
        
        # Calculate synthetic lethality score (ratio of apoptosis)
        sl_score = apoptosis_atm_def / (apoptosis_wt + 1e-9) # Add small epsilon to avoid division by zero
        
        screen_results.append({
            'Drug': drug_name,
            'Target': props['target'],
            'Apoptosis (ATM-WT)': apoptosis_wt,
            'Apoptosis (ATM-deficient)': apoptosis_atm_def,
            'Synthetic Lethality Score': sl_score
        })
        
    return pd.DataFrame(screen_results).sort_values(by='Synthetic Lethality Score', ascending=False)

def plot_results(df, title):
    """
    Generates and saves a bar plot of the virtual screening results.
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Synthetic Lethality Score', y='Drug', data=df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Synthetic Lethality Score (Higher is Better)', fontsize=12)
    plt.ylabel('Drug Candidate', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    print(f'Plot saved to {title.replace(" ", "_")}.png')

if __name__ == '__main__':
    # Define the drug library for virtual screening
    # Inhibition is set to 90% for the primary target
    drug_library = {
        'AZD6738 (Ceralasertib)': {'target': 'ATR', 'effects': {'ATR': 0.9}},
        'VE-822 (Berzosertib)': {'target': 'ATR', 'effects': {'ATR': 0.9}},
        'Prexasertib': {'target': 'CHK1', 'effects': {'CHK1': 0.9}},
        'Adavosertib': {'target': 'WEE1', 'effects': {}}, # WEE1 not in this simplified model, expect low score
        'Olaparib': {'target': 'PARP', 'effects': {}}, # PARP not in this model, expect low score
        'ATR_plus_PARP_synergy': {'target': 'ATR+PARP', 'effects': {'ATR': 0.9}} # Simulating synergy by focusing on ATR effect
    }
    
    # Run the screen
    print("Running virtual drug screen...")
    screening_results_df = run_virtual_screen(drug_library)
    
    # Display and plot results
    print("\n--- Virtual Screening Results ---")
    print(screening_results_df.to_string())
    
    plot_results(screening_results_df, 'DDR Inhibitor Synthetic Lethality Screen in ATM-deficient CLL')
    
    # Example of detailed time-course simulation for the top drug
    top_drug_name = screening_results_df['Drug'].iloc[0]
    top_drug_props = drug_library[top_drug_name]

    print(f"\n--- Detailed Simulation for {top_drug_name} ---")
    model_wt_drug = SimplifiedDDRModel(atm_proficient=True)
    sim_wt_drug = model_wt_drug.run_simulation(48, top_drug_props['effects'])

    model_atm_def_drug = SimplifiedDDRModel(atm_proficient=False)
    sim_atm_def_drug = model_atm_def_drug.run_simulation(48, top_drug_props['effects'])

    # Plotting detailed simulation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sim_wt_drug.plot(x='Time', y='ApoptosisSignal', ax=ax1, title='ATM-Proficient (WT) + ' + top_drug_name, legend=False)
    ax1.set_ylabel('Apoptosis Signal')
    ax1.grid(True)
    sim_atm_def_drug.plot(x='Time', y='ApoptosisSignal', ax=ax2, title='ATM-Deficient + ' + top_drug_name, color='red', legend=False)
    ax2.grid(True)
    plt.suptitle(f'Predicted Apoptotic Response to {top_drug_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('top_drug_simulation.png')
    print('Detailed simulation plot saved to top_drug_simulation.png')
