# --- coding: utf-8 ---

# Author: Hugo Quest (hugo.quest@epfl.ch / hugo.quest@3s-solar.swiss)
# Affiliations: EPFL PV-Lab / 3S Swiss Solar Solutions AG

''' MULTI-ANNUAL YEAR-ON-YEAR (MULTI-YOY) / YoYo 

This module contains a function for calculating the Performance Loss Rate (PLR)
using the multi-annual Year-on-Year (multi-YoY) method

'''

### ------------------------------------------------------------------- ###
### ---------------------------- IMPORTS ------------------------------ ###
### ------------------------------------------------------------------- ###

import pandas as pd
import numpy as np
import datetime as dt

### ------------------------------------------------------------------- ###  
### ---------------------------- FUNCTION ----------------------------- ### 
### ------------------------------------------------------------------- ###

def degradation_multi_year_on_year(energy_normalized, recenter=True,
                                   exceedance_prob=95, confidence_level=95):
    
    """
    Calculate the multi-annual year-on-year performance loss rate from a normalized energy time series.

    Args:
    * energy_normalized (pd.Series): Time-indexed series of normalized energy outputs.
    * recenter (bool): If True, renormalize the energy data to the median of the first year.
    * exceedance_prob (int): The exceedance probability percentage for uncertainty analysis.
    * confidence_level (int): The confidence level for bootstrap uncertainty bounds.

    Returns:
    * tuple: Contains the median performance loss rate, confidence interval, calculation details, 
            and year-on-year degradation values.
    """

    ### Pre-processing for PLR analysis pipeline
    energy_normalized = energy_normalized.sort_index()
    energy_normalized.name = 'energy'
    energy_normalized.index.name = 'dt'

    ### Define the number of years for the multi-YoY analysis
    max_year_diff = energy_normalized.index[-1].year - energy_normalized.index[0].year

    ### Recentering data by renormalising to first year
    if recenter:
        start = energy_normalized.index[0]
        one_year = start + pd.Timedelta(days=365)
        renorm_factor = energy_normalized[start:one_year].median()
    else:
        renorm_factor = 1.0
    
    energy_normalized = energy_normalized.reset_index()
    energy_normalized['energy'] = energy_normalized['energy'] / renorm_factor
    
    ### Creating the dataset for multi-YoY comparisons 
    df_rd = pd.DataFrame()
    for i in np.array(range(1, max_year_diff+1)):
        energy_normalized['dt_shifted'] = energy_normalized.dt + pd.DateOffset(years=i)
        df = pd.merge_asof(energy_normalized[['dt', 'energy']], energy_normalized,
                           left_on='dt', right_on='dt_shifted',
                           suffixes=['', '_right'],
                           tolerance=pd.Timedelta('8D')
                           ).drop(['dt_shifted'], axis=1).dropna()
        df['time_diff_years'] = (df.dt - df.dt_right) / pd.Timedelta(hours=1) / 8766.0
        df_rd = pd.concat([df_rd, df])
    df_rd = df_rd.reset_index(drop=True)

    ### Computing the YoY instances
    df_rd['yoy'] = 100.0 * (df_rd.energy - df_rd.energy_right) / (df_rd.time_diff_years)
    df_rd.index = df_rd.dt
    
    if df_rd.empty:
        raise ValueError('No year-over-year aggregated data pairs found')
    
    ### Computing final PLR and bootstrap uncertainty
    yoy_result = df_rd['yoy'].dropna()
    Rd_pct = yoy_result.median()
    n_samples = len(yoy_result)
    reps = 10000
    bootstrapped_samples = np.random.choice(yoy_result, (n_samples, reps), replace=True)
    median_bootstrapped = np.median(bootstrapped_samples, axis=0)
    half_ci_width = confidence_level / 2.0
    Rd_CI = np.percentile(median_bootstrapped, [50 - half_ci_width, 50 + half_ci_width])
    exceedance_level = np.percentile(median_bootstrapped, 100 - exceedance_prob)

    calc_info = {
        'YoY_values': df_rd,
        'renormalizing_factor': renorm_factor,
        'monte_carlo_medians': median_bootstrapped,
        'exceedance_level': exceedance_level
    }

    return (Rd_pct, Rd_CI, calc_info, yoy_result)