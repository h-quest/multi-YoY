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

def degradation_multi_year_on_year(energy_normalized: pd.Series, recenter: bool = True,
                                   exceedance_prob: int = 95, confidence_level: int = 95):
    
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

    # Sort the index and reset column names
    energy_normalized = energy_normalized.sort_index()
    energy_normalized.name = 'energy'
    energy_normalized.index.name = 'dt'

    # Calculate the number of years in the dataset
    total_years = energy_normalized.index[-1].year - energy_normalized.index[0].year

    # Renormalize to the median of the first year if 'recenter' is True
    if recenter:
        first_year_end = energy_normalized.index[0] + pd.Timedelta(days=365)
        renorm_factor = energy_normalized.loc[:first_year_end].median()
    else:
        renorm_factor = 1.0

    # Normalize the energy series
    energy_normalized = energy_normalized / renorm_factor
    energy_normalized = energy_normalized.reset_index()

    # Create a DataFrame to store multi-annual Year-on-Year (YoY) instances
    df_rd = pd.DataFrame()

    # Iterate over each year to calculate YoY degradation
    for year_diff in range(1, total_years + 1):
        energy_normalized['dt_shifted'] = energy_normalized.dt + pd.DateOffset(years=year_diff)
        df = pd.merge_asof(energy_normalized[['dt', 'energy']], energy_normalized,
                           left_on='dt', right_on='dt_shifted',
                           suffixes=['', '_right'],
                           tolerance=pd.Timedelta('8D')
                           ).drop(['dt_shifted'], axis=1).dropna()
        df['time_diff_years'] = (df.dt - df.dt_right) / pd.Timedelta(hours=1) / 8766.0
        df_rd = pd.concat([df_rd, df])
    
    df_rd = df_rd.reset_index(drop=True)
    df_rd['yoy'] = 100.0 * (df_rd.energy - df_rd.energy_right) / (df_rd.time_diff_years)
    df_rd.index = df_rd.dt
    
    if df_rd.empty:
        raise ValueError("No year-over-year aggregated data pairs found. Check the input data.")
    
    # Calculate the median Year-on-Year degradation and bootstrap confidence intervals
    yoy_result = df_rd['yoy'].dropna()
    Rd_pct = yoy_result.median()
    
    n_samples = len(yoy_result)
    reps = 10000
    bootstrapped_medians = np.median(np.random.choice(yoy_result, (n_samples, reps), replace=True), axis=0)
    
    half_ci_width = confidence_level / 2.0
    Rd_CI = np.percentile(bootstrapped_medians, [50 - half_ci_width, 50 + half_ci_width])
    exceedance_level = np.percentile(bootstrapped_medians, 100 - exceedance_prob)

    # Collect additional calculation information
    calc_info = {
        'YoY_values': df_rd,
        'renormalizing_factor': renorm_factor,
        'monte_carlo_medians': bootstrapped_medians,
        'exceedance_level': exceedance_level
    }

    return Rd_pct, Rd_CI, calc_info, yoy_result