Climate Change Assessment
=========================

This example demonstrates how to use MarrmotFlow for assessing climate change impacts on hydrology using multiple climate scenarios and models.

Overview
--------

Climate change impact assessment typically involves:

1. **Historical baseline analysis** using observed/reanalysis data
2. **Future climate projections** from Global Climate Models (GCMs)
3. **Multi-scenario analysis** using different emission scenarios
4. **Uncertainty quantification** across climate models and hydrological models
5. **Impact assessment** comparing future with historical conditions

This example shows how to implement such an analysis with MarrmotFlow.

Prerequisites
-------------

Climate data from multiple sources:

.. code-block:: bash

   data/
   ├── catchments.shp
   ├── historical/
   │   ├── era5_1990_2019.nc
   │   └── observed_discharge.csv
   ├── gcm_historical/
   │   ├── CanESM5_historical_1990_2019.nc
   │   ├── CNRM-CM6_historical_1990_2019.nc
   │   └── UKESM1_historical_1990_2019.nc
   └── gcm_future/
       ├── CanESM5_ssp245_2070_2099.nc
       ├── CanESM5_ssp585_2070_2099.nc
       ├── CNRM-CM6_ssp245_2070_2099.nc
       ├── CNRM-CM6_ssp585_2070_2099.nc
       ├── UKESM1_ssp245_2070_2099.nc
       └── UKESM1_ssp585_2070_2099.nc

Step 1: Setup and Configuration
-------------------------------

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path
   from marrmotflow import MARRMOTWorkflow
   
   # Set up analysis configuration
   BASELINE_PERIOD = "1990-2019"
   FUTURE_PERIOD = "2070-2099"
   
   # Climate models and scenarios
   CLIMATE_MODELS = ["CanESM5", "CNRM-CM6", "UKESM1"]
   SCENARIOS = ["ssp245", "ssp585"]
   
   # Hydrological models for uncertainty assessment
   HYDRO_MODELS = {"HBV-96": 7, "GR4J": 37}

Define Analysis Framework
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ClimateChangeAssessment:
       """Framework for climate change impact assessment using MarrmotFlow."""
       
       def __init__(self, catchment_file, baseline_period, future_period):
           self.catchments = gpd.read_file(catchment_file)
           self.baseline_period = baseline_period
           self.future_period = future_period
           self.results = {}
           
       def load_climate_data(self, data_path, pattern):
           """Load climate data matching a pattern."""
           data_files = list(Path(data_path).glob(pattern))
           datasets = {}
           
           for file in data_files:
               # Extract model/scenario info from filename
               parts = file.stem.split('_')
               key = '_'.join(parts[:-1])  # Remove time period
               datasets[key] = xr.open_dataset(file)
           
           return datasets
           
       def create_workflows(self, climate_data, period_name):
           """Create workflows for all climate model/scenario combinations."""
           workflows = {}
           
           for climate_key, climate_ds in climate_data.items():
               for hydro_name, hydro_model in HYDRO_MODELS.items():
                   workflow_name = f"{period_name}_{climate_key}_{hydro_name}"
                   
                   # Save climate data to temporary file for workflow
                   temp_file = f"temp_{workflow_name}.nc"
                   climate_ds.to_netcdf(temp_file)
                   
                   workflows[workflow_name] = MARRMOTWorkflow(
                       name=workflow_name,
                       cat=self.catchments,
                       forcing_files=temp_file,
                       forcing_vars=self._get_variable_mapping(climate_ds),
                       forcing_units=self._get_units_mapping(),
                       model_number=hydro_model,
                       pet_method="penman_monteith"
                   )
           
           return workflows
           
       def _get_variable_mapping(self, dataset):
           """Get variable mapping based on dataset structure."""
           var_map = {}
           
           # Common CMIP6 variable names
           if 'pr' in dataset.data_vars:
               var_map['precip'] = 'pr'
           elif 'precipitation' in dataset.data_vars:
               var_map['precip'] = 'precipitation'
               
           if 'tas' in dataset.data_vars:
               var_map['temp'] = 'tas'
           elif 'temperature' in dataset.data_vars:
               var_map['temp'] = 'temperature'
               
           return var_map
           
       def _get_units_mapping(self):
           """Get standard units mapping for climate data."""
           return {
               'precip': 'kg m-2 s-1',  # CMIP6 standard
               'temp': 'K'              # CMIP6 standard
           }

   # Initialize assessment
   assessment = ClimateChangeAssessment(
       "data/catchments.shp",
       BASELINE_PERIOD,
       FUTURE_PERIOD
   )

Step 2: Historical Analysis
---------------------------

Baseline Period Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load historical data
   print("Loading historical climate data...")
   
   # Observational baseline (ERA5)
   era5_data = xr.open_dataset("data/historical/era5_1990_2019.nc")
   
   # GCM historical runs for bias assessment
   gcm_historical = assessment.load_climate_data(
       "data/gcm_historical/",
       "*_historical_*.nc"
   )
   
   print(f"Loaded {len(gcm_historical)} GCM historical datasets")
   
   # Create workflows for historical period
   historical_workflows = {}
   
   # ERA5 baseline
   for hydro_name, hydro_model in HYDRO_MODELS.items():
       workflow_name = f"ERA5_baseline_{hydro_name}"
       
       historical_workflows[workflow_name] = MARRMOTWorkflow(
           name=workflow_name,
           cat=assessment.catchments,
           forcing_files="data/historical/era5_1990_2019.nc",
           forcing_vars={"precip": "total_precipitation", "temp": "2m_temperature"},
           forcing_units={"precip": "m/day", "temp": "kelvin"},
           model_number=hydro_model,
           pet_method="penman_monteith"
       )
   
   # GCM historical workflows
   gcm_historical_workflows = assessment.create_workflows(gcm_historical, "historical")
   historical_workflows.update(gcm_historical_workflows)
   
   print(f"Created {len(historical_workflows)} historical workflows")

Model Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   def load_observed_discharge():
       """Load observed discharge data for validation."""
       obs_data = pd.read_csv("data/historical/observed_discharge.csv", 
                             parse_dates=['date'], index_col='date')
       return obs_data['discharge']  # mm/day

   def calculate_bias_correction(gcm_hist, era5_ref, method='multiplicative'):
       """Calculate bias correction factors for GCM data."""
       
       if method == 'multiplicative':
           # For precipitation
           correction_factor = era5_ref.mean() / gcm_hist.mean()
           corrected = gcm_hist * correction_factor
       else:
           # Additive bias correction (for temperature)
           correction_factor = era5_ref.mean() - gcm_hist.mean()
           corrected = gcm_hist + correction_factor
           
       return corrected, correction_factor

   # Load observations for validation
   observed_discharge = load_observed_discharge()
   
   print(f"Observed discharge period: {observed_discharge.index.min()} to {observed_discharge.index.max()}")
   print(f"Mean observed discharge: {observed_discharge.mean():.2f} mm/day")

Step 3: Future Climate Projections
----------------------------------

Load Future Climate Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load future climate projections
   print("Loading future climate projections...")
   
   future_climate = {}
   
   for scenario in SCENARIOS:
       future_climate[scenario] = assessment.load_climate_data(
           "data/gcm_future/",
           f"*_{scenario}_*.nc"
       )
       print(f"Loaded {len(future_climate[scenario])} datasets for {scenario}")

Apply Bias Correction
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def apply_bias_correction_to_future(gcm_future, gcm_historical, era5_baseline):
       """Apply bias correction to future climate projections."""
       
       corrected_future = gcm_future.copy()
       
       # Precipitation (multiplicative correction)
       if 'pr' in gcm_future.data_vars:
           _, precip_factor = calculate_bias_correction(
               gcm_historical['pr'], era5_baseline['total_precipitation'], 
               method='multiplicative'
           )
           corrected_future['pr'] = gcm_future['pr'] * precip_factor
       
       # Temperature (additive correction)
       if 'tas' in gcm_future.data_vars:
           _, temp_factor = calculate_bias_correction(
               gcm_historical['tas'], era5_baseline['2m_temperature'],
               method='additive'
           )
           corrected_future['tas'] = gcm_future['tas'] + temp_factor
           
       return corrected_future

   # Apply bias correction to future projections
   print("Applying bias correction to future projections...")
   
   corrected_future = {}
   for scenario in SCENARIOS:
       corrected_future[scenario] = {}
       
       for gcm_name, gcm_data in future_climate[scenario].items():
           # Extract model name from full identifier
           model_name = gcm_name.split('_')[0]
           
           # Find corresponding historical data
           hist_key = f"{model_name}_historical"
           if hist_key in gcm_historical:
               corrected_future[scenario][gcm_name] = apply_bias_correction_to_future(
                   gcm_data, gcm_historical[hist_key], era5_data
               )
           else:
               print(f"Warning: No historical data found for {model_name}")
               corrected_future[scenario][gcm_name] = gcm_data

Create Future Workflows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create workflows for future projections
   future_workflows = {}

   for scenario in SCENARIOS:
       scenario_workflows = assessment.create_workflows(
           corrected_future[scenario], 
           f"future_{scenario}"
       )
       future_workflows.update(scenario_workflows)

   print(f"Created {len(future_workflows)} future projection workflows")

Step 4: Climate Change Signal Analysis
--------------------------------------

Calculate Climate Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_climate_changes(historical_data, future_data):
       """Calculate climate change signals between periods."""
       
       changes = {}
       
       # Temperature changes (absolute)
       if 'tas' in future_data.data_vars and 'tas' in historical_data.data_vars:
           temp_change = future_data['tas'].mean('time') - historical_data['tas'].mean('time')
           changes['temperature_change'] = temp_change
       
       # Precipitation changes (relative)
       if 'pr' in future_data.data_vars and 'pr' in historical_data.data_vars:
           precip_change = ((future_data['pr'].mean('time') / historical_data['pr'].mean('time')) - 1) * 100
           changes['precipitation_change'] = precip_change
       
       return changes

   # Calculate climate change signals
   climate_signals = {}

   for scenario in SCENARIOS:
       climate_signals[scenario] = {}
       
       for gcm_name in corrected_future[scenario].keys():
           model_name = gcm_name.split('_')[0]
           hist_key = f"{model_name}_historical"
           
           if hist_key in gcm_historical:
               climate_signals[scenario][gcm_name] = calculate_climate_changes(
                   gcm_historical[hist_key],
                   corrected_future[scenario][gcm_name]
               )

Visualize Climate Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_climate_change_signals(climate_signals):
       """Plot climate change signals across models and scenarios."""
       
       fig, axes = plt.subplots(1, 2, figsize=(15, 6))
       
       # Prepare data for plotting
       temp_changes = []
       precip_changes = []
       scenarios = []
       models = []
       
       for scenario in SCENARIOS:
           for gcm_name, signals in climate_signals[scenario].items():
               model_name = gcm_name.split('_')[0]
               
               if 'temperature_change' in signals:
                   temp_change_mean = float(signals['temperature_change'].mean())
                   temp_changes.append(temp_change_mean)
                   
               if 'precipitation_change' in signals:
                   precip_change_mean = float(signals['precipitation_change'].mean())
                   precip_changes.append(precip_change_mean)
                   
               scenarios.append(scenario.upper())
               models.append(model_name)
       
       # Temperature changes
       df_temp = pd.DataFrame({
           'Temperature Change (°C)': temp_changes,
           'Scenario': scenarios,
           'Model': models
       })
       
       sns.boxplot(data=df_temp, x='Scenario', y='Temperature Change (°C)', ax=axes[0])
       axes[0].set_title('Temperature Change by Scenario')
       axes[0].grid(True, alpha=0.3)
       
       # Precipitation changes
       df_precip = pd.DataFrame({
           'Precipitation Change (%)': precip_changes,
           'Scenario': scenarios,
           'Model': models
       })
       
       sns.boxplot(data=df_precip, x='Scenario', y='Precipitation Change (%)', ax=axes[1])
       axes[1].set_title('Precipitation Change by Scenario')
       axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
       axes[1].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return df_temp, df_precip

   # Plot climate change signals
   temp_changes_df, precip_changes_df = plot_climate_change_signals(climate_signals)

   print("Climate Change Summary:")
   print(f"Temperature changes: {temp_changes_df['Temperature Change (°C)'].min():.1f} to {temp_changes_df['Temperature Change (°C)'].max():.1f} °C")
   print(f"Precipitation changes: {precip_changes_df['Precipitation Change (%)'].min():.1f} to {precip_changes_df['Precipitation Change (%)'].max():.1f} %")

Step 5: Hydrological Impact Assessment
--------------------------------------

Simulate Discharge Response
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   This section demonstrates the analysis that would be performed after running the workflows.

.. code-block:: python

   def create_mock_discharge_results():
       """Create mock discharge results for demonstration."""
       
       # Generate realistic discharge time series
       dates_hist = pd.date_range('1990-01-01', '2019-12-31', freq='D')
       dates_fut = pd.date_range('2070-01-01', '2099-12-31', freq='D')
       
       results = {'historical': {}, 'future': {}}
       
       # Historical results (ERA5 baseline)
       base_discharge = 2.0 + 0.8 * np.sin(2 * np.pi * np.arange(len(dates_hist)) / 365.25)
       noise = np.random.normal(0, 0.3, len(dates_hist))
       
       for hydro_model in HYDRO_MODELS.keys():
           model_bias = {"HBV-96": 0.05, "GR4J": -0.03}[hydro_model]
           discharge = np.maximum(0, base_discharge + noise + model_bias)
           
           results['historical'][f'ERA5_baseline_{hydro_model}'] = pd.Series(
               discharge, index=dates_hist, name='discharge'
           )
       
       # Future results with climate change impacts
       for scenario in SCENARIOS:
           results['future'][scenario] = {}
           
           # Climate change factors
           temp_increase = {"ssp245": 2.5, "ssp585": 4.2}[scenario]
           precip_change = {"ssp245": 5, "ssp585": -10}[scenario]  # Percent change
           
           for model in CLIMATE_MODELS:
               for hydro_model in HYDRO_MODELS.keys():
                   # Apply climate change impacts to discharge
                   base_future = 2.0 + 0.8 * np.sin(2 * np.pi * np.arange(len(dates_fut)) / 365.25)
                   
                   # Temperature effect on ET (simplified)
                   et_increase = temp_increase * 0.1  # 10% increase per °C
                   
                   # Precipitation effect
                   precip_factor = 1 + precip_change / 100
                   
                   # Combined effect on discharge
                   climate_factor = precip_factor - et_increase / 2
                   modified_discharge = base_future * climate_factor
                   
                   # Add model-specific variations
                   model_variation = np.random.normal(1, 0.1, len(dates_fut))
                   hydro_bias = {"HBV-96": 0.05, "GR4J": -0.03}[hydro_model]
                   
                   final_discharge = np.maximum(0, modified_discharge * model_variation + hydro_bias)
                   
                   workflow_name = f"future_{scenario}_{model}_{hydro_model}"
                   results['future'][scenario][workflow_name] = pd.Series(
                       final_discharge, index=dates_fut, name='discharge'
                   )
       
       return results

   # Create mock results for analysis
   discharge_results = create_mock_discharge_results()

Calculate Hydrological Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_hydrological_changes(historical_results, future_results):
       """Calculate changes in hydrological indicators."""
       
       changes = {}
       
       for scenario in SCENARIOS:
           changes[scenario] = {}
           
           # Calculate ensemble statistics for historical baseline
           hist_ensemble = []
           for workflow_name, discharge in historical_results.items():
               if 'ERA5_baseline' in workflow_name:
                   hist_ensemble.append(discharge)
           
           hist_mean = pd.concat(hist_ensemble, axis=1).mean(axis=1)
           
           # Calculate changes for each future projection
           for workflow_name, future_discharge in future_results[scenario].items():
               
               # Annual mean change
               hist_annual_mean = hist_mean.groupby(hist_mean.index.year).mean().mean()
               future_annual_mean = future_discharge.groupby(future_discharge.index.year).mean().mean()
               annual_change = ((future_annual_mean / hist_annual_mean) - 1) * 100
               
               # Seasonal changes
               hist_seasonal = hist_mean.groupby(hist_mean.index.month).mean()
               future_seasonal = future_discharge.groupby(future_discharge.index.month).mean()
               seasonal_changes = ((future_seasonal / hist_seasonal) - 1) * 100
               
               # Extreme flow changes
               hist_q95 = hist_mean.quantile(0.95)
               future_q95 = future_discharge.quantile(0.95)
               high_flow_change = ((future_q95 / hist_q95) - 1) * 100
               
               hist_q05 = hist_mean.quantile(0.05)
               future_q05 = future_discharge.quantile(0.05)
               low_flow_change = ((future_q05 / hist_q05) - 1) * 100
               
               changes[scenario][workflow_name] = {
                   'annual_mean_change': annual_change,
                   'seasonal_changes': seasonal_changes,
                   'high_flow_change': high_flow_change,
                   'low_flow_change': low_flow_change
               }
       
       return changes

   # Calculate hydrological changes
   hydro_changes = calculate_hydrological_changes(
       discharge_results['historical'],
       discharge_results['future']
   )

Visualize Hydrological Impacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_hydrological_impacts(hydro_changes):
       """Plot hydrological impact assessment results."""
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
       # Prepare data for plotting
       annual_changes = []
       high_flow_changes = []
       low_flow_changes = []
       scenarios = []
       models = []
       hydro_models = []
       
       for scenario in SCENARIOS:
           for workflow_name, changes in hydro_changes[scenario].items():
               parts = workflow_name.split('_')
               climate_model = parts[2]  # Extract climate model name
               hydro_model = parts[3]    # Extract hydro model name
               
               annual_changes.append(changes['annual_mean_change'])
               high_flow_changes.append(changes['high_flow_change'])
               low_flow_changes.append(changes['low_flow_change'])
               scenarios.append(scenario.upper())
               models.append(climate_model)
               hydro_models.append(hydro_model)
       
       # Create DataFrame for analysis
       impact_df = pd.DataFrame({
           'Annual Change (%)': annual_changes,
           'High Flow Change (%)': high_flow_changes,
           'Low Flow Change (%)': low_flow_changes,
           'Scenario': scenarios,
           'Climate Model': models,
           'Hydro Model': hydro_models
       })
       
       # Annual mean changes
       sns.boxplot(data=impact_df, x='Scenario', y='Annual Change (%)', 
                   hue='Hydro Model', ax=axes[0,0])
       axes[0,0].set_title('Annual Mean Discharge Changes')
       axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
       axes[0,0].grid(True, alpha=0.3)
       
       # High flow changes
       sns.boxplot(data=impact_df, x='Scenario', y='High Flow Change (%)', 
                   hue='Hydro Model', ax=axes[0,1])
       axes[0,1].set_title('High Flow (Q95) Changes')
       axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
       axes[0,1].grid(True, alpha=0.3)
       
       # Low flow changes
       sns.boxplot(data=impact_df, x='Scenario', y='Low Flow Change (%)', 
                   hue='Hydro Model', ax=axes[1,0])
       axes[1,0].set_title('Low Flow (Q05) Changes')
       axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
       axes[1,0].grid(True, alpha=0.3)
       
       # Model agreement analysis
       model_agreement = impact_df.groupby(['Scenario', 'Climate Model'])['Annual Change (%)'].std()
       model_agreement.unstack().plot(kind='bar', ax=axes[1,1])
       axes[1,1].set_title('Model Agreement (Std Dev of Annual Changes)')
       axes[1,1].set_ylabel('Standard Deviation (%)')
       axes[1,1].tick_params(axis='x', rotation=45)
       axes[1,1].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return impact_df

   # Plot hydrological impacts
   impact_summary = plot_hydrological_impacts(hydro_changes)

   print("Hydrological Impact Summary:")
   print(impact_summary.groupby('Scenario')[['Annual Change (%)', 'High Flow Change (%)', 'Low Flow Change (%)']].describe())

Step 6: Uncertainty Analysis
----------------------------

Quantify Total Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def decompose_uncertainty(impact_df):
       """Decompose uncertainty into different sources."""
       
       uncertainty_components = {}
       
       for scenario in SCENARIOS:
           scenario_data = impact_df[impact_df['Scenario'] == scenario.upper()]
           
           # Total variance
           total_var = scenario_data['Annual Change (%)'].var()
           
           # Climate model uncertainty
           climate_var = scenario_data.groupby('Climate Model')['Annual Change (%)'].mean().var()
           
           # Hydrological model uncertainty
           hydro_var = scenario_data.groupby('Hydro Model')['Annual Change (%)'].mean().var()
           
           # Interaction/residual uncertainty
           residual_var = total_var - climate_var - hydro_var
           
           uncertainty_components[scenario] = {
               'total': total_var,
               'climate_model': climate_var,
               'hydro_model': hydro_var,
               'residual': max(0, residual_var)  # Ensure non-negative
           }
       
       return uncertainty_components

   # Decompose uncertainty sources
   uncertainty_decomp = decompose_uncertainty(impact_summary)

   # Visualize uncertainty decomposition
   fig, ax = plt.subplots(1, 1, figsize=(10, 6))

   scenarios = list(uncertainty_decomp.keys())
   climate_uncertainty = [uncertainty_decomp[s]['climate_model'] for s in scenarios]
   hydro_uncertainty = [uncertainty_decomp[s]['hydro_model'] for s in scenarios]
   residual_uncertainty = [uncertainty_decomp[s]['residual'] for s in scenarios]

   x = np.arange(len(scenarios))
   width = 0.6

   p1 = ax.bar(x, climate_uncertainty, width, label='Climate Model', color='skyblue')
   p2 = ax.bar(x, hydro_uncertainty, width, bottom=climate_uncertainty, 
              label='Hydrological Model', color='lightcoral')
   p3 = ax.bar(x, residual_uncertainty, width, 
              bottom=np.array(climate_uncertainty) + np.array(hydro_uncertainty),
              label='Residual', color='lightgray')

   ax.set_title('Uncertainty Decomposition in Discharge Projections')
   ax.set_ylabel('Variance in Annual Change (%²)')
   ax.set_xlabel('Emission Scenario')
   ax.set_xticks(x)
   ax.set_xticklabels([s.upper() for s in scenarios])
   ax.legend()
   ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   print("Uncertainty Decomposition:")
   for scenario, components in uncertainty_decomp.items():
       print(f"\n{scenario.upper()} scenario:")
       total = components['total']
       for source, var in components.items():
           if source != 'total':
               percentage = (var / total) * 100 if total > 0 else 0
               print(f"  {source.replace('_', ' ').title()}: {percentage:.1f}%")

Step 7: Synthesis and Reporting
-------------------------------

Generate Summary Report
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_impact_report(climate_signals, hydro_changes, uncertainty_decomp):
       """Generate a comprehensive climate change impact report."""
       
       report = {
           'executive_summary': {},
           'climate_changes': {},
           'hydrological_impacts': {},
           'uncertainty_assessment': {},
           'recommendations': []
       }
       
       # Climate change summary
       all_temp_changes = []
       all_precip_changes = []
       
       for scenario in SCENARIOS:
           scenario_temp = []
           scenario_precip = []
           
           for gcm_name, signals in climate_signals[scenario].items():
               if 'temperature_change' in signals:
                   scenario_temp.append(float(signals['temperature_change'].mean()))
               if 'precipitation_change' in signals:
                   scenario_precip.append(float(signals['precipitation_change'].mean()))
           
           report['climate_changes'][scenario] = {
               'temperature_change_range': [min(scenario_temp), max(scenario_temp)],
               'precipitation_change_range': [min(scenario_precip), max(scenario_precip)],
               'temperature_change_mean': np.mean(scenario_temp),
               'precipitation_change_mean': np.mean(scenario_precip)
           }
       
       # Hydrological impact summary
       for scenario in SCENARIOS:
           annual_changes = [changes['annual_mean_change'] 
                           for changes in hydro_changes[scenario].values()]
           high_flow_changes = [changes['high_flow_change'] 
                              for changes in hydro_changes[scenario].values()]
           low_flow_changes = [changes['low_flow_change'] 
                             for changes in hydro_changes[scenario].values()]
           
           report['hydrological_impacts'][scenario] = {
               'annual_discharge_change_range': [min(annual_changes), max(annual_changes)],
               'annual_discharge_change_mean': np.mean(annual_changes),
               'high_flow_change_range': [min(high_flow_changes), max(high_flow_changes)],
               'low_flow_change_range': [min(low_flow_changes), max(low_flow_changes)]
           }
       
       # Uncertainty assessment
       report['uncertainty_assessment'] = uncertainty_decomp
       
       # Generate recommendations
       report['recommendations'] = [
           "Consider ensemble approach using multiple climate and hydrological models",
           "Account for uncertainty ranges in adaptation planning",
           "Monitor key hydrological indicators for early detection of changes",
           "Develop flexible adaptation strategies that perform well under multiple scenarios",
           "Update assessments regularly as new climate projections become available"
       ]
       
       return report

   # Generate comprehensive report
   impact_report = generate_impact_report(climate_signals, hydro_changes, uncertainty_decomp)

   print("CLIMATE CHANGE IMPACT ASSESSMENT REPORT")
   print("=" * 50)
   
   for scenario in SCENARIOS:
       scenario_upper = scenario.upper()
       climate = impact_report['climate_changes'][scenario]
       hydro = impact_report['hydrological_impacts'][scenario]
       
       print(f"\n{scenario_upper} SCENARIO:")
       print(f"  Climate Changes:")
       print(f"    Temperature: +{climate['temperature_change_mean']:.1f}°C "
             f"(range: +{climate['temperature_change_range'][0]:.1f} to +{climate['temperature_change_range'][1]:.1f}°C)")
       print(f"    Precipitation: {climate['precipitation_change_mean']:+.1f}% "
             f"(range: {climate['precipitation_change_range'][0]:+.1f} to {climate['precipitation_change_range'][1]:+.1f}%)")
       
       print(f"  Hydrological Impacts:")
       print(f"    Annual discharge: {hydro['annual_discharge_change_mean']:+.1f}% "
             f"(range: {hydro['annual_discharge_change_range'][0]:+.1f} to {hydro['annual_discharge_change_range'][1]:+.1f}%)")
       print(f"    High flows: {np.mean([min(hydro['high_flow_change_range']), max(hydro['high_flow_change_range'])]):+.1f}% change")
       print(f"    Low flows: {np.mean([min(hydro['low_flow_change_range']), max(hydro['low_flow_change_range'])]):+.1f}% change")

Complete Climate Change Assessment Script
-----------------------------------------

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete climate change impact assessment using MarrmotFlow
   """
   
   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   from marrmotflow import MARRMOTWorkflow
   
   def main():
       # Configuration
       CLIMATE_MODELS = ["CanESM5", "CNRM-CM6", "UKESM1"]
       SCENARIOS = ["ssp245", "ssp585"] 
       HYDRO_MODELS = {"HBV-96": 7, "GR4J": 37}
       
       # Load catchment data
       catchments = gpd.read_file("data/catchments.shp")
       
       print("Climate Change Impact Assessment")
       print("=" * 40)
       print(f"Catchments: {len(catchments)}")
       print(f"Climate models: {len(CLIMATE_MODELS)}")
       print(f"Emission scenarios: {len(SCENARIOS)}")
       print(f"Hydrological models: {len(HYDRO_MODELS)}")
       
       # Create workflows for different periods and scenarios
       all_workflows = {}
       
       # Historical baseline
       for hydro_name, hydro_model in HYDRO_MODELS.items():
           workflow_name = f"historical_ERA5_{hydro_name}"
           all_workflows[workflow_name] = MARRMOTWorkflow(
               name=workflow_name,
               cat=catchments,
               forcing_files="data/historical/era5_1990_2019.nc",
               forcing_vars={"precip": "total_precipitation", "temp": "2m_temperature"},
               forcing_units={"precip": "m/day", "temp": "kelvin"},
               model_number=hydro_model,
               pet_method="penman_monteith"
           )
       
       # Future projections
       for scenario in SCENARIOS:
           for climate_model in CLIMATE_MODELS:
               for hydro_name, hydro_model in HYDRO_MODELS.items():
                   workflow_name = f"future_{scenario}_{climate_model}_{hydro_name}"
                   forcing_file = f"data/gcm_future/{climate_model}_{scenario}_2070_2099.nc"
                   
                   if Path(forcing_file).exists():
                       all_workflows[workflow_name] = MARRMOTWorkflow(
                           name=workflow_name,
                           cat=catchments,
                           forcing_files=forcing_file,
                           forcing_vars={"precip": "pr", "temp": "tas"},
                           forcing_units={"precip": "kg m-2 s-1", "temp": "K"},
                           model_number=hydro_model,
                           pet_method="penman_monteith"
                       )
       
       print(f"\nTotal workflows created: {len(all_workflows)}")
       print("Ready for climate change impact assessment!")
       
       # Execute workflows and perform analysis here...
       
   if __name__ == "__main__":
       main()

Key Insights
------------

This comprehensive example demonstrates:

1. **Multi-scale uncertainty**: Climate models, emission scenarios, and hydrological models all contribute to uncertainty
2. **Bias correction importance**: Raw GCM data often needs correction before hydrological modeling
3. **Ensemble approaches**: Using multiple models provides more robust impact assessments
4. **Impact indicators**: Different aspects of hydrology (mean flows, extremes, seasonality) may respond differently
5. **Decision support**: Uncertainty quantification is crucial for adaptation planning

Next Steps
----------

* **Downscaling**: Apply statistical or dynamical downscaling to climate projections
* **Parameter uncertainty**: Include parameter uncertainty in hydrological models
* **Extreme event analysis**: Focus on changes in flood and drought characteristics
* **Sectoral impacts**: Extend analysis to water supply, agriculture, and ecosystem impacts
* **Adaptation assessment**: Evaluate effectiveness of different adaptation measures
