Multi-Model Comparison
======================

This example demonstrates how to compare multiple MARRMOT models using MarrmotFlow to assess model uncertainty and performance differences.

Overview
--------

In this example, we will:

1. Set up multiple model configurations
2. Run HBV-96, GR4J, and other models
3. Compare model outputs and performance
4. Analyze model agreement and uncertainty
5. Visualize comparative results

Prerequisites
-------------

* Multiple catchments for robust comparison
* Quality-controlled forcing data
* Optional: Observed discharge data for validation

.. code-block:: bash

   # Example data structure
   data/
   ├── catchments/
   │   ├── basin_001.shp
   │   ├── basin_002.shp
   │   └── basin_003.shp
   ├── climate/
   │   ├── era5_precip_2010_2020.nc
   │   └── era5_temp_2010_2020.nc
   └── observations/
       └── discharge_observations.csv

Step 1: Setup and Data Loading
------------------------------

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from marrmotflow import MARRMOTWorkflow
   from marrmotflow._default_dicts import default_model_dict
   
   # Set plotting style
   plt.style.use('seaborn-v0_8')
   sns.set_palette("husl")

Load Multiple Catchments
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load multiple catchment files
   import glob
   
   catchment_files = glob.glob("data/catchments/*.shp")
   catchments = []
   
   for file in catchment_files:
       gdf = gpd.read_file(file)
       catchments.append(gdf)
   
   # Combine into single GeoDataFrame
   all_catchments = gpd.GeoDataFrame(pd.concat(catchments, ignore_index=True))
   
   print(f"Loaded {len(all_catchments)} catchments")
   print(f"Catchment IDs: {all_catchments['id'].tolist() if 'id' in all_catchments.columns else 'No ID column'}")

Load Climate Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load climate forcing data
   precip_data = xr.open_dataset("data/climate/era5_precip_2010_2020.nc")
   temp_data = xr.open_dataset("data/climate/era5_temp_2010_2020.nc")
   
   # Combine datasets
   climate_data = xr.merge([precip_data, temp_data])
   
   print("Climate data loaded:")
   print(f"Variables: {list(climate_data.data_vars)}")
   print(f"Time range: {climate_data.time.min().item()} to {climate_data.time.max().item()}")

Step 2: Define Model Comparison Setup
-------------------------------------

Model Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Select models for comparison
   models_to_compare = {
       "HBV-96": 7,
       "GR4J": 37,
       "Collie1": 1,
       "Wetland": 2
   }
   
   print("Models selected for comparison:")
   for name, number in models_to_compare.items():
       model_info = default_model_dict.get(number, {"name": "Unknown", "description": "N/A"})
       print(f"  {name} (Model {number}): {model_info['description']}")

Common Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Common workflow configuration
   common_config = {
       "cat": all_catchments,
       "forcing_files": [
           "data/climate/era5_precip_2010_2020.nc",
           "data/climate/era5_temp_2010_2020.nc"
       ],
       "forcing_vars": {
           "precip": "total_precipitation",
           "temp": "2m_temperature"
       },
       "forcing_units": {
           "precip": "m/day",    # ERA5 units
           "temp": "kelvin"      # ERA5 units
       },
       "pet_method": "penman_monteith",
       "forcing_time_zone": "UTC",
       "model_time_zone": "America/Vancouver"
   }

Step 3: Create Multiple Workflows
---------------------------------

.. code-block:: python

   # Create workflow for each model
   workflows = {}
   
   for model_name, model_number in models_to_compare.items():
       print(f"Creating workflow for {model_name}...")
       
       workflows[model_name] = MARRMOTWorkflow(
           name=f"Comparison_{model_name}",
           model_number=model_number,
           **common_config
       )
   
   print(f"Created {len(workflows)} workflows for model comparison")

Step 4: Expected Results Analysis
---------------------------------

.. note::
   This section shows how to analyze results once workflows are executed.
   The actual execution method depends on the current MarrmotFlow implementation.

Result Structure
~~~~~~~~~~~~~~~

.. code-block:: python

   # Expected structure for comparison results
   def create_mock_results():
       """Create mock results for demonstration purposes."""
       
       # Generate sample time series
       dates = pd.date_range('2010-01-01', '2020-12-31', freq='D')
       n_days = len(dates)
       
       results = {}
       for model_name, model_number in models_to_compare.items():
           # Generate realistic discharge patterns
           base_discharge = 2.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
           noise = np.random.normal(0, 0.3, n_days)
           model_bias = {"HBV-96": 0.1, "GR4J": -0.05, "Collie1": 0.0, "Wetland": 0.15}.get(model_name, 0)
           
           discharge = np.maximum(0, base_discharge + noise + model_bias)
           
           results[model_name] = {
               'discharge': xr.DataArray(
                   discharge,
                   dims=['time'],
                   coords={'time': dates},
                   attrs={'units': 'mm/day', 'model': model_name}
               )
           }
       
       return results
   
   # For demonstration purposes
   results = create_mock_results()

Step 5: Model Comparison Analysis
---------------------------------

Time Series Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_discharge_comparison(results, start_date='2015-01-01', end_date='2016-12-31'):
       """Compare discharge time series from multiple models."""
       
       fig, axes = plt.subplots(2, 1, figsize=(15, 10))
       
       # Time series plot
       for model_name, model_results in results.items():
           discharge = model_results['discharge'].sel(time=slice(start_date, end_date))
           discharge.plot(ax=axes[0], label=model_name, alpha=0.8)
       
       axes[0].set_title('Model Discharge Comparison (2015-2016)')
       axes[0].set_ylabel('Discharge (mm/day)')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
       
       # Box plot for seasonal comparison
       seasonal_data = []
       model_names = []
       seasons = []
       
       for model_name, model_results in results.items():
           discharge = model_results['discharge'].sel(time=slice(start_date, end_date))
           df = discharge.to_dataframe(name='discharge')
           df['season'] = df.index.month.map({
               12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring', 4: 'Spring', 5: 'Spring',
               6: 'Summer', 7: 'Summer', 8: 'Summer',
               9: 'Fall', 10: 'Fall', 11: 'Fall'
           })
           
           for season in ['Winter', 'Spring', 'Summer', 'Fall']:
               season_data = df[df['season'] == season]['discharge']
               seasonal_data.extend(season_data.values)
               model_names.extend([model_name] * len(season_data))
               seasons.extend([season] * len(season_data))
       
       comparison_df = pd.DataFrame({
           'discharge': seasonal_data,
           'model': model_names,
           'season': seasons
       })
       
       sns.boxplot(data=comparison_df, x='season', y='discharge', hue='model', ax=axes[1])
       axes[1].set_title('Seasonal Discharge Distribution by Model')
       axes[1].set_ylabel('Discharge (mm/day)')
       
       plt.tight_layout()
       plt.show()
   
   # Plot comparison
   plot_discharge_comparison(results)

Statistical Comparison
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_model_statistics(results):
       """Calculate comparative statistics for all models."""
       
       stats_dict = {}
       
       for model_name, model_results in results.items():
           discharge = model_results['discharge']
           
           stats = {
               'mean': discharge.mean().item(),
               'std': discharge.std().item(),
               'min': discharge.min().item(),
               'max': discharge.max().item(),
               'q25': discharge.quantile(0.25).item(),
               'q50': discharge.quantile(0.50).item(),
               'q75': discharge.quantile(0.75).item(),
               'cv': (discharge.std() / discharge.mean()).item()
           }
           
           stats_dict[model_name] = stats
       
       # Convert to DataFrame for easy comparison
       stats_df = pd.DataFrame(stats_dict).T
       return stats_df
   
   # Calculate and display statistics
   model_stats = calculate_model_statistics(results)
   print("Model Comparison Statistics:")
   print(model_stats.round(3))
   
   # Visualize statistics
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Mean discharge
   model_stats['mean'].plot(kind='bar', ax=axes[0,0], color='skyblue')
   axes[0,0].set_title('Mean Annual Discharge')
   axes[0,0].set_ylabel('Discharge (mm/day)')
   axes[0,0].tick_params(axis='x', rotation=45)
   
   # Standard deviation
   model_stats['std'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
   axes[0,1].set_title('Discharge Variability (Std Dev)')
   axes[0,1].set_ylabel('Standard Deviation (mm/day)')
   axes[0,1].tick_params(axis='x', rotation=45)
   
   # Coefficient of variation
   model_stats['cv'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
   axes[1,0].set_title('Coefficient of Variation')
   axes[1,0].set_ylabel('CV (-)')
   axes[1,0].tick_params(axis='x', rotation=45)
   
   # Range (max - min)
   discharge_range = model_stats['max'] - model_stats['min']
   discharge_range.plot(kind='bar', ax=axes[1,1], color='gold')
   axes[1,1].set_title('Discharge Range')
   axes[1,1].set_ylabel('Range (mm/day)')
   axes[1,1].tick_params(axis='x', rotation=45)
   
   plt.tight_layout()
   plt.show()

Model Agreement Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_model_agreement(results):
       """Analyze agreement between different models."""
       
       # Create combined dataset
       discharge_data = {}
       for model_name, model_results in results.items():
           discharge_data[model_name] = model_results['discharge'].values
       
       combined_df = pd.DataFrame(discharge_data)
       
       # Calculate correlation matrix
       correlation_matrix = combined_df.corr()
       
       # Plot correlation heatmap
       fig, axes = plt.subplots(1, 2, figsize=(15, 6))
       
       # Correlation heatmap
       sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0], square=True, cbar_kws={'label': 'Correlation'})
       axes[0].set_title('Model Discharge Correlations')
       
       # Model agreement (ensemble statistics)
       ensemble_mean = combined_df.mean(axis=1)
       ensemble_std = combined_df.std(axis=1)
       
       # Plot ensemble statistics
       time_index = pd.date_range('2010-01-01', periods=len(ensemble_mean), freq='D')
       
       axes[1].fill_between(time_index, 
                           ensemble_mean - ensemble_std,
                           ensemble_mean + ensemble_std,
                           alpha=0.3, label='±1 std')
       axes[1].plot(time_index, ensemble_mean, 'k-', linewidth=2, label='Ensemble Mean')
       
       # Plot individual models (subset for clarity)
       subset_indices = slice(0, 365)  # First year only
       for model_name in ['HBV-96', 'GR4J']:
           axes[1].plot(time_index[subset_indices], 
                       combined_df[model_name].iloc[subset_indices],
                       alpha=0.7, label=model_name)
       
       axes[1].set_title('Model Ensemble Analysis (First Year)')
       axes[1].set_ylabel('Discharge (mm/day)')
       axes[1].legend()
       axes[1].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return correlation_matrix, ensemble_mean, ensemble_std
   
   # Analyze model agreement
   corr_matrix, ens_mean, ens_std = analyze_model_agreement(results)
   
   print("Model Correlation Summary:")
   print(f"Average inter-model correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
   print(f"Highest correlation: {corr_matrix.values.max():.3f}")
   print(f"Lowest correlation: {corr_matrix.values[corr_matrix.values < 1].min():.3f}")

Step 6: Performance Assessment
-----------------------------

.. code-block:: python

   def load_observed_data():
       """Load observed discharge data for validation."""
       # Mock observed data for demonstration
       dates = pd.date_range('2010-01-01', '2020-12-31', freq='D')
       
       # Generate realistic observed discharge
       base_obs = 2.0 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
       noise = np.random.normal(0, 0.2, len(dates))
       observed = np.maximum(0, base_obs + noise)
       
       return pd.Series(observed, index=dates, name='observed_discharge')

   def calculate_performance_metrics(observed, simulated):
       """Calculate hydrological performance metrics."""
       
       # Ensure same length and no missing data
       common_idx = observed.index.intersection(simulated.index)
       obs = observed.loc[common_idx].dropna()
       sim = simulated.loc[common_idx].dropna()
       
       # Nash-Sutcliffe Efficiency
       nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)
       
       # Root Mean Square Error
       rmse = np.sqrt(np.mean((obs - sim)**2))
       
       # Percent Bias
       pbias = 100 * np.sum(sim - obs) / np.sum(obs)
       
       # Correlation coefficient
       correlation = np.corrcoef(obs, sim)[0, 1]
       
       # Kling-Gupta Efficiency
       r = correlation
       alpha = sim.std() / obs.std()
       beta = sim.mean() / obs.mean()
       kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
       
       return {
           'NSE': nse,
           'RMSE': rmse,
           'PBIAS': pbias,
           'R': correlation,
           'KGE': kge
       }

   # Load observed data and calculate performance
   observed = load_observed_data()
   
   performance_results = {}
   for model_name, model_results in results.items():
       simulated = pd.Series(
           model_results['discharge'].values,
           index=model_results['discharge'].time.values,
           name=f'{model_name}_discharge'
       )
       
       performance_results[model_name] = calculate_performance_metrics(observed, simulated)
   
   # Create performance comparison table
   performance_df = pd.DataFrame(performance_results).T
   print("Model Performance Comparison:")
   print(performance_df.round(3))
   
   # Visualize performance metrics
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   axes = axes.flatten()
   
   metrics = ['NSE', 'RMSE', 'PBIAS', 'R', 'KGE']
   colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
   
   for i, metric in enumerate(metrics):
       performance_df[metric].plot(kind='bar', ax=axes[i], color=colors[i])
       axes[i].set_title(f'{metric}')
       axes[i].tick_params(axis='x', rotation=45)
       axes[i].grid(True, alpha=0.3)
       
       # Add reference lines for some metrics
       if metric == 'NSE':
           axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Good (0.5)')
           axes[i].legend()
       elif metric == 'PBIAS':
           axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Perfect (0)')
           axes[i].legend()
   
   # Remove extra subplot
   axes[-1].remove()
   
   plt.tight_layout()
   plt.show()

Step 7: Uncertainty Quantification
----------------------------------

.. code-block:: python

   def quantify_model_uncertainty(results):
       """Quantify uncertainty across models."""
       
       # Combine all model results
       all_discharges = []
       for model_name, model_results in results.items():
           all_discharges.append(model_results['discharge'].values)
       
       all_discharges = np.array(all_discharges)
       
       # Calculate ensemble statistics
       ensemble_mean = np.mean(all_discharges, axis=0)
       ensemble_std = np.std(all_discharges, axis=0)
       ensemble_min = np.min(all_discharges, axis=0)
       ensemble_max = np.max(all_discharges, axis=0)
       
       # Calculate uncertainty metrics
       relative_uncertainty = ensemble_std / ensemble_mean * 100
       spread = ensemble_max - ensemble_min
       
       # Time series for plotting
       time_index = results[list(results.keys())[0]]['discharge'].time.values
       
       # Plot uncertainty analysis
       fig, axes = plt.subplots(3, 1, figsize=(15, 12))
       
       # Ensemble with uncertainty bounds
       axes[0].fill_between(time_index[0:365], 
                           (ensemble_mean - ensemble_std)[0:365],
                           (ensemble_mean + ensemble_std)[0:365],
                           alpha=0.3, color='gray', label='±1 std')
       axes[0].fill_between(time_index[0:365],
                           ensemble_min[0:365],
                           ensemble_max[0:365],
                           alpha=0.2, color='red', label='Min-Max range')
       axes[0].plot(time_index[0:365], ensemble_mean[0:365], 'k-', linewidth=2, label='Ensemble Mean')
       axes[0].set_title('Model Uncertainty (First Year)')
       axes[0].set_ylabel('Discharge (mm/day)')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
       
       # Relative uncertainty
       axes[1].plot(time_index[0:365], relative_uncertainty[0:365])
       axes[1].set_title('Relative Uncertainty (%)')
       axes[1].set_ylabel('Uncertainty (%)')
       axes[1].grid(True, alpha=0.3)
       
       # Monthly uncertainty
       monthly_uncertainty = []
       month_names = []
       for month in range(1, 13):
           month_mask = pd.to_datetime(time_index).month == month
           monthly_uncertainty.append(relative_uncertainty[month_mask].mean())
           month_names.append(pd.to_datetime(f'2020-{month:02d}-01').strftime('%b'))
       
       axes[2].bar(month_names, monthly_uncertainty, color='steelblue', alpha=0.7)
       axes[2].set_title('Average Monthly Model Uncertainty')
       axes[2].set_ylabel('Relative Uncertainty (%)')
       axes[2].tick_params(axis='x', rotation=45)
       axes[2].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       # Summary statistics
       print("Uncertainty Summary:")
       print(f"Mean relative uncertainty: {np.mean(relative_uncertainty):.1f}%")
       print(f"Maximum relative uncertainty: {np.max(relative_uncertainty):.1f}%")
       print(f"Average ensemble spread: {np.mean(spread):.2f} mm/day")
       
       return {
           'ensemble_mean': ensemble_mean,
           'ensemble_std': ensemble_std,
           'relative_uncertainty': relative_uncertainty,
           'spread': spread
       }
   
   # Quantify uncertainty
   uncertainty_results = quantify_model_uncertainty(results)

Step 8: Model Ranking and Selection
-----------------------------------

.. code-block:: python

   def rank_models(performance_df):
       """Rank models based on multiple performance criteria."""
       
       # Define weights for different metrics (adjust based on study objectives)
       weights = {
           'NSE': 0.3,
           'KGE': 0.3,
           'R': 0.2,
           'RMSE': -0.1,  # Negative because lower is better
           'PBIAS': -0.1  # Negative because closer to 0 is better (absolute value)
       }
       
       # Normalize metrics to 0-1 scale
       normalized_metrics = performance_df.copy()
       
       for metric in ['NSE', 'KGE', 'R']:
           # Higher is better - normalize to 0-1
           normalized_metrics[metric] = (performance_df[metric] - performance_df[metric].min()) / (performance_df[metric].max() - performance_df[metric].min())
       
       for metric in ['RMSE']:
           # Lower is better - invert and normalize
           normalized_metrics[metric] = 1 - (performance_df[metric] - performance_df[metric].min()) / (performance_df[metric].max() - performance_df[metric].min())
       
       for metric in ['PBIAS']:
           # Closer to 0 is better
           normalized_metrics[metric] = 1 - np.abs(performance_df[metric]) / np.abs(performance_df[metric]).max()
       
       # Calculate weighted score
       weighted_scores = {}
       for model in performance_df.index:
           score = sum(weights[metric] * normalized_metrics.loc[model, metric] for metric in weights.keys())
           weighted_scores[model] = score
       
       # Rank models
       ranked_models = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
       
       print("Model Ranking (based on weighted performance):")
       for i, (model, score) in enumerate(ranked_models, 1):
           print(f"{i}. {model}: {score:.3f}")
       
       return ranked_models, weighted_scores
   
   # Rank models
   model_ranking, scores = rank_models(performance_df)
   
   # Visualize ranking
   models = [item[0] for item in model_ranking]
   scores_list = [item[1] for item in model_ranking]
   
   plt.figure(figsize=(10, 6))
   bars = plt.bar(models, scores_list, color=['gold', 'silver', '#CD7F32', 'lightgray'])
   plt.title('Model Performance Ranking')
   plt.ylabel('Weighted Performance Score')
   plt.xticks(rotation=45)
   plt.grid(True, alpha=0.3)
   
   # Add score labels on bars
   for bar, score in zip(bars, scores_list):
       plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
   
   plt.tight_layout()
   plt.show()

Complete Multi-Model Workflow Script
------------------------------------

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete multi-model comparison workflow for MarrmotFlow
   """
   
   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from marrmotflow import MARRMOTWorkflow
   
   def main():
       # Configuration
       models_to_compare = {"HBV-96": 7, "GR4J": 37, "Collie1": 1}
       
       # Load data
       print("Loading data...")
       catchments = gpd.read_file("data/catchments.shp")
       
       # Create workflows
       workflows = {}
       for model_name, model_number in models_to_compare.items():
           print(f"Creating workflow for {model_name}...")
           workflows[model_name] = MARRMOTWorkflow(
               name=f"Comparison_{model_name}",
               cat=catchments,
               forcing_files="data/climate_data.nc",
               forcing_vars={"precip": "precipitation", "temp": "temperature"},
               model_number=model_number,
               pet_method="penman_monteith"
           )
       
       print(f"Multi-model comparison setup complete!")
       print(f"Ready to compare {len(workflows)} models across {len(catchments)} catchments")
       
       # Execute workflows and analysis here...
       
   if __name__ == "__main__":
       main()

Next Steps
----------

After completing this multi-model comparison:

1. **Parameter sensitivity analysis**: Examine how parameter uncertainty affects model agreement
2. **Spatial analysis**: Compare model performance across different catchment types
3. **Climate change projections**: Use ensemble results for climate impact assessment
4. **Operational forecasting**: Implement ensemble-based forecasting systems

Key Takeaways
-------------

* **Model diversity is valuable**: Different models capture different aspects of hydrological processes
* **Ensemble approaches reduce uncertainty**: Combining multiple models often provides more robust results
* **Performance varies by metric**: Models may rank differently depending on evaluation criteria
* **Context matters**: Model performance can vary by season, climate, and catchment characteristics
