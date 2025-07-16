Regional Analysis
================

This example demonstrates how to perform regional hydrological analysis using MarrmotFlow across multiple catchments with varying characteristics.

Overview
--------

Regional analysis in hydrology involves:

1. **Multi-catchment modeling** across diverse landscapes
2. **Spatial pattern analysis** of hydrological responses
3. **Catchment classification** based on characteristics
4. **Regional calibration** and parameter transfer
5. **Scaling relationships** and regionalization

This example shows how to implement comprehensive regional analysis with MarrmotFlow.

Prerequisites
-------------

Regional dataset with multiple catchments:

.. code-block:: bash

   data/
   ├── catchments/
   │   ├── regional_catchments.shp     # Multiple catchments
   │   └── catchment_attributes.csv    # Physical characteristics
   ├── climate/
   │   ├── gridded_precip_2000_2020.nc
   │   └── gridded_temp_2000_2020.nc
   ├── observations/
   │   ├── streamflow_gauges.csv
   │   └── gauge_locations.shp
   └── dem/
       └── regional_dem.tif

Step 1: Setup and Data Loading
------------------------------

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path
   from scipy import stats
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   from marrmotflow import MARRMOTWorkflow
   
   # Set up analysis parameters
   ANALYSIS_PERIOD = "2000-2020"
   MODELS_TO_TEST = {"HBV-96": 7, "GR4J": 37}

Regional Data Management
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RegionalAnalysis:
       """Framework for regional hydrological analysis using MarrmotFlow."""
       
       def __init__(self, catchment_file, attribute_file=None):
           self.catchments = gpd.read_file(catchment_file)
           self.n_catchments = len(self.catchments)
           
           # Load catchment attributes if provided
           if attribute_file:
               self.attributes = pd.read_csv(attribute_file)
               self.catchments = self.catchments.merge(
                   self.attributes, on='catchment_id', how='left'
               )
           
           self.workflows = {}
           self.results = {}
           
       def load_regional_climate(self, precip_file, temp_file):
           """Load gridded climate data covering all catchments."""
           self.climate_data = xr.Dataset({
               'precipitation': xr.open_dataarray(precip_file),
               'temperature': xr.open_dataarray(temp_file)
           })
           
           # Verify spatial coverage
           self._check_spatial_coverage()
           
       def _check_spatial_coverage(self):
           """Check that climate data covers all catchments."""
           cat_bounds = self.catchments.total_bounds
           clim_bounds = [
               self.climate_data.lon.min().item(),
               self.climate_data.lat.min().item(),
               self.climate_data.lon.max().item(),
               self.climate_data.lat.max().item()
           ]
           
           if not (clim_bounds[0] <= cat_bounds[0] and 
                   clim_bounds[1] <= cat_bounds[1] and
                   clim_bounds[2] >= cat_bounds[2] and
                   clim_bounds[3] >= cat_bounds[3]):
               print("Warning: Climate data may not fully cover all catchments")
           
           print(f"Climate data coverage: {clim_bounds}")
           print(f"Catchment bounds: {cat_bounds}")

   # Initialize regional analysis
   regional = RegionalAnalysis(
       "data/catchments/regional_catchments.shp",
       "data/catchments/catchment_attributes.csv"
   )
   
   print(f"Loaded {regional.n_catchments} catchments for regional analysis")

Load and Explore Catchment Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Display catchment characteristics
   print("Catchment Characteristics Summary:")
   if hasattr(regional, 'attributes'):
       numeric_cols = regional.catchments.select_dtypes(include=[np.number]).columns
       print(regional.catchments[numeric_cols].describe())
       
       # Visualize catchment characteristics
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
       characteristics = ['area_km2', 'mean_elevation', 'mean_slope', 
                         'forest_fraction', 'urban_fraction', 'mean_annual_precip']
       
       for i, char in enumerate(characteristics):
           if char in regional.catchments.columns:
               row, col = i // 3, i % 3
               regional.catchments[char].hist(bins=20, ax=axes[row, col], alpha=0.7)
               axes[row, col].set_title(f'{char.replace("_", " ").title()}')
               axes[row, col].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

Step 2: Catchment Classification
-------------------------------

Classify Based on Physical Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def classify_catchments(catchments, classification_vars, n_clusters=4):
       """Classify catchments based on physical characteristics."""
       
       # Prepare data for clustering
       cluster_data = catchments[classification_vars].dropna()
       
       # Standardize variables
       scaler = StandardScaler()
       scaled_data = scaler.fit_transform(cluster_data)
       
       # Perform k-means clustering
       kmeans = KMeans(n_clusters=n_clusters, random_state=42)
       cluster_labels = kmeans.fit_predict(scaled_data)
       
       # Add cluster labels to catchments
       catchments_classified = catchments.copy()
       catchments_classified.loc[cluster_data.index, 'cluster'] = cluster_labels
       
       # Calculate cluster characteristics
       cluster_summary = []
       for i in range(n_clusters):
           cluster_catchments = catchments_classified[catchments_classified['cluster'] == i]
           summary = {
               'cluster': i,
               'n_catchments': len(cluster_catchments),
               **{var: cluster_catchments[var].mean() for var in classification_vars}
           }
           cluster_summary.append(summary)
       
       cluster_df = pd.DataFrame(cluster_summary)
       
       return catchments_classified, cluster_df, scaler, kmeans

   # Classify catchments
   classification_vars = ['area_km2', 'mean_elevation', 'mean_slope', 
                         'forest_fraction', 'mean_annual_precip']
   
   available_vars = [var for var in classification_vars if var in regional.catchments.columns]
   
   if len(available_vars) >= 3:  # Need at least 3 variables for meaningful classification
       catchments_classified, cluster_summary, scaler, kmeans = classify_catchments(
           regional.catchments, available_vars, n_clusters=4
       )
       
       print("Catchment Classification Results:")
       print(cluster_summary)
       
       # Visualize clusters
       fig, axes = plt.subplots(1, 2, figsize=(15, 6))
       
       # Plot clusters on map
       catchments_classified.plot(column='cluster', categorical=True, 
                                 legend=True, ax=axes[0], cmap='Set1')
       axes[0].set_title('Catchment Clusters (Spatial Distribution)')
       
       # Plot cluster characteristics
       cluster_summary_melted = cluster_summary.melt(
           id_vars=['cluster', 'n_catchments'], 
           value_vars=available_vars,
           var_name='characteristic', 
           value_name='value'
       )
       
       sns.boxplot(data=cluster_summary_melted, x='characteristic', y='value', 
                  hue='cluster', ax=axes[1])
       axes[1].set_title('Cluster Characteristics')
       axes[1].tick_params(axis='x', rotation=45)
       
       plt.tight_layout()
       plt.show()
   else:
       print("Insufficient catchment attributes for classification")
       catchments_classified = regional.catchments.copy()
       catchments_classified['cluster'] = 0  # Single cluster

Step 3: Multi-Catchment Workflow Creation
-----------------------------------------

Create Individual Catchment Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_regional_workflows(catchments, climate_data, models_to_test):
       """Create workflows for all catchments and models."""
       
       workflows = {}
       
       for idx, catchment in catchments.iterrows():
           catchment_id = catchment.get('catchment_id', f'catchment_{idx}')
           
           # Extract climate data for this catchment
           catchment_climate = extract_catchment_climate(catchment, climate_data)
           
           # Save catchment-specific climate data
           climate_file = f"temp_climate_{catchment_id}.nc"
           catchment_climate.to_netcdf(climate_file)
           
           # Create single-catchment GeoDataFrame
           single_catchment = gpd.GeoDataFrame([catchment], crs=catchments.crs)
           
           for model_name, model_number in models_to_test.items():
               workflow_name = f"{catchment_id}_{model_name}"
               
               try:
                   workflows[workflow_name] = MARRMOTWorkflow(
                       name=workflow_name,
                       cat=single_catchment,
                       forcing_files=climate_file,
                       forcing_vars={"precip": "precipitation", "temp": "temperature"},
                       forcing_units={"precip": "mm/day", "temp": "celsius"},
                       model_number=model_number,
                       pet_method="penman_monteith"
                   )
               except Exception as e:
                   print(f"Warning: Could not create workflow for {workflow_name}: {e}")
       
       return workflows

   def extract_catchment_climate(catchment, climate_data):
       """Extract climate data for a specific catchment."""
       
       # Get catchment centroid for point extraction
       centroid = catchment.geometry.centroid
       lon, lat = centroid.x, centroid.y
       
       # Find nearest grid points
       lon_idx = np.argmin(np.abs(climate_data.lon - lon))
       lat_idx = np.argmin(np.abs(climate_data.lat - lat))
       
       # Extract point data
       point_data = climate_data.isel(lon=lon_idx, lat=lat_idx)
       
       return point_data

   # Load regional climate data
   regional.load_regional_climate(
       "data/climate/gridded_precip_2000_2020.nc",
       "data/climate/gridded_temp_2000_2020.nc"
   )

   # Create workflows for all catchments
   regional.workflows = create_regional_workflows(
       catchments_classified, regional.climate_data, MODELS_TO_TEST
   )

   print(f"Created {len(regional.workflows)} workflows for regional analysis")
   print(f"Workflows per catchment: {len(MODELS_TO_TEST)}")

Step 4: Simulated Regional Results Analysis
-------------------------------------------

Generate Mock Regional Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_regional_results(catchments, models_to_test, analysis_period):
       """Generate mock regional results for demonstration."""
       
       start_date = f"{analysis_period.split('-')[0]}-01-01"
       end_date = f"{analysis_period.split('-')[1]}-12-31"
       dates = pd.date_range(start_date, end_date, freq='D')
       
       results = {}
       
       for idx, catchment in catchments.iterrows():
           catchment_id = catchment.get('catchment_id', f'catchment_{idx}')
           
           # Catchment-specific characteristics affecting discharge
           area = catchment.get('area_km2', 100)
           elevation = catchment.get('mean_elevation', 500)
           forest = catchment.get('forest_fraction', 0.5)
           
           # Base discharge influenced by catchment characteristics
           base_discharge = (
               2.0 * (area / 100)**0.3 *  # Size effect
               (elevation / 1000)**0.2 *   # Elevation effect
               (forest + 0.5)              # Vegetation effect
           )
           
           # Seasonal pattern
           seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
           
           for model_name, model_number in models_to_test.items():
               # Model-specific bias
               model_bias = {"HBV-96": 0.1, "GR4J": -0.05}[model_name]
               
               # Random variation
               noise = np.random.normal(0, 0.2, len(dates))
               
               # Combine effects
               discharge = np.maximum(0, base_discharge + seasonal + model_bias + noise)
               
               workflow_name = f"{catchment_id}_{model_name}"
               results[workflow_name] = {
                   'discharge': pd.Series(discharge, index=dates, name='discharge'),
                   'catchment_id': catchment_id,
                   'model': model_name,
                   'catchment_area': area,
                   'catchment_elevation': elevation
               }
       
       return results

   # Generate regional results
   regional.results = generate_regional_results(
       catchments_classified, MODELS_TO_TEST, ANALYSIS_PERIOD
   )

   print(f"Generated results for {len(regional.results)} catchment-model combinations")

Step 5: Regional Pattern Analysis
---------------------------------

Analyze Spatial Patterns
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_regional_patterns(results, catchments):
       """Analyze spatial patterns in hydrological variables."""
       
       # Extract catchment-level statistics
       catchment_stats = {}
       
       for workflow_name, result in results.items():
           catchment_id = result['catchment_id']
           model = result['model']
           discharge = result['discharge']
           
           if catchment_id not in catchment_stats:
               catchment_stats[catchment_id] = {}
           
           # Calculate statistics
           stats = {
               'mean_annual_discharge': discharge.groupby(discharge.index.year).sum().mean(),
               'cv_annual': discharge.groupby(discharge.index.year).sum().std() / 
                          discharge.groupby(discharge.index.year).sum().mean(),
               'mean_daily_discharge': discharge.mean(),
               'q95': discharge.quantile(0.95),
               'q05': discharge.quantile(0.05),
               'baseflow_index': estimate_baseflow_index(discharge)
           }
           
           catchment_stats[catchment_id][model] = stats
       
       return catchment_stats

   def estimate_baseflow_index(discharge):
       """Estimate baseflow index using a simple approach."""
       # Simple baseflow separation: 7-day minimum rolling window
       baseflow = discharge.rolling(window=7, min_periods=1).min()
       return baseflow.mean() / discharge.mean()

   def plot_regional_patterns(catchment_stats, catchments):
       """Plot regional patterns in hydrological responses."""
       
       # Prepare data for plotting
       plot_data = []
       
       for catchment_id, model_stats in catchment_stats.items():
           for model, stats in model_stats.items():
               plot_data.append({
                   'catchment_id': catchment_id,
                   'model': model,
                   **stats
               })
       
       plot_df = pd.DataFrame(plot_data)
       
       # Merge with catchment characteristics
       catchment_attrs = catchments.drop('geometry', axis=1)
       plot_df = plot_df.merge(catchment_attrs, on='catchment_id', how='left')
       
       # Create subplots for different relationships
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
       # Mean discharge vs. area
       for model in MODELS_TO_TEST.keys():
           model_data = plot_df[plot_df['model'] == model]
           axes[0,0].scatter(model_data['area_km2'], model_data['mean_annual_discharge'], 
                           label=model, alpha=0.7)
       axes[0,0].set_xlabel('Catchment Area (km²)')
       axes[0,0].set_ylabel('Mean Annual Discharge (mm)')
       axes[0,0].set_title('Discharge vs. Catchment Area')
       axes[0,0].legend()
       axes[0,0].grid(True, alpha=0.3)
       
       # Discharge vs. elevation
       for model in MODELS_TO_TEST.keys():
           model_data = plot_df[plot_df['model'] == model]
           axes[0,1].scatter(model_data['mean_elevation'], model_data['mean_annual_discharge'], 
                           label=model, alpha=0.7)
       axes[0,1].set_xlabel('Mean Elevation (m)')
       axes[0,1].set_ylabel('Mean Annual Discharge (mm)')
       axes[0,1].set_title('Discharge vs. Elevation')
       axes[0,1].legend()
       axes[0,1].grid(True, alpha=0.3)
       
       # Coefficient of variation vs. forest fraction
       if 'forest_fraction' in plot_df.columns:
           for model in MODELS_TO_TEST.keys():
               model_data = plot_df[plot_df['model'] == model]
               axes[0,2].scatter(model_data['forest_fraction'], model_data['cv_annual'], 
                               label=model, alpha=0.7)
           axes[0,2].set_xlabel('Forest Fraction')
           axes[0,2].set_ylabel('CV of Annual Discharge')
           axes[0,2].set_title('Variability vs. Forest Cover')
           axes[0,2].legend()
           axes[0,2].grid(True, alpha=0.3)
       
       # Baseflow index vs. characteristics
       for model in MODELS_TO_TEST.keys():
           model_data = plot_df[plot_df['model'] == model]
           axes[1,0].scatter(model_data['area_km2'], model_data['baseflow_index'], 
                           label=model, alpha=0.7)
       axes[1,0].set_xlabel('Catchment Area (km²)')
       axes[1,0].set_ylabel('Baseflow Index')
       axes[1,0].set_title('Baseflow vs. Area')
       axes[1,0].legend()
       axes[1,0].grid(True, alpha=0.3)
       
       # Model comparison
       model_comparison = plot_df.pivot_table(
           values='mean_annual_discharge', 
           index='catchment_id', 
           columns='model'
       )
       
       if len(MODELS_TO_TEST) == 2:
           model_names = list(MODELS_TO_TEST.keys())
           axes[1,1].scatter(model_comparison[model_names[0]], 
                           model_comparison[model_names[1]], alpha=0.7)
           axes[1,1].plot([0, model_comparison.values.max()], [0, model_comparison.values.max()], 
                         'r--', alpha=0.5)
           axes[1,1].set_xlabel(f'{model_names[0]} Discharge (mm)')
           axes[1,1].set_ylabel(f'{model_names[1]} Discharge (mm)')
           axes[1,1].set_title('Model Comparison')
           axes[1,1].grid(True, alpha=0.3)
       
       # Regional statistics summary
       regional_summary = plot_df.groupby('model').agg({
           'mean_annual_discharge': ['mean', 'std'],
           'cv_annual': ['mean', 'std'],
           'baseflow_index': ['mean', 'std']
       }).round(3)
       
       # Display summary as text
       axes[1,2].axis('off')
       summary_text = "Regional Summary:\n\n"
       for model in MODELS_TO_TEST.keys():
           summary_text += f"{model}:\n"
           summary_text += f"  Mean Discharge: {regional_summary.loc[model, ('mean_annual_discharge', 'mean')]:.1f} ± {regional_summary.loc[model, ('mean_annual_discharge', 'std')]:.1f} mm\n"
           summary_text += f"  CV Annual: {regional_summary.loc[model, ('cv_annual', 'mean')]:.3f} ± {regional_summary.loc[model, ('cv_annual', 'std')]:.3f}\n"
           summary_text += f"  Baseflow Index: {regional_summary.loc[model, ('baseflow_index', 'mean')]:.3f} ± {regional_summary.loc[model, ('baseflow_index', 'std')]:.3f}\n\n"
       
       axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
       
       plt.tight_layout()
       plt.show()
       
       return plot_df

   # Analyze regional patterns
   catchment_statistics = analyze_regional_patterns(regional.results, catchments_classified)
   regional_patterns = plot_regional_patterns(catchment_statistics, catchments_classified)

Step 6: Model Performance by Region
-----------------------------------

Assess Model Performance Across Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def assess_regional_model_performance(regional_patterns, catchments_classified):
       """Assess model performance by catchment clusters."""
       
       # Merge with cluster information
       cluster_info = catchments_classified[['catchment_id', 'cluster']].copy()
       performance_data = regional_patterns.merge(cluster_info, on='catchment_id', how='left')
       
       # Calculate performance metrics by cluster
       cluster_performance = {}
       
       for cluster_id in performance_data['cluster'].unique():
           if pd.isna(cluster_id):
               continue
               
           cluster_data = performance_data[performance_data['cluster'] == cluster_id]
           
           cluster_performance[cluster_id] = {
               'n_catchments': len(cluster_data) // len(MODELS_TO_TEST),
               'model_stats': {}
           }
           
           for model in MODELS_TO_TEST.keys():
               model_data = cluster_data[cluster_data['model'] == model]
               
               cluster_performance[cluster_id]['model_stats'][model] = {
                   'mean_discharge': model_data['mean_annual_discharge'].mean(),
                   'std_discharge': model_data['mean_annual_discharge'].std(),
                   'mean_cv': model_data['cv_annual'].mean(),
                   'mean_baseflow': model_data['baseflow_index'].mean()
               }
       
       return cluster_performance, performance_data

   def plot_cluster_performance(cluster_performance, performance_data):
       """Plot model performance by catchment clusters."""
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
       # Prepare data for box plots
       discharge_data = []
       cv_data = []
       baseflow_data = []
       
       for _, row in performance_data.iterrows():
           if not pd.isna(row['cluster']):
               discharge_data.append({
                   'cluster': int(row['cluster']),
                   'model': row['model'],
                   'mean_annual_discharge': row['mean_annual_discharge']
               })
               cv_data.append({
                   'cluster': int(row['cluster']),
                   'model': row['model'],
                   'cv_annual': row['cv_annual']
               })
               baseflow_data.append({
                   'cluster': int(row['cluster']),
                   'model': row['model'],
                   'baseflow_index': row['baseflow_index']
               })
       
       discharge_df = pd.DataFrame(discharge_data)
       cv_df = pd.DataFrame(cv_data)
       baseflow_df = pd.DataFrame(baseflow_data)
       
       # Box plots by cluster
       sns.boxplot(data=discharge_df, x='cluster', y='mean_annual_discharge', 
                  hue='model', ax=axes[0,0])
       axes[0,0].set_title('Annual Discharge by Cluster')
       axes[0,0].set_ylabel('Mean Annual Discharge (mm)')
       
       sns.boxplot(data=cv_df, x='cluster', y='cv_annual', 
                  hue='model', ax=axes[0,1])
       axes[0,1].set_title('Discharge Variability by Cluster')
       axes[0,1].set_ylabel('CV Annual Discharge')
       
       sns.boxplot(data=baseflow_df, x='cluster', y='baseflow_index', 
                  hue='model', ax=axes[1,0])
       axes[1,0].set_title('Baseflow Index by Cluster')
       axes[1,0].set_ylabel('Baseflow Index')
       
       # Model agreement by cluster
       model_agreement = []
       for cluster_id, cluster_data in cluster_performance.items():
           models = list(cluster_data['model_stats'].keys())
           if len(models) == 2:
               model1_discharge = cluster_data['model_stats'][models[0]]['mean_discharge']
               model2_discharge = cluster_data['model_stats'][models[1]]['mean_discharge']
               agreement = abs(model1_discharge - model2_discharge) / max(model1_discharge, model2_discharge)
               model_agreement.append({'cluster': cluster_id, 'disagreement': agreement})
       
       if model_agreement:
           agreement_df = pd.DataFrame(model_agreement)
           agreement_df['cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1])
           axes[1,1].set_title('Number of Catchments by Cluster')
           axes[1,1].set_ylabel('Number of Catchments')
           axes[1,1].tick_params(axis='x', rotation=0)
       
       plt.tight_layout()
       plt.show()
       
       return cluster_performance

   # Assess model performance by region
   cluster_perf, perf_data = assess_regional_model_performance(regional_patterns, catchments_classified)
   cluster_performance = plot_cluster_performance(cluster_perf, perf_data)

   print("Regional Model Performance Summary:")
   for cluster_id, cluster_data in cluster_perf.items():
       print(f"\nCluster {cluster_id} ({cluster_data['n_catchments']} catchments):")
       for model, stats in cluster_data['model_stats'].items():
           print(f"  {model}:")
           print(f"    Mean discharge: {stats['mean_discharge']:.1f} ± {stats['std_discharge']:.1f} mm")
           print(f"    Mean CV: {stats['mean_cv']:.3f}")
           print(f"    Mean baseflow index: {stats['mean_baseflow']:.3f}")

Step 7: Regionalization and Parameter Transfer
----------------------------------------------

Develop Regional Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def develop_regional_relationships(regional_patterns, catchments_classified):
       """Develop relationships for parameter regionalization."""
       
       # Focus on key hydrological signatures
       signatures = ['mean_annual_discharge', 'cv_annual', 'baseflow_index', 'q95']
       predictors = ['area_km2', 'mean_elevation', 'mean_slope', 'forest_fraction']
       
       # Available predictors in dataset
       available_predictors = [p for p in predictors if p in catchments_classified.columns]
       
       relationships = {}
       
       for model in MODELS_TO_TEST.keys():
           model_data = regional_patterns[regional_patterns['model'] == model].copy()
           
           # Merge with catchment characteristics
           model_data = model_data.merge(
               catchments_classified[['catchment_id'] + available_predictors],
               on='catchment_id', how='left'
           )
           
           relationships[model] = {}
           
           for signature in signatures:
               if signature in model_data.columns:
                   relationships[model][signature] = {}
                   
                   for predictor in available_predictors:
                       # Calculate correlation
                       valid_data = model_data[[signature, predictor]].dropna()
                       if len(valid_data) > 3:
                           correlation, p_value = stats.pearsonr(
                               valid_data[signature], valid_data[predictor]
                           )
                           
                           # Simple linear regression
                           slope, intercept, r_value, p_value, std_err = stats.linregress(
                               valid_data[predictor], valid_data[signature]
                           )
                           
                           relationships[model][signature][predictor] = {
                               'correlation': correlation,
                               'r_squared': r_value**2,
                               'slope': slope,
                               'intercept': intercept,
                               'p_value': p_value,
                               'n_samples': len(valid_data)
                           }
       
       return relationships

   def plot_regional_relationships(relationships, regional_patterns, catchments_classified):
       """Plot key regional relationships."""
       
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))
       
       # Select best relationships to plot
       plot_configs = [
           ('mean_annual_discharge', 'area_km2', 'Discharge vs Area'),
           ('mean_annual_discharge', 'mean_elevation', 'Discharge vs Elevation'),
           ('cv_annual', 'forest_fraction', 'CV vs Forest Fraction'),
           ('baseflow_index', 'area_km2', 'Baseflow vs Area'),
           ('q95', 'mean_elevation', 'High Flows vs Elevation'),
           ('cv_annual', 'mean_slope', 'CV vs Slope')
       ]
       
       for i, (signature, predictor, title) in enumerate(plot_configs):
           if i >= 6:  # Only 6 subplots
               break
               
           row, col = i // 3, i % 3
           
           for model in MODELS_TO_TEST.keys():
               model_data = regional_patterns[regional_patterns['model'] == model].copy()
               model_data = model_data.merge(
                   catchments_classified[['catchment_id', predictor]],
                   on='catchment_id', how='left'
               )
               
               if signature in model_data.columns and predictor in model_data.columns:
                   valid_data = model_data[[signature, predictor]].dropna()
                   
                   if len(valid_data) > 0:
                       axes[row, col].scatter(valid_data[predictor], valid_data[signature], 
                                            label=model, alpha=0.7)
                       
                       # Add regression line if relationship exists
                       if (model in relationships and 
                           signature in relationships[model] and
                           predictor in relationships[model][signature]):
                           
                           rel = relationships[model][signature][predictor]
                           if rel['p_value'] < 0.05:  # Significant relationship
                               x_range = np.linspace(valid_data[predictor].min(), 
                                                   valid_data[predictor].max(), 100)
                               y_pred = rel['slope'] * x_range + rel['intercept']
                               axes[row, col].plot(x_range, y_pred, '--', alpha=0.8)
           
           axes[row, col].set_xlabel(predictor.replace('_', ' ').title())
           axes[row, col].set_ylabel(signature.replace('_', ' ').title())
           axes[row, col].set_title(title)
           axes[row, col].legend()
           axes[row, col].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

   # Develop regional relationships
   regional_relationships = develop_regional_relationships(regional_patterns, catchments_classified)
   plot_regional_relationships(regional_relationships, regional_patterns, catchments_classified)

   # Print significant relationships
   print("Significant Regional Relationships (p < 0.05):")
   for model, signatures in regional_relationships.items():
       print(f"\n{model}:")
       for signature, predictors in signatures.items():
           for predictor, stats in predictors.items():
               if stats['p_value'] < 0.05 and stats['r_squared'] > 0.1:
                   print(f"  {signature} vs {predictor}: R² = {stats['r_squared']:.3f}, p = {stats['p_value']:.3f}")

Step 8: Regional Summary and Validation
---------------------------------------

Cross-Validation of Regional Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def cross_validate_regional_model(relationships, regional_patterns, catchments_classified, signature, predictor):
       """Perform leave-one-out cross-validation for regional relationships."""
       
       results = {}
       
       for model in MODELS_TO_TEST.keys():
           if (model in relationships and 
               signature in relationships[model] and
               predictor in relationships[model][signature]):
               
               model_data = regional_patterns[regional_patterns['model'] == model].copy()
               model_data = model_data.merge(
                   catchments_classified[['catchment_id', predictor]],
                   on='catchment_id', how='left'
               )
               
               valid_data = model_data[[signature, predictor, 'catchment_id']].dropna()
               
               if len(valid_data) > 5:  # Need sufficient data for CV
                   predictions = []
                   observations = []
                   
                   for i in range(len(valid_data)):
                       # Leave one out
                       train_data = valid_data.drop(valid_data.index[i])
                       test_data = valid_data.iloc[i]
                       
                       # Fit model on training data
                       slope, intercept, _, _, _ = stats.linregress(
                           train_data[predictor], train_data[signature]
                       )
                       
                       # Predict on test data
                       prediction = slope * test_data[predictor] + intercept
                       
                       predictions.append(prediction)
                       observations.append(test_data[signature])
                   
                   # Calculate validation statistics
                   predictions = np.array(predictions)
                   observations = np.array(observations)
                   
                   cv_r2 = stats.pearsonr(observations, predictions)[0]**2
                   cv_rmse = np.sqrt(np.mean((observations - predictions)**2))
                   cv_bias = np.mean(predictions - observations)
                   
                   results[model] = {
                       'cv_r2': cv_r2,
                       'cv_rmse': cv_rmse,
                       'cv_bias': cv_bias,
                       'n_samples': len(valid_data)
                   }
       
       return results

   # Validate key relationships
   validation_results = {}
   key_relationships = [
       ('mean_annual_discharge', 'area_km2'),
       ('baseflow_index', 'forest_fraction'),
       ('cv_annual', 'mean_elevation')
   ]

   for signature, predictor in key_relationships:
       if predictor in catchments_classified.columns:
           cv_results = cross_validate_regional_model(
               regional_relationships, regional_patterns, 
               catchments_classified, signature, predictor
           )
           if cv_results:
               validation_results[f"{signature}_vs_{predictor}"] = cv_results

   print("Cross-Validation Results:")
   for relationship, models in validation_results.items():
       print(f"\n{relationship}:")
       for model, stats in models.items():
           print(f"  {model}: R² = {stats['cv_r2']:.3f}, RMSE = {stats['cv_rmse']:.3f}, Bias = {stats['cv_bias']:.3f}")

Generate Regional Summary Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_regional_report(regional_patterns, cluster_perf, regional_relationships, validation_results):
       """Generate comprehensive regional analysis report."""
       
       report = {
           'summary_statistics': {},
           'regional_patterns': {},
           'model_performance': {},
           'regionalization': {},
           'recommendations': []
       }
       
       # Summary statistics
       for model in MODELS_TO_TEST.keys():
           model_data = regional_patterns[regional_patterns['model'] == model]
           report['summary_statistics'][model] = {
               'n_catchments': len(model_data),
               'mean_discharge_range': [model_data['mean_annual_discharge'].min(), 
                                       model_data['mean_annual_discharge'].max()],
               'mean_discharge_mean': model_data['mean_annual_discharge'].mean(),
               'cv_range': [model_data['cv_annual'].min(), model_data['cv_annual'].max()],
               'baseflow_range': [model_data['baseflow_index'].min(), model_data['baseflow_index'].max()]
           }
       
       # Regional patterns
       report['regional_patterns'] = {
           'n_clusters': len(cluster_perf),
           'cluster_characteristics': cluster_perf
       }
       
       # Regionalization success
       successful_relationships = 0
       total_relationships = 0
       
       for model, signatures in regional_relationships.items():
           for signature, predictors in signatures.items():
               for predictor, stats in predictors.items():
                   total_relationships += 1
                   if stats['p_value'] < 0.05 and stats['r_squared'] > 0.1:
                       successful_relationships += 1
       
       report['regionalization'] = {
           'successful_relationships': successful_relationships,
           'total_relationships': total_relationships,
           'success_rate': successful_relationships / total_relationships if total_relationships > 0 else 0,
           'validation_results': validation_results
       }
       
       # Recommendations
       report['recommendations'] = [
           f"Analyzed {len(regional_patterns)//len(MODELS_TO_TEST)} catchments across {len(cluster_perf)} clusters",
           f"Found {successful_relationships} significant relationships out of {total_relationships} tested",
           "Consider cluster-specific model selection for improved performance",
           "Develop parameter transfer functions for ungauged catchments",
           "Validate regional relationships with additional data when available"
       ]
       
       return report

   # Generate final report
   regional_report = generate_regional_report(
       regional_patterns, cluster_perf, regional_relationships, validation_results
   )

   print("REGIONAL ANALYSIS REPORT")
   print("=" * 40)
   print(f"Analysis Period: {ANALYSIS_PERIOD}")
   print(f"Models Tested: {list(MODELS_TO_TEST.keys())}")
   print(f"Catchments Analyzed: {regional_report['summary_statistics'][list(MODELS_TO_TEST.keys())[0]]['n_catchments']}")
   print(f"Clusters Identified: {regional_report['regional_patterns']['n_clusters']}")
   print(f"Regionalization Success Rate: {regional_report['regionalization']['success_rate']:.1%}")

   for recommendation in regional_report['recommendations']:
       print(f"• {recommendation}")

Complete Regional Analysis Script
---------------------------------

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete regional hydrological analysis using MarrmotFlow
   """
   
   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import numpy as np
   from marrmotflow import MARRMOTWorkflow
   
   def main():
       # Configuration
       MODELS_TO_TEST = {"HBV-96": 7, "GR4J": 37}
       
       # Load regional dataset
       catchments = gpd.read_file("data/catchments/regional_catchments.shp")
       
       print("Regional Hydrological Analysis")
       print("=" * 35)
       print(f"Number of catchments: {len(catchments)}")
       print(f"Spatial extent: {catchments.total_bounds}")
       print(f"Models to test: {list(MODELS_TO_TEST.keys())}")
       
       # Create workflows for all catchments
       workflows = {}
       
       for idx, catchment in catchments.iterrows():
           catchment_id = catchment.get('catchment_id', f'catchment_{idx}')
           single_catchment = gpd.GeoDataFrame([catchment], crs=catchments.crs)
           
           for model_name, model_number in MODELS_TO_TEST.items():
               workflow_name = f"{catchment_id}_{model_name}"
               
               workflows[workflow_name] = MARRMOTWorkflow(
                   name=workflow_name,
                   cat=single_catchment,
                   forcing_files="data/climate/gridded_climate.nc",
                   forcing_vars={"precip": "precipitation", "temp": "temperature"},
                   forcing_units={"precip": "mm/day", "temp": "celsius"},
                   model_number=model_number,
                   pet_method="penman_monteith"
               )
       
       print(f"\nCreated {len(workflows)} workflows for regional analysis")
       print("Ready for regional hydrological modeling!")
       
   if __name__ == "__main__":
       main()

Key Insights from Regional Analysis
-----------------------------------

This comprehensive example demonstrates:

1. **Spatial heterogeneity**: Hydrological responses vary systematically across regions
2. **Catchment classification**: Physical characteristics can group similar catchments
3. **Model transferability**: Different models may perform better in different regions
4. **Scaling relationships**: Discharge patterns relate to catchment characteristics
5. **Regionalization potential**: Parameter transfer functions can be developed

Applications
------------

* **Ungauged basin prediction**: Transfer parameters to similar catchments
* **Model selection guidance**: Choose appropriate models for different regions
* **Water resource assessment**: Understand regional water availability patterns
* **Climate impact assessment**: Apply regional relationships to future scenarios
* **Monitoring network design**: Identify representative catchments for monitoring
