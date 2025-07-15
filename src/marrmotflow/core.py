"""Core functionality for MarrmotFlow package."""
# internal imports
from ._default_dicts import default_forcing_units

# built-in imports
import glob
from typing import Dict, Sequence, Union
import os
import warnings


# third-party imports
import xarray as xr
import geopandas as gpd
import pandas as pd
import pint_xarray
import pint
import pyet

from scipy.io import savemat

# define types
try:
    from os import PathLike
except ImportError:
    PathLike = str
else:
    PathLike = Union[str, PathLike]

class MARRMOTWorkflow:
    """A class representing a MARRMOT workflow."""
    
    def __init__(
        self,
        cat: gpd.GeoDataFrame | PathLike = None,
        forcing_vars: Dict[str, str] = {},
        forcing_files: Sequence[PathLike] | PathLike = None, # type: ignore
        forcing_units: Dict[str, str] = {},
        pet_method: str = "penman_monteith",
        model_number: Sequence[int] | int = [7, 37], # HBV-96 and GR4J as default models
    ) -> 'MARRMOTWorkflow':
        """
        Initialize the MARRMOT workflow with forcing variables, files, units, PET method, and model number.

        Parameters
        ----------
        forcing_vars : Dict[str, str], optional
            Dictionary of forcing variables and their units, by default {}.
            The mandatory keys to this dictionary are:
            - 'precip': Precipitation variable name
            - 'temp': Temperature variable name
            The values must be present in the `forcing_files`.
        forcing_files : Sequence[PathLike] | PathLike, optional
            Sequence of paths to forcing files or a single path, by default None
        forcing_units : Dict[str, str], optional
            Dictionary of units for the forcing variables, by default {}.
            The keys must match the keys in `forcing_vars`, and the values
            must be valid pint units.
        pet_method : str, optional
            Method for calculating potential evapotranspiration, by default "hamon"
        model_number : Sequence[int] | int, optional
            Model number(s) to be used in the workflow, by default [7, 37] (HBV-96 and GR4J)

        Raises
        ------
        ValueError
            If forcing files are not provided or are not in the correct format.
        TypeError
            If forcing files are not a sequence or a PathLike object.

        Notes
        -----
        - `pet_method` only accepts "hamon" as a valid method. Other methods
           will be added in the future.
        """
        # assign the catchment (cat) as a GeoDataFrame or PathLike
        if cat is None:
            raise ValueError("Catchment (cat) must be provided as a GeoDataFrame or PathLike.")
        if isinstance(cat, gpd.GeoDataFrame):
            self.cat = cat
        elif isinstance(cat, PathLike):
            self.cat = gpd.read_file(cat)
        else:
            raise TypeError("cat must be a GeoDataFrame or a PathLike object.")
        
        # assign forcing variables
        self.forcing_vars = forcing_vars

        # if not a list of forcing files and is a PathLike, read the files using glob.glob
        if isinstance(forcing_files, PathLike):
            forcing_files = glob.glob(os.path.join(forcing_files, "**/*.nc*"), recursive=True)

        # if not, then the user must provide a list of forcing files in NetCDF format
        # assigning a class attribute
        self.forcing_files = forcing_files

        # assign forcing units
        self.forcing_units = forcing_units

        # assign pet method
        self.pet_method = pet_method

        # assign model number
        if isinstance(model_number, int):
            model_number = [model_number]
        self.model_number = model_number

        # assign an output object
        self.output_mat = None  # Placeholder for output matrix

    def run(self):
        """Run the workflow."""
        self.init_forcing_files() # defines self.df
        self.init_pet() # defines self.pet

        return f"Running workflow: {self.name}"
    
    def save(self, output_path: PathLike):
        """Save the workflow output to a specified path."""
        if self.output_mat is None:
            raise ValueError("No output matrix to save. Run the workflow first.")

        # Create .mat file using the scipy.io.savemat function
        # the dataframe must be a cobination of self.df and self.pet
        combined_data = {
            'precip': self.df['precip'],
            'temp': self.df['temp'],
            'pet': self.pet['pet'],
        }

        savemat(output_path, combined_data)

    def __str__(self):
        return f"MARRMOTWorkflow(forcing_files={self.forcing_files})"
    
    def init_forcing_files(self):
        """Initialize forcing files."""
        _ureg = pint.UnitRegistry(force_ndarray_like=True)

        # read the forcing files using xarray and create a dataset
        ds = xr.open_mfdataset(
            self.forcing_files,
            combine='by_coords',
            parallel=False,
            engine='netcdf4'
        )
        
        # rename the dataset variables to match the forcing_vars
        # and assign pint units to the dataset variables
        ds = ds.rename(self.forcing_vars)

        # drop the variable not in self.forcing_vars
        ds = ds[list(self.forcing_vars.keys())]

        # assign pint units to the dataset variables
        ds = ds.pint.quantify(units=self.forcing_units, unit_registry=_ureg)

        # convert the dataset units to the default forcing units
        ds = ds.pint.to(units=default_forcing_units)

        # after unit conversion, dequantify the dataset
        ds = ds.pint.dequantify()

        # resample the dataset to daily frequency if it is not already
        # first converting the dataset to a pandas.DataFrame
        df = ds.to_dataframe()

        # resample the precipitation varaible to daily frequency
        if 'precip' in df.columns:
            df_precip = df['precip'].resample('D').sum()

        # resample the tempearture variable to daily frequency
        if 'temp' in df.columns:
            df_temp = df['temp'].resample('D').mean()

        # create a new dtaframe with the resampled data
        df = pd.DataFrame({
            'precip': df_precip,
            'temp': df_temp
        })

        # create the new attribute
        self.forcing = df

        return

    def init_pet(self):
        """Initialize potential evapotranspiration (PET) calculation."""
        if self.pet_method != "penman_monteith":
            raise ValueError(f"Invalid PET method: {self.pet_method}. Only 'penman_monteith' is supported.")
        
        # FIXME: More flexibility is needed in the future for different PET methods
        # extract the latitude from the catchment object, first checking if `cat` has a crs
        if not self.cat.crs:
            warnings.warn(
                "Catchment (cat) does not have a CRS. A default of "
                "EPSG:4326 will be used for latitude calculation.",
                UserWarning)
            self.cat.set_crs(epsg=4326, inplace=True)
        # if crs is not set to EPSG:4326, then convert it to EPSG:4326
        elif self.cat.crs != 'EPSG:4326':
            self.cat.to_crs(epsg=4326, inplace=True)

        else:
            raise ValueError("Cannot calculate latitude for the provided `cat` argument.")

        # calculate the latitude from the catchment geometry
        lat = self.cat.geometry.centroid.y.mean()

        # calculate the potential evapotranspiration using the Hamon method
        self.pet = pyet.temperature.hamon(
            tmean=self.df['temp'],
            lat=lat,
        )

        # assign the PET values to the dataframe
        self.df['pet'] = self.pet


        return
    
    def init_params(self):
        """Initialize model parameters."""
        