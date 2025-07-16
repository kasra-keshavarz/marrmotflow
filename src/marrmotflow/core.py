"""Core functionality for MarrmotFlow package."""
# internal imports
from ._default_dicts import (
    default_forcing_units,
    default_forcing_vars,
    default_model_dict
)

from .templating import render_models

# built-in imports
import glob
import os
import warnings
import json
import re

from typing import Dict, Sequence, Union

# third-party imports
import xarray as xr
import geopandas as gpd
import pandas as pd
import pint_xarray
import pint
import pyet
import numpy as np
import timezonefinder

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
        name: str = "MARRMOTModels",
        cat: gpd.GeoDataFrame | PathLike = None,
        forcing_vars: Dict[str, str] = {},
        forcing_files: Sequence[PathLike] | PathLike = None, # type: ignore
        forcing_units: Dict[str, str] = {},
        pet_method: str = "penman_monteith",
        model_number: Sequence[int] | int = [7, 37], # HBV-96 and GR4J as default models
        forcing_time_zone: str = None,
        model_time_zone: str = None,
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
        # assing the name of the workflow
        self.name = name

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

        # assign time zones
        self.forcing_time_zone = forcing_time_zone
        self.model_time_zone = model_time_zone

        # assign an output object
        self.output_mat = None  # Placeholder for output matrix

    @classmethod
    def from_dict(
        cls: 'MARRMOTWorkflow',
        init_dict: Dict = {},
    ) -> 'MARRMOTWorkflow':
        """
        Constructor to use a dictionary to instantiate
        """
        if len(init_dict) == 0:
            raise KeyError("`init_dict` cannot be empty")
        assert isinstance(init_dict, dict), "`init_dict` must be a `dict`"

        return cls(**init_dict)

    @classmethod
    def from_json(
        cls: 'MARRMOTWorkflow',
        json_str: str,
    ) -> 'MARRMOTWorkflow':
        """
        Constructor to use a loaded JSON string
        """
        # building customized MARRMOTWorkflow's JSON string decoder object
        decoder = json.JSONDecoder(object_hook=MARRMOTWorkflow._json_decoder)
        json_dict = decoder.decode(json_str)
        # return class instance
        return cls.from_dict(json_dict)

    @classmethod
    def from_json_file(
        cls: 'MARRMOTWorkflow',
        json_file: 'str',
    ) -> 'MARRMOTWorkflow':
        """
        Constructor to use a JSON file path
        """
        with open(json_file) as f:
            json_dict = json.load(f, object_hook=MARRMOTWorkflow._json_decoder)

        return cls.from_dict(json_dict)

    @staticmethod
    def _env_var_decoder(s):
        """
        OS environmental variable decoder
        """
        # RE patterns
        env_pat = r'\$(.*?)/'
        bef_pat = r'(.*?)\$.*?/?'
        aft_pat = r'\$.*?(/.*)'
        # strings after re matches
        e = re.search(env_pat, s).group(1)
        b = re.search(bef_pat, s).group(1)
        a = re.search(aft_pat, s).group(1)
        # extract environmental variable
        v = os.getenv(e)
        # return full: before+env_var+after
        if v:
            return b+v+a
        return s

    @staticmethod
    def _json_decoder(obj):
        """
        Decoding typical JSON strings returned into valid Python objects
        """
        if obj in ["true", "True", "TRUE"]:
            return True
        elif obj in ["false", "False", "FALSE"]:
            return False
        elif isinstance(obj, str):
            if '$' in obj:
                return MARRMOTWorkflow._env_var_decoder(obj)
            if MARRMOTWorkflow._is_valid_integer(obj):
                return int(obj)
        elif isinstance(obj, dict):
            return {MARRMOTWorkflow._json_decoder(k): MARRMOTWorkflow._json_decoder(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def datetime64_to_matlab_datenum_fast(dt64):
        """
        Python's datetime64 to MATLAB datenum conversion.
        """
        dt64 = pd.to_datetime(dt64)
        matlab_epoch = 719529  # 1970-01-01
        unix_epoch = np.datetime64('1970-01-01', 'D')
        days = (dt64.values - unix_epoch) / np.timedelta64(1, 'D')

        # Convert to MATLAB datenum format in integer format
        return (matlab_epoch + days).astype(int)

    # class methods
    def run(self):
        """Run the workflow."""
        self.init_forcing_files() # defines self.df
        self.init_pet() # defines self.pet
        self.init_model_file(self.model_number)

        # print a message about the timezones
        print(f"Using forcing time zone: {self.forcing_time_zone}")
        print(f"Using model time zone: {self.model_time_zone}")

        return f"Workflow executed successfully with {len(self.forcing_files)} forcing files."

    def save(self, save_path: PathLike): # type: ignore
        """Save the workflow output to a specified path."""
        if self.forcing is None or self.pet is None:
            raise ValueError("No output matrix to save. Run the workflow first.")

        # Create .mat file using the scipy.io.savemat function
        # the dataframe must be a cobination of self.df and self.pet
        combined_data = {
            'name': self.name,
            'gaugeID': f'{self.name} gauge',
            'dates_as_datenum': MARRMOTWorkflow.datetime64_to_matlab_datenum_fast(self.forcing.index),
            'precip': self.forcing['precip'],
            'temp': self.forcing['temp'],
            'pet': self.pet,
            'delta_t': 1, # Assuming daily data, so delta_t is 1 day
        }

        # create the output directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save the combined data to a .mat file
        savemat(os.path.join(save_path, 'marrmot_data.mat'), {'marrmot_data': combined_data})

        # save the model file into a .m file
        model_file_path = os.path.join(save_path, 'marrmot_model.m')
        with open(model_file_path, 'w') as f:
            f.write(self.model)

        return f"Outputs saved to {save_path}"

    def __str__(self):
        return f"MARRMOTWorkflow(forcing_files={self.forcing_files})"
    
    def init_forcing_files(self):
        """Initialize forcing files."""
        # check the timezones
        if self.forcing_time_zone is None:
            warnings.warn(
                "Forcing time zone is not set. Defaulting to 'UTC'.",
                UserWarning
            )
            self.forcing_time_zone = "UTC"

        # check the model time-zone
        if self.model_time_zone is None:
            # try extracting the time zone from the provided `cat` file
            # calculate area's centroid coordinates---basically what is done
            # in the `init_class` method
            warnings.warn(
                "No `model_time_zone` provided in the settings. "
                "Autodetecting the time zone using `timezonefinder` "
                "based on the centroid coordinates of the catchment.",
                UserWarning,
            )
            if not self.cat.crs:
                warnings.warn(
                    "Catchment (cat) does not have a CRS. A default of "
                    "EPSG:4326 will be used for latitude calculation.",
                    UserWarning)
                self.cat.set_crs(epsg=4326, inplace=True)

            # if crs is not set to EPSG:4326, then convert it to EPSG:4326
            elif self.cat.crs != 'EPSG:4326':
                self.cat.to_crs(epsg=4326, inplace=True)

            # calculate the latitude from the catchment geometry
            lat = self.cat.geometry.centroid.y.mean()
            lng = self.cat.geometry.centroid.x.mean()

            # extracing the model time zone from the coordinates
            self.model_time_zone = timezonefinder.TimezoneFinder().timezone_at(
                lat=lat,
                lng=lng
            )

            # Print the model time zone
            if self.model_time_zone:
                warnings.warn(
                    f"Autodetected model time zone: {self.model_time_zone}",
                    UserWarning,
                )
            # if the model time zone is None, then assume UTC
            # and warn the user
            else:
                self.model_time_zone = 'UTC'
                warnings.warn(
                    "No `model_time_zone` provided in the settings and"
                    " autodetection using `timezonefinder` failed."
                    " Assuming UTC time zone.",
                    UserWarning,
                )

        _ureg = pint.UnitRegistry(force_ndarray_like=True)

        # read the forcing files using xarray and create a dataset
        ds = xr.open_mfdataset(
            self.forcing_files,
            combine='by_coords',
            parallel=False,
            engine='netcdf4'
        )

        # adjust the model time zone
        if self.model_time_zone != self.forcing_time_zone:
            # convert the dataset to the forcing time zone
            ds = ds.assign_coords({
                   'time': ds.time.to_index().tz_localize(self.forcing_time_zone).tz_convert(self.model_time_zone).tz_localize(None)
                })

        # rename the dataset variables to match the forcing_vars
        # and assign pint units to the dataset variables
        rename_vars_dict = {}
        for key, value in self.forcing_vars.items():
            if value in ds.variables:
                rename_vars_dict[value] = default_forcing_vars.get(key)
        ds = ds.rename(rename_vars_dict)

        # drop the variable not in self.forcing_vars
        ds = ds[list(default_forcing_vars.values())]

        # assign pint units to the dataset variables
        renamed_forcing_units = {}
        for key, value in self.forcing_units.items():
            new_key = default_forcing_vars.get(key)
            renamed_forcing_units[new_key] = value
        ds = ds.pint.quantify(units=renamed_forcing_units, unit_registry=_ureg)

        # convert the dataset units to the default forcing units
        renamed_to_forcing_units = {}
        for key, value in default_forcing_units.items():
            new_key = default_forcing_vars.get(key)
            renamed_to_forcing_units[new_key] = value
        ds = ds.pint.to(units=renamed_to_forcing_units)

        # after unit conversion, dequantify the dataset
        ds = ds.pint.dequantify()

        # resample the dataset to daily frequency if it is not already
        # first converting the dataset to a pandas.DataFrame
        df = ds.to_dataframe()
        # if a multi-index is present, drop any level not named `time`
        if df.index.nlevels > 1:
            df = df.reset_index(level=[level for level in df.index.names if level != 'time'])

        # resample the precipitation varaible to daily frequency
        df = df.resample('D').mean()

        # drop any column not named 'precip' or 'temp'
        df = df[list(default_forcing_vars.values())]

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

        # calculate the latitude from the catchment geometry
        lat = self.cat.geometry.centroid.y.mean()

        # calculate the potential evapotranspiration using the Hamon method
        self.pet = pyet.temperature.hamon(
            tmean=self.forcing['temp'],
            lat=lat,
        )

        return

    def init_model_file(
        self,
        model_number: Sequence[int] | int
    ) -> None:
        """Initialize the model file for the given model number."""

        if isinstance(model_number, int):
            model_number = [model_number]

        # create a list of model files using the model_number
        model_files = []
        for num in model_number:
            # check if the model number is in the default model dict
            if num in default_model_dict:
                model_files.append(default_model_dict[num])
            else:
                raise ValueError(f"Model number {num} in MARRMoT is not supported.")
                    
        # create the content of the model file
        self.model = render_models(model_files)

        # return the rendered content
        return