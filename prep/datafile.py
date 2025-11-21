# from prep import lakegeom, metdata
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import Union
import warnings

from config import INPUT_VARS
from config import DSET_COORDS
from prep.metdata import get_gridmet_at_points
from prep.lakegeom import calc_fetch_length

MET_SOURCES = ["gridmet", "daymet", "from_file"]

AREA_METHODS = ["static", "from_elevation", "from_storage", "direct"]

FETCH_METHODS = ["static", "dynamic"]


class CreateInputFile:
    """
    Prepares/provides methods to prepare a xarray Dataset for use in the pydlem model based on input sites/locations
    where the model will calculate lake evaporation (e.g., lake polygon layer in the form of a geopandas GeoDataFrame).
    pydelm required inputs are meteorology (see Penman documentation for which variables), lake area, and lake depth.
    These datasets are required as input to format the datafile.

    :param geoms: geopandas.GeoDataFrame - geometries for which to compute evaporation estimates (point or polygon)
    :param lake_area: pd.Series, pd.DataFrame, xr.DataArray, or xr.Dataset - lake area formatted as one of the accepted
        variable formats. Must either contain multiindex with ['time', 'location'] or those coordinate dimensions
        as xarray object. Can use methods from lakegeom.py module to help with this.
    :param lake_depth: pd.Series, pd.DataFrame, xr.DataArray, or xr.Dataset - lake depth formatted as one of the accepted
        variable formats. Must either contain multiindex with ['time', 'location'] or those coordinate dimensions
        as xarray object. Can use methods from lakegeom.py module to help with this.
    :param index_col: str or None(default) - defines the column name in geoms that will be used as the unique identifier for
        each geometry location. None defaults to the GeoDataFrame index.
    :param met_data: xarray.Dataset or None (default) - if Dataset is provided it is used to construct the input
        datafile from file. If None, met_source must be provided to build initial input datafile.
    :param met_source: str - from accepted sources ['gridmet', 'daymet', 'from_file']
    :return: A formatted datafile class that contains necessary info to save a properly formatted pydelm input netcdf.
    """

    def __init__(
        self,
        geoms: Union[gpd.GeoDataFrame, None],
        lake_area: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
        lake_depth: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
        index_col: Union[str, None] = None,
        met_data: Union[xr.Dataset, None] = None,
        met_source="gridmet",
    ):

        self.data = self._create_metinputs(geoms, index_col, met_data, met_source)
        self.add_variable(lake_area, "LakeArea", var_attrs={"standard_name": "Lake Surface Area", "units": "km^2"})
        self.add_variable(lake_depth, "LakeDepth", var_attrs={"standard_name": "Average Lake Depth", "units": "m"})

    def _create_metinputs(
        self,
        geoms: gpd.GeoDataFrame,
        index_col: Union[str, None],
        met_data: Union[xr.Dataset, None] = None,
        met_source: str = "gridmet",
    ) -> xr.Dataset:
        """
        Function to format a .netcdf meteorology file for input into pydlem.
        __________________

        Valid met_source = ['gridmet', 'daymet', 'from_file']

        :param geoms: geopandas.GeoDataFrame - geometries where meteorology data will be extracted.
        :param met_data: xarray.Dataset or None(default) - Pre-formatted as discrete sampling locations netcdf with dims
            (time, location) for each variable. Must also contain coordinate
            vars ["lat", "lon", "elev", "location", "time"].
        :param met_source: str - a string representing the source of meteorology data to use include
        :return: xarray.DataSet with meteorology variables necessary for Penman Equation calculations at each geometry
            location in geoms.
        """

        if met_source == "from_file":
            if met_data is None:
                raise ValueError("met_data was not defined.")
            else:
                metinputs = met_data

        elif met_source == "gridmet":
            # This is where the addition of start, end dates could be added to provide an alternative to the default
            #   behavior of the met_source = 'gridmet' of downloading the entire GridMET POR.
            metinputs = get_gridmet_at_points(geoms, index_col)

        elif met_source == "daymet":
            print("Sorry, daymet is not yet available. Defaulting to gridment source.")
            metinputs = get_gridmet_at_points(geoms, index_col)

        else:
            raise ValueError("Meteorology source is invalid. Valid entries:{0}".format(MET_SOURCES))

        return metinputs

    def add_variable(self, data, variable_name, var_attrs=None):
        """
        Function to add a variable to an existing input dataset. Only accepts one variable at a time.
        :param data: pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset - pandas objects must be formatted with
        multiindex levels = [time, location] where location index matches those of the class.data object's location ids.
        Time must be datetime64[ns]. Can also be xarray object formatted identical to the class.data object.
        :param variable_name: str - name to assign to new variable from available variables
            ['precip', 'min_temp', 'max_temp', 'solrad', 'min_rh', 'max_rh', 'mean_rh', 'dew_temp',
            'wind_dir', 'wind_vel', 'LakeArea', 'LakeDepth', 'vpd', 'ftch_len']
        :param var_attrs: dict or None(default) - dictionary of attributes and associated values for the new variable
        (usually at minimum includes 'standard_name' and 'units')
        :return: None - updates class data object with new variable
        """
        accepted_vars = [
            "precip",
            "min_temp",
            "max_temp",
            "solrad",
            "min_rh",
            "max_rh",
            "mean_rh",
            "dew_temp",
            "wind_dir",
            "wind_vel",
            "LakeArea",
            "LakeDepth",
            "vpd",
            "ftch_len",
        ]
        if variable_name not in accepted_vars:
            raise ValueError(
                "Variable name not compatible, select from: ['precip', 'min_temp', 'max_temp',"
                " 'solrad', 'min_rh', 'max_rh', 'mean_rh', 'dew_temp','wind_dir',"
                " 'wind_vel', 'LakeArea', 'LakeDepth', 'vpd', 'ftch_len']"
            )

        if isinstance(data, pd.DataFrame):
            data.index.names = ["time", "location"]
            if len(data.columns) > 1:
                print("Too many columns in dataframe, due to ambiguity, no variable was loaded.")
            else:
                print("Variable {0} loaded.".format(data.columns[0]))
                var_nam = data.columns[0]
                if var_nam != variable_name:
                    data.columns = [variable_name]
                    print("Variable {0} renamed to {1}".format(var_nam, variable_name))
                else:
                    print("Variable {0} did not need renaming.".format(var_nam))

                new_ds = xr.merge([self.data, data.to_xarray()])
                new_ds[variable_name].attrs = var_attrs
                self.data = new_ds
                print("New variable added.")
        elif isinstance(data, pd.Series):
            data.index.names = ["time", "location"]
            data = pd.DataFrame(data, columns=[variable_name])
            print("Series loaded and converted to DataFrame with {0} variable.".format(variable_name))
            new_ds = xr.merge([self.data, data.to_xarray()])
            new_ds[variable_name].attrs = var_attrs
            self.data = new_ds
            print("New variable added.")
        elif isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
            data.name = variable_name
            new_ds = xr.merge([self.data, data])
            if var_attrs is None:
                self.data = new_ds
                print("New variable added.")
            else:
                new_ds[variable_name].attrs = var_attrs
                self.data = new_ds
                print("New variable added.")
        else:
            print("Data type is neither pandas Series or Dataframe, no variable loaded.")

    # def add_coordinate(self, data, coordinate_name, var_attrs=None):

    # def format_variables(self,
    #                  precip=None,
    #                  Tmin=None,
    #                  Tmax=None,
    #                  Tmean=None,
    #                  srad=None,
    #                  lrad=None,
    #                  vpd=None,
    #                  windv=None,
    #                  winddir=None,
    #                  time=None,
    #                  lat=None,
    #                  long=None,
    #                  elev=None,
    #                  loc_id=None):

    def save_datafile(self, pthname):
        self.data.to_netcdf(pthname)


def check_format(xrdset):
    """
    :param xrdset: xarray.Dataset - meteorological and lake geometry data formatted to create model for dlem
    :return str - review of xrdset for compatibility with dlem CreateModel class
    """
    vars = [x for x in INPUT_VARS if x not in list(xrdset.data_vars)]
    coords = [x for x in DSET_COORDS if x not in list(xrdset.coords)]
    if len(vars) != 0:
        warnings.warn("There are missing or mislabeled variables in the dataset. See the following:")
        print("MISSING VARIABLES", *vars, sep="\n")
    else:
        print("All necessary variables exist and are labeled properly.")

    if len(coords) != 0:
        warnings.warn("There are missing or mislabeled coordinates in the dataset. See the following:")
        print("MISSING COORDINATES", *coords, sep="\n")
    else:
        print("All necessary coordinates exist and are labeled properly")


# def format_input_variables(precip=None,
#                            min_temp=None,
#                            max_temp=None,
#                            mean_temp=None,
#                            srad=None,
#                            lrad=None,
#                            vpd=None,
#                            wind_vel=None,
#                            wind_dir=None,
#                            LakeArea=None,
#                            LakeDepth=None,
#                            ftch_len=None,
#                            time=None,
#                            lat=None,
#                            long=None,
#                            elev=None,
#                            location=None):
#     dlem_inputs = ['precip', 'mean_temp', 'solrad', 'lrad', 'vpd', 'wind_vel', 'LakeArea', 'LakeDepth', 'ftch_len',
#                    'location', 'lat', 'long', 'elev', 'time']


def prep_precip(met_data):
    """
    Function to create a xr.DataArray for precipitation as a length from xr.Dataset with precipitation as volume.
    :param met_data: xarray.Dataset - Pre-formatted as discrete with dims (time, location) for each variable. Must also
     contain coordinate "area" and data variable "precip_volume" with units in m^2.
    :return: xarray.DataArray - precipitation data with units of length (mm)
    """
    # divide precip volume(m^3) by lake area (converted from km^2 to m^2). Returns precip (m)
    precip_arr = met_data.precip_volume.values / (np.tile(met_data.area.values, (len(met_data.time), 1)) * (1000**2))
    precip_arr = precip_arr * 1000  # convert from m to mm
    precip_xda = xr.DataArray(
        precip_arr, coords=met_data.precip_volume.coords, attrs={"standard_name": "Precipitation", "units": "mm"}
    )
    precip_xda.name = "precip"

    return precip_xda


def prep_lakearea(met_data):
    """
    Function to format lake area variable into array shape accepted by DLEM model.
    :param met_data: xarray.Dataset - Pre-formatted as discrete with dims (time, location) for each variable. Must also
     contain coordinate "area" with units in km^2.
    :return: xarray.DataArray - lake area with array dimensions equal to met_data data variable shape
    """
    lareas_arr = np.tile(met_data.coords["area"], (len(met_data.time + 1), 1))
    lake_area_xda = xr.DataArray(
        lareas_arr, coords=met_data.coords, attrs={"standard_name": "Lake Surface Area", "units": "km^2"}
    )
    lake_area_xda.name = "LakeArea"

    return lake_area_xda


def prep_lakedepth(met_data, gdf, gdf_index_col, gdf_depth_col):
    """
    Function to format lake depth variable into array shape accepted by DLEM model.
    :param met_data: xarray.Dataset - Pre-formatted as discrete with dims (time, location) for each variable.
    :param gdf: geopandas.GeoDataFrame - contains depth data
    :param gdf_index_col: str - name of column in GeoDataFrame to use as a unique identifier for each geometry
    :param gdf_depth_col: str - name of column in GeoDataFrame with depth value of each geometry. Units are in m
    :return: xarray.DataArray - lake depth with array dimensions equal to met_data data variable shape
    """
    gdf[gdf_index_col] = gdf[gdf_index_col].astype(str)

    ldpth = np.array(
        [gdf[gdf[gdf_index_col] == x][gdf_depth_col].iloc[0] for x in met_data.location.values.astype(str)]
    )
    ldpth_arr = np.tile(ldpth, (len(met_data.time + 1), 1))
    lake_depth_arr = xr.DataArray(
        ldpth_arr, coords=met_data.coords, attrs={"standard_name": "Lake Depth", "units": "m"}
    )
    lake_depth_arr.name = "LakeDepth"

    return lake_depth_arr


def prep_fetch(met_data, gdf, gdf_index_col):
    """
    Function to calculate and format fetch length variable into array shape accepted by DLEM model.
    :param met_data: xarray.Dataset - Pre-formatted as discrete with dims (time, location) for each variable. Must also
     contain coordinate the data variables "wind_dir" and "LakeArea".
    :return: xarray.DataArray - fetch length with array dimensions equal to met_data data variable shape
    """
    # assign necessary variables and complete any conversions
    rows = met_data.wind_dir.shape[0]
    cols = met_data.wind_dir.shape[1]
    wnd_d = met_data.wind_dir.values
    lk_a = met_data.LakeArea.values * 1000.0**2

    # loop through data arrays of wind direction for each timestamp(x) and location(y)
    resvs = gdf.to_crs(gdf.estimate_utm_crs())
    xs = []
    for x in range(0, rows):
        ys = []
        for y in range(0, cols):
            gmtry = resvs.loc[resvs[gdf_index_col] == met_data.location[0].values.astype(str)].geometry
            wnd = wnd_d[x, y]
            a = lk_a[x, y]
            ftchl = calc_fetch_length(gmtry, wnd, a)
            ys.append(ftchl)
        xs.append(ys)
    # create xarray DataArray from the resulting ndarray
    flen = np.array(xs)
    flen = xr.DataArray(flen, coords=met_data.coords, attrs={"standard_name": "Wind Fetch Length", "units": "m"})

    return flen


# Default behavior create input datafile from gridmet given static reservoir variables and gridmet POR
if __name__ == "__main__":

    pass
