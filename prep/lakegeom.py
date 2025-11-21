## Functions to calculate reservoir data from eac-curves, mean depth (storage/area), and
##  width variable in the fetch term of DLEM.
import pandas as pd
import numpy as np
import geopandas as gpd


def area_from_eac_curve(indata, curve, in_type='storage'):
    """
    Takes input reservoir data as storage or elevation and converts to surface area
    using an elevation-area-capacity curve.
    :param indata: pd.Series - Input data (either elevation or storage)
    :param curve: pd.DataFrame - Table with 3 columns in the order [elevation, area, storage]
    :param in_type: str - value either 'storage' or 'elevation' to specify what the input data are,
    default is 'storage'
    :return: pd.DataFrame - Timeseries of reservoir surface area & storage if using elevation as an input
    """
    inx = indata.values
    s_curv = curve.sort_values(by=curve.columns[0], ascending=True)

    if in_type == 'storage':
        xa = s_curv.iloc[:,2].values
        ya = s_curv.iloc[:,1].values
        area = np.interp(inx, xa, ya)
        dfout = pd.DataFrame({'LakeArea': area}, index=indata.index)
    elif in_type == 'elevation':
        xa = s_curv.iloc[:,0].values
        ya = s_curv.iloc[:,1].values
        xs = s_curv.iloc[:,0].values
        ys = s_curv.iloc[:,2].values
        area = np.interp(inx, xa, ya)
        stor = np.interp(inx, xs, ys)
        dfout = pd.DataFrame({'LakeArea': area,
                              'Storage': stor}, index=indata.index)
    else:
        print("Area not computed, 'in_type' argument is not recognized.")
        dfout = None

    return dfout


def calc_lake_depth(lk_area_array, lk_str_array):
    """
    Calculates the mean, daily lake depth from area and storage volume (make sure units are the same)
    :param lk_area_array: pd.Series - multiindex Series levels = [datetime, location]
    :param lk_str_array: pd.Series - multiindex Series levels = [datetime, location]
    :return: pd.DataFrame - multiindex Frame of average depth in units of inputs
    """
    dpth = lk_str_array / lk_area_array
    dpth_DF = pd.DataFrame(dpth, columns=['LakeDepth'])

    return dpth_DF


def calc_fetch_length(lake_shp, wind_dir, lake_area):
    """
    Calculates the fetch length, the tangent lines of the lake polygon parallel to the wind direction. Make sure
    :param lake_shp: gpd.GeoSeries or gpd.GeoDataFrame - with 1 row, the polygon representing the shape of the lake in
    UTM coordinates
    :param wind_dir: float - daily wind direction in degrees clockwise from North
    :param lake_area: float - the lake surface area in square meters
    :return: float - daily fetch length in meters
    """
    rot = lake_shp.rotate(wind_dir)
    width = rot.bounds.maxx.values[0] - rot.bounds.minx.values[0]
    llength = lake_area / width
    return llength
