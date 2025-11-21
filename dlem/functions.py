import numpy as np
import pandas as pd

# Constants
from config import atmosphericP_sealevel
from config import specificheat_air
from config import molecularweight


# This is verbatum from Zhao et al. 2019, 2022 - 2020 code repo (https://github.com/gzhaowater/lakeEvap)
# Equation from  Reference: Harrison (1963)
def latent_heat_vaporization(Ta):
    """
    Function to calculate the latent heat of vaporization variable in Penman Equation.
    :param Ta: pd.Series - time series of Air Temperature
    :return: pd.Series - time series of resulting lambda values
    """
    alambda = 2.501 - (Ta * 2.361e-3)  # From Zhao et al. 2022
    return alambda


# This is verbatum from Zhao et al. 2019, 2022 - 2020 code repo (https://github.com/gzhaowater/lakeEvap)
def psychrometric_const(atmp_adj, lat_ht_vap):
    gamma = ((specificheat_air['value'] / molecularweight['value']) * atmp_adj) / lat_ht_vap
    return gamma


# This is verbatum from Zhao et al. 2019, 2022 - 2020 code repo (https://github.com/gzhaowater/lakeEvap)
def altitude_adjusted_atmp(Ta, elev):
    atmp = atmosphericP_sealevel['value'] * np.power((273.15 + Ta - 0.0065 * elev) / (273.15 + Ta), 5.26)
    return atmp


# This is verbatum from Zhao et al. 2019, 2022 - 2020 code repo (https://github.com/gzhaowater/lakeEvap)
def calc_slope_swv_curve(Ta):
    # slope of the saturation water vapour curve
    # at each air temperature (kPa deg C-1)

    ea = 0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))
    delcalc = 4098 * ea / np.power((Ta + 237.3), 2.0)
    return delcalc


# This is verbatum from Zhao et al. 2019, 2022 see 2020 code repo (https://github.com/gzhaowater/lakeEvap)
def cloud_factor(srad, lat, elev):
    """
    Calculations copied from Zhao et al. 2019, 2022 - code from 2020 repo((https://github.com/gzhaowater/lakeEvap).
    Calculates cloud factor for net radiation term in Penman Equation.
    :param srad: np.array with 2D shape (time, locations) - Daily downward shortwave radiation in MJ/m^2 * day
    :param lat: np.array 1D - Latitude of the lake/location corresponding to the locations dimension of srad
    :param elev: np.array 1D - Elevation of the lake/location corresponding to the locations dimension of srad
    :return:
    """

    # srad is incoming shortwave radiation (MJ/m2/d) and mth is the month
    # J, omega, delta, dr
    # Kso, Ket, Kr, fcd
    # lat_r

    J = pd.DatetimeIndex(srad.time).dayofyear.values
    J = J.astype(int)
    delta = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)
    lat_r = lat[None, :] / 180.0 * np.pi
    omega_input = -np.tan(lat_r) * np.tan(delta[:, None])

    omega = np.arccos(omega_input)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * J)
    Ket = 24.0 / np.pi * 4.92 * dr[:, None] * (
                omega * np.sin(lat_r) * np.sin(delta[:, None]) + np.cos(lat_r) * np.cos(delta[:, None]) * np.sin(omega))
    Kso = (0.75 + 2e-5 * elev[None, :]) * Ket
    Kr = srad / Kso

    # Kr = 1.0 if Kr > 1.0 else Kr
    Kr = np.where(Kr > 1.0, 1.0, Kr)
    # Kr = 0.0 if Kr < 0.0 else Kr
    Kr = np.where(Kr < 0.0, 0.0, Kr)

    fcd = 1.0 - Kr

    fcd = np.where(omega_input < 1, fcd, 0)  # Polar night

    return fcd
