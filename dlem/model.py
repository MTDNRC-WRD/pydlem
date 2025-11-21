from pathlib import Path
import xarray as xr
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from dlem.functions import latent_heat_vaporization, psychrometric_const, altitude_adjusted_atmp
from dlem.functions import calc_slope_swv_curve, cloud_factor

# Constants
from config import emissivity_water
from config import water_albedo
from config import stefan_boltzman
from config import absolute_zero
from config import water_density
from config import specificheat_water
from config import timestep


class CreateModel:

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.error_codes = None
        self.ice_cover = None

    def load_datafile(self, ncfile_pth):
        self.inputs = xr.open_dataset(ncfile_pth)

    def run_model(self, start, end, sim_ice=False):
        """
        Function that calculates the evaporation rate in mm/day. Code is modified from Zhao 2020
        (https://github.com/gzhaowater/lakeEvap/blob/main/evaporationCalc.py)
        :param start: string - date string formatted "YYYY-MM-DD"
        :param end: string - date string formatted "YYYY-MM-DD"
        :param sim_ice: Boolean - default is False, determines if the ice phenology methods are simulated in the model
        run.
        :return: xr.dataset - xarray dataset with one variable, evaporation rate, or 3 variables
        (evaporation rate, ice on or off, and freeze thaw lags) if sim_ice=True
        """
        # TODO - no matter the start time it seems to return a zero for first timestamp?
        #   Probably should root out the cause but for now just subtract a day from the input start date
        #   and have a 1-day burn in.
        dataset = self.inputs.sel(time=slice(start, end))

        # Check Inputs and Formatting
        # check for long-wave radiation input
        if "lrad" in list(dataset.keys()):
            lrad = True
        else:
            lrad = False

        # Perform Pre-Calculations if necessary
        ta = ((dataset.min_temp + dataset.max_temp) / 2) - 273.15
        depth = dataset.LakeDepth
        area = dataset.LakeArea
        fch = dataset.ftch_len
        ut = dataset.wind_vel
        vpd = dataset.vpd
        srad = dataset.solrad
        lat = dataset.lat.values
        elev = dataset.elev.values

        # check for missing values in each variable array
        ta_nan = np.isnan(ta)
        if np.count_nonzero(ta_nan) > 0:
            warnings.warn(
                "There are missing TEMPERATURE values for the run time period, missing"
                " values will be propogated to the model outputs."
            )
        depth_nan = np.isnan(depth)
        if np.count_nonzero(depth_nan) > 0:
            warnings.warn(
                "There are missing LAKE DEPTH values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        area_nan = np.isnan(area)
        if np.count_nonzero(area_nan) > 0:
            warnings.warn(
                "There are missing LAKE AREA values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        fch_nan = np.isnan(fch)
        if np.count_nonzero(fch_nan) > 0:
            warnings.warn(
                "There are missing FETCH values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        ut_nan = np.isnan(ut)
        if np.count_nonzero(ut_nan) > 0:
            warnings.warn(
                "There are missing WIND SPEED values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        vpd_nan = np.isnan(vpd)
        if np.count_nonzero(vpd_nan) > 0:
            warnings.warn(
                "There are missing VAPOR PRESSURE DEFICIT values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        srad_nan = np.isnan(srad)
        if np.count_nonzero(srad_nan) > 0:
            warnings.warn(
                "There are missing SHORTWAVE RADIATION values for the run time period, missing "
                "values will be propagated to the model outputs."
            )
        lat_nan = np.isnan(lat)
        if np.count_nonzero(lat_nan) > 0:
            warnings.warn(
                "There are missing LATITUDE values for some locations, this will result in "
                "no value outputs for locations missing latitude."
            )
        elev_nan = np.isnan(elev)
        if np.count_nonzero(elev_nan) > 0:
            warnings.warn(
                "There are missing ELEVATION values for some locations, this will result in "
                "no value outputs for locations missing elevation."
            )

        ierr = np.zeros((ta.shape))
        ierr = np.where(depth <= 0, 2, ierr)
        depth = xr.where(depth <= 0, 0.001, depth)
        depth_calc = 4.6 * np.power(area, 0.205)
        # depth = depth if depth < depth_calc else depth_calc
        depth = xr.where(depth < depth_calc, depth, depth_calc)
        # depth = 20.0 if depth>20.0 else depth

        # fch = 10000 if fch > 10000 else fch
        fch = xr.where(fch > 10000, 10000.0, fch)

        ierr = np.where(srad <= 0, 3, ierr)

        ierr = np.where(ut <= 0.01, 4, ierr)
        ut = xr.where(ut <= 0.01, 0.01, ut)

        ierr = np.where(vpd <= 0.0, 5, ierr)
        vpd = xr.where(vpd <= 0.0, 0.0001, vpd)

        es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
        ierr = np.where(es <= vpd, 6, ierr)
        vpd = xr.where(es <= vpd, es * 0.99, vpd)
        ea = es - vpd
        # p=100=pressure of station
        atmp = altitude_adjusted_atmp(ta, ta.elev.values[None, :])
        t_d = (116.9 + 237.3 * np.log(ea)) / (16.78 - np.log(ea))
        twb = (0.00066 * atmp * ta + 4098.0 * (ea) / np.power(t_d + 237.3, 2) * t_d) / (
            0.00066 * atmp + 4098.0 * (ea) / np.power(t_d + 237.3, 2.0)
        )
        ierr = np.where(twb > ta, 7, ierr)
        twb = xr.where(twb > ta, ta, twb)

        self.error_codes = ierr

        ##############################################################################################
        ########################### Calculate water equilibrium temperature ##########################
        ##############################################################################################

        # some variables
        alambda = latent_heat_vaporization(ta)
        # atmp = altitude_adjusted_atmp(ta, ta.elev.values[None, :])
        gamma = psychrometric_const(atmp, alambda)
        # airds = airdens(ta, elev)

        # slope of the saturation water vapour curve at the temperatures (kPa deg C-1)
        deltaa = calc_slope_swv_curve(ta)
        deltawb = calc_slope_swv_curve(twb)

        # Emissivity of air and water (unitless)
        sradj = srad * 0.0864  # convert from W m-2 to MJ m-2 d-1

        fcd = cloud_factor(sradj, lat, elev)
        em_a = (
            1.08
            * (1.0 - np.exp(-np.power(ea * 10.0, (ta + absolute_zero["value"]) / 2016.0)))
            * (1 + 0.22 * np.power(fcd, 2.75))
        )
        em_w = emissivity_water["value"]

        # lradj = em_a * sigma * pow((ta + T_abs), 4.) if lrad == -9999 else lrad * 0.0864
        if lrad:
            lradj = lrad * 0.0864  # convert from W m-2 to MJ m-2 d-1
        else:
            lradj = em_a * stefan_boltzman["value"] * np.power((ta + absolute_zero["value"]), 4.0)

        # wind function using the method of McJannet, 2012 (MJ m-2 d-1 kPa-1)
        windf = (2.33 + 1.65 * ut) * np.power(fch, -0.1) * alambda

        # calculate equilibrium temperature of the water body (C) Zhao and Gao */
        te = (
            (0.46 * em_a + windf * (deltaa + gamma)) * ta
            + (1.0 - water_albedo["value"]) * sradj
            - 28.38 * (em_w - em_a)
            - windf * vpd
        ) / (0.46 * em_w + windf * (deltaa + gamma))

        ###############################################################################################
        ############################## Calculate water column temperature #############################
        ###############################################################################################

        # time constant (d)
        tau = (water_density["value"] * specificheat_water["value"] * depth) / (
            4.0 * stefan_boltzman["value"] * np.power((twb + absolute_zero["value"]), 3.0) + windf * (deltawb + gamma)
        )

        # water column temperature (deg. C)
        # get initial tw0 if no missing values
        if (np.count_nonzero(np.isnan(te)) == 0) and (np.count_nonzero(np.isnan(tau)) == 0):
            ixl, iyl = te.values.shape
            ix = np.arange(ixl)
            iy = np.arange(iyl)
            interp = RegularGridInterpolator((ix, iy), te.values, bounds_error=False, fill_value=None)
            tw0 = interp((ix[0] - 1, iy))

            tw0_tmstmp = [tw0]
            htstrg = []
            for row in range(te.values.shape[0]):
                tw = te.values[row] + (tw0_tmstmp[row] - te.values[row]) * np.exp(-timestep["value"] / tau.values[row])
                tw[tw < 0] = 0
                tw0_tmstmp.append(tw)
                heat_stg = (
                    water_density["value"]
                    * specificheat_water["value"]
                    * depth.values[row]
                    * (tw - tw0_tmstmp[row])
                    / timestep["value"]
                )
                htstrg.append(heat_stg)
            heat_stg = np.stack(htstrg, axis=0)
        # get initial tw0 if missing values
        else:
            warnings.warn(
                "Missing values detected, this may lead to unequal length columns. "
                "Heat Storage calculation will use a slower method."
            )
            interp_te = []
            interp_tau = []
            for c in range(te.values.shape[1]):
                tev = te.values[:, c]
                tev = np.insert(tev, 0, np.nan)
                tauv = tau.values[:, c]

                tev_nans, tev_x = np.isnan(tev), lambda z: z.nonzero()[0]
                tauv_nans, tauv_x = np.isnan(tauv), lambda z: z.nonzero()[0]

                tev[tev_nans] = np.interp(tev_x(tev_nans), tev_x(~tev_nans), tev[~tev_nans])
                tauv[tauv_nans] = np.interp(tauv_x(tauv_nans), tauv_x(~tauv_nans), tauv[~tauv_nans])
                interp_te.append(tev)
                interp_tau.append(tauv)

            te_intrp = np.stack(interp_te, axis=1)
            tau_intrp = np.stack(interp_tau, axis=1)
            tw0 = te_intrp[0]
            te_intrp = np.delete(te_intrp, 0, axis=0)

            # change in heat storage (MJ m-2 d-1)
            tw0_tmstmp = [tw0]
            htstrg = []
            for row in range(te_intrp.shape[0]):
                tw = te_intrp[row] + (tw0_tmstmp[row] - te_intrp[row]) * np.exp(-timestep["value"] / tau_intrp[row])
                tw[tw < 0] = 0
                tw0_tmstmp.append(tw)
                heat_stg = (
                    water_density["value"]
                    * specificheat_water["value"]
                    * depth.values[row]
                    * (tw - tw0_tmstmp[row])
                    / timestep["value"]
                )
                htstrg.append(heat_stg)
            heat_stg = np.stack(htstrg, axis=0)

        ################################################################################################
        ################################### Calculate the evaporation ##################################
        ################################################################################################

        # calculate the Penman evaporation
        rn = (
            sradj * (1.0 - water_albedo["value"])
            + lradj
            - em_w * (stefan_boltzman["value"] * np.power((ta + absolute_zero["value"]), 4.0))
        )

        le = (deltaa * (rn - heat_stg) + gamma * windf * vpd) / (deltaa + gamma)
        evap_hs = le / alambda
        evap_hs = xr.where(evap_hs < 0, 0, evap_hs)
        evap_hs = evap_hs.to_dataset(name="evap")

        ################################################################################################
        ################################### Ice Phenology ##############################################
        ################################################################################################

        if sim_ice:
            print("Estimating ice cover...")
            iceds = simulate_ice(ta, depth)
            evap_hs = xr.where(iceds.ice == 1, 0, evap_hs)
            evap_hs.evap.attrs = {"standard_name": "Evaporation Rate", "units": "mm/day"}
            evap_hs = xr.merge([evap_hs, iceds])
            self.outputs = evap_hs
        else:
            evap_hs.evap.attrs = {"standard_name": "Evaporation Rate", "units": "mm/day"}
            self.outputs = evap_hs

    def save_outputs(self, filepath):
        self.outputs.to_netcdf(filepath)


def simulate_ice(airtemp, depth):
    """
    Function to determine ice on/ice off (ice phenology) of lakes. Based on Zhao et al. (2022) empirical
    ice phenology method.
    :param airtemp: xr.DataArray of mean daily air temperatures - dims (time, location)
    :param depth: xr.DataArray of mean lake depth - dims (time, location)
    :return: xr.Dataset with 2 variables "ice" and "lags" - ice is a boolean or binary array where 1 = ice, 0 = no ice
    lags represents periods of valid/active freeze lags (-1), no lag active (0), and valid/active thaw lags (1)
    """
    # find sign of temp values, above or below freezing
    asign = np.sign(airtemp.values)
    # identify where there are temp reversals (i.e., change from negative to positive and vice versa) > 0 is
    # from freezing to thaw, < 0 is thaw to freezing
    schng = asign - np.roll(asign, 1, axis=0)
    schng[0] = 0
    # make empty freeze array
    fz_arr = np.zeros(schng.shape)
    lags_arr = np.zeros(schng.shape)
    # get indexes of temp reversal
    revsi, revsj = np.nonzero(schng)
    # calc seasonal depths for dealing with potential nans
    dseas = depth.groupby("time.season").mean("time")

    init_lst = []
    # Loop through temp reversals
    for i in tqdm(range(len(revsi))):
        xi = revsi[i]
        yj = revsj[i]
        # get reversal and check type
        rev = schng[xi, yj]
        # get date
        dt = airtemp.time[xi]
        # set initial freeze array values up-to 1st event
        if yj not in init_lst:
            init_t = airtemp.values[:xi, yj]
            if len(init_t) < 30:
                init_mn_t = airtemp.values[: xi + 30, yj].mean()
            else:
                init_mn_t = init_t.mean()

            if init_mn_t <= 0:
                init_frz = 1
            elif init_mn_t > 0:
                init_frz = 0
            else:
                raise ValueError("Too many missing inputs, not enough information to initialize freeze-thaw cycle.")
            fz_arr[:xi, yj] = init_frz
            init_lst.append(yj)

        if rev == -2:  # a freezing event
            # get depth
            d = depth.loc[dt].values[yj]
            # check for nan
            if np.isnan(d):
                d = dseas.loc[depth["time.season"].values[xi]].values[yj]
            else:
                pass
            # calculate freeze lag from Zhao et al. 2022 ice phenology equations
            fzlg = int(5.815 * d**0.626)
            # calculate mean temperature over the freeze lag period
            fz_mn_t = airtemp.values[xi : xi + fzlg, yj]
            if fz_mn_t.size == 0:
                continue
            elif fz_mn_t.mean() < 0:
                if fz_arr[xi - 1, yj] == 1:
                    fz_arr[xi:, yj] = 1
                    lags_arr[xi : xi + fzlg, yj] = -1
                    lags_arr[xi + fzlg :, yj] = 0
                else:
                    lags_arr[xi : xi + fzlg, yj] = -1
                    lags_arr[xi + fzlg :, yj] = 0
                    fz_arr[xi + fzlg :, yj] = 1
            else:
                fz_arr[xi:, yj] = fz_arr[xi - 1, yj]

        elif rev == 2:  # thawing event
            # get year
            Yr = int(dt.dt.year)
            # get winter seasonal mean for year
            ytmp = airtemp.loc["{0}".format(Yr)]
            yssn = ytmp.groupby("time.season").mean("time")
            Twinter = yssn.loc["DJF"].values[yj]
            # calculate thaw lag from Zhao et al. 2022 ice phenology equations
            thwlg = int(-1.003 * Twinter + 21.078)
            # calculate mean temperature over the thaw lag period
            thw_mn_t = airtemp.values[xi : xi + thwlg, yj]
            if thw_mn_t.size == 0:
                continue
            elif thw_mn_t.mean() >= 0:
                if fz_arr[xi - 1, yj] == 0:
                    fz_arr[xi:, yj] = 0
                    lags_arr[xi : xi + thwlg, yj] = 1
                    lags_arr[xi + thwlg :, yj] = 0
                else:
                    lags_arr[xi : xi + thwlg, yj] = 1
                    lags_arr[xi + thwlg :, yj] = 0
                    fz_arr[xi + thwlg :, yj] = 0
            else:
                fz_arr[xi:, yj] = fz_arr[xi - 1, yj]

        else:
            continue

    frz_ds = airtemp.to_dataset(name="AirTemp")
    frz_ds = frz_ds.assign(ice=(("time", "location"), fz_arr))
    frz_ds = frz_ds.assign(lags=(("time", "location"), lags_arr))
    frz_ds = frz_ds.drop_vars(["AirTemp"])

    return frz_ds


# Default behavior create input datafile from gridmet given static reservoir variables and gridmet POR
if __name__ == "__main__":

    pass
