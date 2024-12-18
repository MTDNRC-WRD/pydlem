from pathlib import Path
import tomli

path = Path(__file__).parent / 'pydlem_settings.toml'
with path.open(mode='rb') as f:
    settings = tomli.load(f)

atmosphericP_sealevel = settings['CONSTANTS']['atmosphericP_sealevel']
specificheat_air = settings['CONSTANTS']['specificheat_air']
specificheat_water = settings['CONSTANTS']['specificheat_water']
molecularweight = settings['CONSTANTS']['molecularweight']
water_albedo = settings['CONSTANTS']['water_albedo']
emissivity_water = settings['CONSTANTS']['emissivity_water']
stefan_boltzman = settings['CONSTANTS']['stefan_boltzman']
absolute_zero = settings['CONSTANTS']['absolute_zero']
water_density = settings['CONSTANTS']['water_density']
timestep = settings['CONSTANTS']['timestep']
GRIDMET_PARAMS = settings['GRIDMET']['GRIDMET_PARAMS']
GRIDMET_BOUNDS = settings['GRIDMET']['GRIDMET_BOUNDS']
GRIDMET_NROWS = settings['GRIDMET']['GRIDMET_NROWS']
GRIDMET_NCOLS = settings['GRIDMET']['GRIDMET_NCOLS']
GRIDMET_XRES = settings['GRIDMET']['GRIDMET_XRES']
GRIDMET_YRES = settings['GRIDMET']['GRIDMET_YRES']
INPUT_VARS = settings['DLEM']['INPUT_VARS']
DSET_COORDS = settings['DLEM']['DSET_COORDS']
