# analysis_modules.py
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import numba
import rasterio
from rasterio.transform import from_origin

from numba import jit

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def min_max_normalize(series):
    """Normalize a pandas Series to range [0,1]."""
    s = series.fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_max - s_min == 0:
        return s.apply(lambda x: 0.0 if s_max == 0 else 1.0)
    return (s - s_min) / (s_max - s_min)

def compute_index_for_factor_high(gdf, factor_name, config):
    """
    For factors where higher raw values are better.
    Uses the raw field specified in config.dataset_info[factor_name]['raw']
    and writes the normalized index into the column specified in config.dataset_info[factor_name]['alias'].
    """
    info = config.dataset_info[factor_name]
    raw_col = info["raw"]
    index_col = info["alias"]
    if raw_col not in gdf.columns:
        gdf[raw_col] = 0.0
    gdf[index_col] = min_max_normalize(gdf[raw_col])
    return gdf

def compute_index_for_factor_low(gdf, factor_name, config):
    """
    For factors where lower raw values are better.
    Uses the raw field specified in config.dataset_info[factor_name]['raw']
    and writes the normalized (inverted) index into the column specified in config.dataset_info[factor_name]['alias'].
    """
    info = config.dataset_info[factor_name]
    raw_col = info["raw"]
    index_col = info["alias"]
    if raw_col not in gdf.columns:
        gdf[raw_col] = 0.0
    normalized = min_max_normalize(gdf[raw_col])
    gdf[index_col] = 1 - normalized
    return gdf

def area_weighted_average(buffer_geom, features_gdf, field_name):
    """
    Compute the area-weighted average for the specified field over the intersections 
    between a buffer geometry and the features in a GeoDataFrame.
    
    Parameters:
      buffer_geom (shapely.geometry): The geometry (e.g., a buffer) to use for intersection.
      features_gdf (GeoDataFrame): A GeoDataFrame of features containing the field.
      field_name (str): The name of the field for which to compute the average.
    
    Returns:
      float: The area-weighted average of the field, or NaN if no intersections occur.
    """
    total_area = 0.0
    weighted_sum = 0.0

    # Loop over each feature in the GeoDataFrame
    for idx, row in features_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Compute the intersection between the buffer and the feature's geometry
        intersection = buffer_geom.intersection(geom)
        if not intersection.is_empty:
            area = intersection.area
            try:
                value = float(row.get(field_name, 0))
            except (ValueError, TypeError):
                continue
            weighted_sum += value * area
            total_area += area

    return weighted_sum / total_area if total_area > 0 else float('nan')

###############################################################################
# ADAPTABILITY ANALYSIS – Raw Value Computation Integration
###############################################################################
def compute_raw_adaptability(gdf, config):
    """
    Compute the raw building adaptability value.
    
    This function calculates a weighted sum of normalized building characteristics:
      - "CAPACITY"
      - "BldgArea"
      - "StrgeArea"
    
    Each field is normalized to the range [0,1] using min_max_normalize before weighting.
    The weights are taken from config.COMPONENTS["Adaptability_Index_components"].
    The result is stored in the column defined in config.dataset_info["Adaptability_Index"]["raw"]
    (for example, "bldg_adapt").
    """
    # Ensure the necessary fields exist, are numeric, and then normalize them.
    for field in ["CAPACITY", "BldgArea", "StrgeArea"]:
        if field not in gdf.columns:
            gdf[field] = 0.0
        else:
            gdf[field] = pd.to_numeric(gdf[field], errors='coerce').fillna(0.0)
        # Normalize each field to the range [0,1]
        gdf[field] = min_max_normalize(gdf[field])
    
    # Retrieve the component weights from the config.
    weights = config.COMPONENTS["Adaptability_Index_components"]
    
    # Compute the weighted sum of the normalized values.
    gdf["bldg_adapt"] = (
        gdf["CAPACITY"] * weights.get("CAPACITY", 0) +
        gdf["BldgArea"] * weights.get("BldgArea", 0) +
        gdf["StrgeArea"] * weights.get("StrgeArea", 0)
    )
    logger.info("Raw adaptability values computed and stored in 'bldg_adapt'.")
    return gdf

def compute_adaptability_index(gdf, config):
    """
    Compute the Adaptability Index.
    
    First, the raw adaptability value is computed as a weighted sum of building characteristics.
    Then, because higher values of adaptability are better, the raw value is normalized using a
    high-is-good normalization.
    
    The raw value is stored in the column given by config.dataset_info["Adaptability_Index"]["raw"]
    (e.g. "bldg_adapt"), and the normalized index is stored in the column given by the alias
    (e.g. "Adaptability").
    """
    gdf = compute_raw_adaptability(gdf, config)
    gdf = compute_index_for_factor_high(gdf, "Adaptability_Index", config)
    return gdf


###############################################################################
# SOLAR ANALYSIS – Raw Value Computation Integration
###############################################################################
import geopandas as gpd
import pandas as pd
import numpy as np
import numba
from numba import njit
import logging
import pvlib
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import time
import shapely.geometry

# (1) NumPy/Numba routine for shadow impact calculation
@numba.njit
def calculate_shadow_impact_numba(height_diff_array, sun_altitude, distance_array):
    """
    Calculate the shadow impact for arrays of buildings.
    Returns an array of shadow impact values between 0 and 1.
    """
    impacts = np.zeros_like(height_diff_array)
    tan_sun_altitude = np.tan(np.radians(sun_altitude))
    for i in range(len(height_diff_array)):
        height_diff = height_diff_array[i]
        distance = distance_array[i]
        if sun_altitude <= 0:
            impacts[i] = 1.0
            continue
        if height_diff <= 0:
            impacts[i] = 0.0
            continue
        shadow_length = height_diff / tan_sun_altitude
        if distance <= shadow_length:
            impacts[i] = 1.0 - (distance / shadow_length)
        else:
            impacts[i] = 0.0
    return impacts

def calculate_shadow_impact_vectorized(building_row, buildings_gdf, solar_pos, spatial_index):
    """
    Calculate shadow impact for a building, taking into account nearby buildings.
    Returns a single shadow factor (between 0 and 1).
    """
    try:
        MAX_SHADOW_DISTANCE = 3 * building_row['heightroof']
        bounds = building_row.geometry.bounds
        bounds = (
            bounds[0] - MAX_SHADOW_DISTANCE,
            bounds[1] - MAX_SHADOW_DISTANCE,
            bounds[2] + MAX_SHADOW_DISTANCE,
            bounds[3] + MAX_SHADOW_DISTANCE
        )
        possible_idxs = list(spatial_index.intersection(bounds))
        nearby_buildings = buildings_gdf.iloc[possible_idxs]
        nearby_buildings = nearby_buildings[nearby_buildings.index != building_row.name]
        if nearby_buildings.empty:
            return 1.0
        ref_centroid = building_row.geometry.centroid.coords[0]
        nearby_centroids = np.array([geom.centroid.coords[0] for geom in nearby_buildings.geometry])
        dx = nearby_centroids[:, 0] - ref_centroid[0]
        dy = nearby_centroids[:, 1] - ref_centroid[1]
        distances = np.sqrt(dx**2 + dy**2)
        height_diff = nearby_buildings['heightroof'].values - building_row['heightroof']
        sun_altitude = solar_pos['apparent_elevation']
        shadow_impacts = calculate_shadow_impact_numba(height_diff, sun_altitude, distances)
        total_shadow_impact = np.mean(shadow_impacts)
        shadow_factor = max(0.0, 1.0 - total_shadow_impact)
        return shadow_factor
    except Exception as e:
        logger.error(f"Error calculating shadow impact for building {building_row.name}: {str(e)}")
        return 1.0

# Global variables for worker processes
global_buildings_gdf = None
global_spatial_index = None
global_solar_position = None
global_annual_radiation = None
global_panel_density = None
global_panel_efficiency = None
global_performance_ratio = None

def init_worker(buildings_gdf, spatial_index, solar_position, annual_radiation,
                panel_density, panel_efficiency, performance_ratio):
    """
    Initialize global variables in worker processes.
    """
    global global_buildings_gdf, global_spatial_index, global_solar_position
    global global_annual_radiation, global_panel_density, global_panel_efficiency, global_performance_ratio
    global_buildings_gdf = buildings_gdf
    global_spatial_index = spatial_index
    global_solar_position = solar_position
    global_annual_radiation = annual_radiation
    global_panel_density = panel_density
    global_panel_efficiency = panel_efficiency
    global_performance_ratio = performance_ratio

def worker_process_building(args):
    """
    Worker function to compute raw solar potential for a single building.
    Returns a dictionary of computed raw values.
    """
    idx, building = args
    try:
        geometry = building.geometry
        area_ft2 = geometry.area
        area_m2 = area_ft2 * 0.092903
        # Retrieve global parameters
        buildings_gdf = global_buildings_gdf
        spatial_index = global_spatial_index
        solar_pos = global_solar_position
        annual_radiation = global_annual_radiation
        panel_density = global_panel_density
        panel_efficiency = global_panel_efficiency
        performance_ratio = global_performance_ratio
        # Compute shadow factor
        shadow_factor = calculate_shadow_impact_vectorized(building, buildings_gdf, solar_pos, spatial_index)
        solar_potential = (annual_radiation * area_m2 * panel_density *
                           panel_efficiency * performance_ratio * shadow_factor)
        return {
            'solar_pot': float(solar_potential),
            'effective_area': float(area_m2 * panel_density),
            'peak_power': float(area_m2 * panel_density * panel_efficiency),
            'shadow_factor': float(shadow_factor),
            'area_ft2': float(area_ft2),
            'area_m2': float(area_m2)
        }
    except Exception as e:
        logger.error(f"Error processing building {idx}: {str(e)}")
        return {
            'solar_pot': 0.0,
            'effective_area': 0.0,
            'peak_power': 0.0,
            'shadow_factor': 0.0,
            'area_ft2': 0.0,
            'area_m2': 0.0
        }

class SolarAnalyzer:
    """Class for computing raw solar potential for buildings."""
    def __init__(self):
        logger.info("Initializing SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()
        self.buildings_gdf = None
        self.spatial_index = None

    def _initialize_constants(self):
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060
        self.PANEL_EFFICIENCY = 0.20
        self.PERFORMANCE_RATIO = 0.75
        self.PANEL_DENSITY = 0.70
        self._solar_position = None

    def _initialize_cache(self):
        self._monthly_radiation = {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }
        self._annual_radiation = sum(self._monthly_radiation.values()) * 365 / 12

    def initialize_spatial_index(self, buildings_gdf):
        self.spatial_index = buildings_gdf.sindex
        return buildings_gdf

    def get_solar_position(self):
        if self._solar_position is None:
            import pandas as pd
            times = pd.date_range('2020-06-21 12:00:00', periods=1, freq='h', tz='UTC')
            location = pvlib.location.Location(latitude=self.NYC_LAT, longitude=self.NYC_LON)
            self._solar_position = location.get_solarposition(times).iloc[0]
        return self._solar_position

def compute_raw_solar(gdf, config):
    """
    Compute the raw solar potential for each building in gdf.
    This function integrates the solar analysis processing:
      - Ensures that 'heightroof' is numeric and fills missing values.
      - Uses a SolarAnalyzer (with shadow calculations and annual radiation) via multiprocessing.
    The resulting raw solar potential is stored in the column specified by
    config.dataset_info["Solar_Energy_Index"]["raw"] (e.g., "solar_pot").
    """
    # Ensure that 'heightroof' exists and is numeric.
    if 'heightroof' not in gdf.columns:
        gdf['heightroof'] = 0.0
    else:
        gdf['heightroof'] = pd.to_numeric(gdf['heightroof'], errors='coerce').fillna(0.0)
    
    # Create a spatial index, if needed, and prepare for multiprocessing.
    analyzer = SolarAnalyzer()
    # For our purposes, we assume that gdf already has all needed building geometries.
    buildings_projected = gdf.copy()  # assuming gdf is already in the proper CRS (e.g., EPSG:6539)
    analyzer.initialize_spatial_index(buildings_projected)
    solar_position = analyzer.get_solar_position()
    
    # Get global parameters for solar calculations from the analyzer.
    annual_radiation = analyzer._annual_radiation
    panel_density = analyzer.PANEL_DENSITY
    panel_efficiency = analyzer.PANEL_EFFICIENCY
    performance_ratio = analyzer.PERFORMANCE_RATIO
    
    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Computing raw solar potential using {num_processes} processes.")
    
    # Build list of arguments for each building
    args_list = [(idx, row) for idx, row in buildings_projected.iterrows()]
    
    with Pool(processes=num_processes, initializer=init_worker, initargs=(
        buildings_projected, analyzer.spatial_index, solar_position,
        annual_radiation, panel_density, panel_efficiency, performance_ratio
    )) as pool:
        results_list = pool.map(worker_process_building, args_list)
    
    # Convert results to DataFrame and assign to the raw column.
    results_df = pd.DataFrame(results_list, index=buildings_projected.index)
    raw_col = config.dataset_info["Solar_Energy_Index"]["raw"]
    gdf[raw_col] = results_df['solar_pot']
    # (Optionally, you could also add additional raw fields such as effective_area, peak_power, etc.)
    return gdf

def compute_solar_energy_index(gdf, config):
    """
    Compute the Solar Energy Index.
    First, calculate the raw solar potential (using processing steps from the old solar analysis)
    and store it in the raw column (e.g., "solar_pot"). Then, normalize that column (since higher
    solar potential is better) to create the index.
    """
    # Compute the raw solar potential and update gdf.
    gdf = compute_raw_solar(gdf, config)
    # Now use the standard high-is-good normalization.
    gdf = compute_index_for_factor_high(gdf, "Solar_Energy_Index", config)
    return gdf

###############################################################################
# HEAT ANALYSIS – Raw Value Computation Integration
###############################################################################
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from pathlib import Path
import multiprocessing as mp
import warnings
import os
import time

# Re-use functions from the old heat_analysis code:
def ensure_crs_vector(gdf, target_crs):
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def ensure_crs_raster(raster_path, target_crs, resolution):
    from pathlib import Path
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    # Convert the raster_path to a Path object if it isn't already one
    raster_path = Path(raster_path)
    
    with rasterio.open(raster_path) as src:
        same_crs = (src.crs is not None and src.crs.to_string() == target_crs)
        same_res = np.isclose(src.res[0], resolution, atol=0.1)
        if not same_crs or not same_res:
            print("Reprojecting raster to target CRS and resolution...")
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
            )
            profile = src.meta.copy()
            profile.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
            temp_path = raster_path.parent / f"reprojected_{raster_path.name}"
            with rasterio.open(temp_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            # Return as string if needed by the caller
            return str(temp_path)
        else:
            return str(raster_path)

def kelvin_to_fahrenheit(K):
    return (K - 273.15) * 9/5 + 32

def load_raster_distribution_f(raster_path):
    import rasterio
    print("Loading and analyzing full heat raster distribution...")
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
    data_f = kelvin_to_fahrenheit(data)
    valid = data_f.compressed()
    sorted_values = np.sort(valid)
    return sorted_values

def percentile_from_distribution(value, distribution):
    idx = np.searchsorted(distribution, value, side='right')
    percentile = (idx / len(distribution)) * 100.0
    return percentile

def extract_mean_temperature(site, raster_path):
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import box
    geom = site.geometry
    if geom is None or geom.is_empty:
        return np.nan
    centroid = geom.centroid
    BUFFER = 2000.0  # we assume this is defined in config
    xmin = centroid.x - BUFFER
    xmax = centroid.x + BUFFER
    ymin = centroid.y - BUFFER
    ymax = centroid.y + BUFFER
    bbox = box(xmin, ymin, xmax, ymax)
    with rasterio.open(raster_path) as src:
        row_start, col_start = src.index(xmin, ymax)
        row_end, col_end = src.index(xmax, ymin)
        row_start, row_end = sorted([row_start, row_end])
        col_start, col_end = sorted([col_start, col_end])
        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(row_end, src.height - 1)
        col_end = min(col_end, src.width - 1)
        if row_end < row_start or col_end < col_start:
            return np.nan
        window = Window(col_start, row_start, col_end - col_start + 1, row_end - row_start + 1)
        data = src.read(1, window=window, masked=True)
        if data.size == 0:
            return np.nan
        data_f = kelvin_to_fahrenheit(data)
        mean_temp = data_f.mean()
        return float(mean_temp)

def process_site_heat(args):
    site, raster_path = args
    return extract_mean_temperature(site, raster_path)

def compute_raw_heat(gdf, config):
    """
    Compute the raw heat exposure for each site.
    - Ensures gdf is in the correct CRS.
    - Loads the heat raster from config.HEAT_FILE.
    - For each site, computes the mean temperature (in Fahrenheit) within a BUFFER.
    - Loads the entire raster to build a cumulative distribution.
    - For each site, computes the percentile of its mean temperature and inverts it 
      (since lower temperatures are better) to produce a normalized raw heat index.
    The function stores two new columns:
      - 'heat_mean': the mean Fahrenheit temperature for the site.
      - 'heat_index': 1 - (percentile/100), a value between 0 and 1.
    """
    # Ensure gdf is in the proper CRS.
    gdf = ensure_crs_vector(gdf, config.crs)
    # Load and reproject heat raster.
    heat_raster_path = config.HEAT_FILE  # ensure this attribute is defined in your config
    heat_raster_path = ensure_crs_raster(heat_raster_path, config.crs, config.RESOLUTION)
    # Use multiprocessing to compute mean temperature for each site.
    sites_list = [(row, heat_raster_path) for idx, row in gdf.iterrows()]
    cpu_cnt = mp.cpu_count()
    with mp.Pool(cpu_cnt - 1) as pool:
        mean_temps = pool.map(process_site_heat, sites_list)
    gdf["heat_mean"] = mean_temps
    # Build cumulative distribution from the entire raster.
    distribution = load_raster_distribution_f(heat_raster_path)
    percentiles = [percentile_from_distribution(val, distribution) if np.isfinite(val) else np.nan for val in gdf["heat_mean"]]
    # Since lower heat is better, we invert the percentile:
    gdf["heat_index"] = [round(1 - (p / 100), 2) if np.isfinite(p) else np.nan for p in percentiles]
    return gdf

def compute_heat_index(gdf, config):
    """
    Compute the Heat Hazard index.
    First, calculate the raw heat exposure (raw values in 'heat_mean' and 'heat_index')
    using the integrated processing steps.
    Then, because higher heat hazard is worse, use our generic 'low-is-good' normalization.
    (In this example, since we already inverted the percentile in compute_raw_heat,
     the 'heat_index' column is our final normalized value.)
    """
    gdf = compute_raw_heat(gdf, config)
    # In our config, the data dictionary for heat might define:
    # "Heat_Hazard_Index": {"alias": "HeatHaz", "raw": "heat_mean", ... }
    # For this example, we assume that a lower heat_mean is good, and we already computed heat_index.
    # You could, if desired, run:
    # gdf = compute_index_for_factor_low(gdf, "Heat_Hazard_Index", config)
    # However, since our compute_raw_heat() already provides a percentile-based normalized value,
    # we can simply copy that over:
    info = config.dataset_info["Heat_Hazard_Index"]
    gdf[info["raw"]] = gdf["heat_mean"]
    gdf[info["alias"]] = gdf["heat_index"]
    return gdf


###############################################################################
# FLOOD ANALYSIS – Raw Value Computation Integration
###############################################################################
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings
import time

# Flood category definitions (these can also be stored in config if desired)
COAST_VALUES = {1: '500', 2: '100'}
STORM_VALUES = {1: 'Shl', 2: 'Dp', 3: 'Tid'}

def read_raster_window(raster_path, bbox, target_crs):
    """
    Read a window from the raster defined by bbox.
    Checks that the raster CRS matches target_crs.
    Returns the array and the transform.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is not None and src.crs.to_string() != target_crs:
            raise ValueError(f"Raster {raster_path} CRS ({src.crs}) does not match {target_crs}.")
        window = src.window(*bbox)
        data = src.read(1, window=window, masked=False)
        transform = src.window_transform(window)
        return data, transform

def process_site_flood(args):
    """
    Process a single site to calculate flood-related fractions.
    Uses a circular buffer (of radius specified by buffer_dist) around the site's centroid.
    Returns a dictionary with the following keys:
      - Cst_500_in, Cst_500_nr, Cst_100_in, Cst_100_nr,
      - StrmShl_in, StrmShl_nr, StrmDp_in, StrmDp_nr, StrmTid_in, StrmTid_nr.
    These represent the fraction of the site (or its neighborhood) that meets each flood threshold.
    """
    idx, site, fema_path, storm_path, buffer_dist, target_crs = args
    geom = site.geometry
    if geom is None or geom.is_empty:
        return idx, {col: 0.0 for col in [
            'Cst_500_in','Cst_500_nr','Cst_100_in','Cst_100_nr',
            'StrmShl_in','StrmShl_nr','StrmDp_in','StrmDp_nr','StrmTid_in','StrmTid_nr'
        ]}
    centroid = geom.centroid
    circle_geom = centroid.buffer(buffer_dist)
    minx, miny, maxx, maxy = circle_geom.bounds
    bbox = (minx, miny, maxx, maxy)
    
    # Read FEMA and Storm raster windows for the area.
    try:
        fema_arr, fema_transform = read_raster_window(fema_path, bbox, target_crs)
        storm_arr, storm_transform = read_raster_window(storm_path, bbox, target_crs)
    except Exception as e:
        logger.error(f"Error reading raster windows for site {idx}: {str(e)}")
        return idx, {col: 0.0 for col in [
            'Cst_500_in','Cst_500_nr','Cst_100_in','Cst_100_nr',
            'StrmShl_in','StrmShl_nr','StrmDp_in','StrmDp_nr','StrmTid_in','StrmTid_nr'
        ]}
    
    # Ensure both arrays are the same shape.
    min_height = min(fema_arr.shape[0], storm_arr.shape[0])
    min_width = min(fema_arr.shape[1], storm_arr.shape[1])
    fema_arr = fema_arr[:min_height, :min_width]
    storm_arr = storm_arr[:min_height, :min_width]
    width, height = fema_arr.shape[1], fema_arr.shape[0]
    
    # Rasterize the site polygon (for "inside" calculations)
    site_rast = features.rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=fema_transform,
        fill=0,
        dtype=np.uint8
    )
    # Rasterize the circle (for "neighborhood" calculations)
    circle_rast = features.rasterize(
        [(circle_geom, 1)],
        out_shape=(height, width),
        transform=fema_transform,
        fill=0,
        dtype=np.uint8
    )
    site_mask = (site_rast == 1)
    circle_mask = (circle_rast == 1)
    inside_count = site_mask.sum()
    circle_count = circle_mask.sum()
    
    results = {}
    # For FEMA values ("Coast")
    if inside_count == 0:
        for cval in COAST_VALUES.values():
            results[f"Cst_{cval}_in"] = 0.0
    else:
        for cval, ctag in COAST_VALUES.items():
            inside_match = ((site_mask) & (fema_arr == cval)).sum()
            results[f"Cst_{ctag}_in"] = inside_match / inside_count
    if circle_count == 0:
        for cval in COAST_VALUES.values():
            results[f"Cst_{cval}_nr"] = 0.0
    else:
        for cval, ctag in COAST_VALUES.items():
            nr_match = ((circle_mask) & (fema_arr == cval)).sum()
            results[f"Cst_{ctag}_nr"] = nr_match / circle_count
    
    # For Storm values
    if inside_count == 0:
        for sval in STORM_VALUES.values():
            results[f"Strm{sval}_in"] = 0.0
    else:
        for sval, stag in STORM_VALUES.items():
            inside_match = ((site_mask) & (storm_arr == sval)).sum()
            results[f"Strm{stag}_in"] = inside_match / inside_count
    if circle_count == 0:
        for sval in STORM_VALUES.values():
            results[f"Strm{sval}_nr"] = 0.0
    else:
        for sval, stag in STORM_VALUES.items():
            nr_match = ((circle_mask) & (storm_arr == sval)).sum()
            results[f"Strm{stag}_nr"] = nr_match / circle_count

    return idx, results

def compute_raw_flood(gdf, config):
    """
    Compute the raw flood hazard for each site.
    Uses FEMA and Storm flood rasters from config (e.g., config.FEMA_RASTER and config.STORM_RASTER),
    and a buffer distance (from config.analysis_params or default 2000 ft).
    For each site, processes the flood fractions via multiprocessing and then aggregates
    the 10 computed fraction fields by taking their mean.
    The result is stored in the column specified by config.dataset_info["Flood_Hazard_Index"]["raw"].
    """
    # Ensure gdf is in the correct CRS.
    gdf = ensure_crs_vector(gdf, config.crs)
    buffer_dist = config.analysis_params.get("analysis_buffer_ft", 2000)
    # Get flood raster paths from config (assumed to be added to your config)
    fema_raster = config.FEMA_RASTER  # e.g., defined in config as a Path
    storm_raster = config.STORM_RASTER
    # Build list of arguments for multiprocessing.
    args_list = [(idx, row, fema_raster, storm_raster, buffer_dist, config.crs)
                 for idx, row in gdf.iterrows()]
    cpu_cnt = mp.cpu_count()
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        results = list(executor.map(process_site_flood, args_list))    # Convert results (list of (idx, dict)) into a DataFrame
    results_dict = {}
    for idx, res in results:
        results_dict[idx] = res
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    # Now, aggregate the 10 flood fraction fields into one raw flood risk value.
    # For example, we take the mean of these 10 fields.
    fields = [
        'Cst_500_in','Cst_500_nr','Cst_100_in','Cst_100_nr',
        'StrmShl_in','StrmShl_nr','StrmDp_in','StrmDp_nr','StrmTid_in','StrmTid_nr'
    ]
    results_df["flood_risk"] = results_df[fields].mean(axis=1)
    # Merge the computed raw flood risk back into gdf (using the same index)
    gdf = gdf.join(results_df["flood_risk"])
    return gdf

def compute_flood_hazard_index(gdf, config):
    """
    Compute the Flood Hazard Index.
    First, calculate the raw flood hazard using processing steps from the old flood analysis.
    Then, because higher flood risk is bad, apply a low-is-good normalization.
    """
    gdf = compute_raw_flood(gdf, config)
    # In our config, assume the data dictionary for flood hazard is defined as:
    # "Flood_Hazard_Index": {"alias": "FloodHaz", "raw": "flood_risk", ... }
    # Because higher flood_risk means higher hazard (worse), we want to use the low-is-good normalization.
    gdf = compute_index_for_factor_low(gdf, "Flood_Hazard_Index", config)
    return gdf

###############################################################################
# VULNERABILITY ANALYSIS – Raw Value Computation Integration
###############################################################################
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
import warnings
import time

def compute_raw_heat_vulnerability(gdf, config):
    """
    Compute the raw heat vulnerability for each site in gdf using the HVI dataset.
    
    For each site:
      - Create a buffer (using config.analysis_params["analysis_buffer_ft"], defaulting to 2000 ft)
        around the site's centroid.
      - Compute an area‑weighted average from the HVI dataset (using the field "HVI").
    
    The raw heat vulnerability value is stored in the column defined by
    config.dataset_info["Heat_Vulnerability_Index"]["raw"] (defaulting to "hvi_area").
    
    Returns:
      The updated GeoDataFrame with raw heat vulnerability values.
    """
    # Ensure gdf is in the proper CRS.
    gdf = ensure_crs_vector(gdf, config.crs)
    buffer_dist = config.analysis_params.get("analysis_buffer_ft", 2000)
    
    # Load the HVI dataset.
    hvi = gpd.read_file(config.HVI_DATA)
    hvi = ensure_crs_vector(hvi, config.crs)
    
    # Build a spatial index for the HVI dataset.
    hvi_sindex = hvi.sindex
    hvi_values = []
    
    # Process each site.
    for idx, site in gdf.iterrows():
        geom = site.geometry
        if geom is None or geom.is_empty:
            hvi_values.append(np.nan)
            continue
        
        # Create a buffer around the site's centroid.
        centroid = geom.centroid
        buffer_geom = centroid.buffer(buffer_dist)
        
        # Limit to HVI polygons that intersect the buffer.
        possible_hvi = hvi.iloc[list(hvi_sindex.intersection(buffer_geom.bounds))]
        # Compute the area-weighted average for HVI.
        hvi_val = area_weighted_average(buffer_geom, possible_hvi, "HVI")
        hvi_values.append(hvi_val)
    
    # Determine the raw field name from config (default "hvi_area").
    raw_field = config.dataset_info["Heat_Vulnerability_Index"].get("raw", "hvi_area")
    gdf[raw_field] = hvi_values
    return gdf

def compute_heat_vulnerability_index(gdf, config):
    """
    Compute the final (normalized) Heat Vulnerability Index.
    
    This function:
      - Ensures that the raw heat vulnerability values (e.g. in "hvi_area") are computed.
      - Applies a low‑is‑good normalization (using compute_index_for_factor_low)
        to create the normalized index.
    
    The final normalized index is stored in the column defined by
    config.dataset_info["Heat_Vulnerability_Index"]["alias"] (for example, "HeatHaz").
    
    Returns:
      The updated GeoDataFrame with the normalized Heat Vulnerability Index.
    """
    # Determine the raw field name.
    raw_field = config.dataset_info["Heat_Vulnerability_Index"].get("raw", "hvi_area")
    # Compute raw values if they are not already in the GeoDataFrame.
    if raw_field not in gdf.columns:
        gdf = compute_raw_heat_vulnerability(gdf, config)
    
    # Apply the low‑is‑good normalization.
    gdf = compute_index_for_factor_high(gdf, "Heat_Vulnerability_Index", config)
    return gdf

def compute_raw_flood_vulnerability(gdf, config):
    """
    Compute the raw flood vulnerability for each site in gdf using only FVI data.
    
    For each site:
      - Create a buffer (using config.analysis_params["analysis_buffer_ft"], defaulting to 2000 ft)
        around the site's centroid.
      - Load the FVI dataset from config.FVI_DATA.
      - Build a spatial index on the FVI dataset.
      - Compute area-weighted averages for FVI fields "ss_80s" and "tid_80s" using the shared area_weighted_average().
      - Compute the overall raw flood vulnerability as the mean of the two values.
      - Store this raw value in the column defined by config.dataset_info["Flood_Vulnerability_Index"]["raw"].
    
    Returns:
      The updated GeoDataFrame with raw flood vulnerability values.
    """
    # Ensure gdf is in the target CRS.
    gdf = ensure_crs_vector(gdf, config.crs)
    buffer_dist = config.analysis_params.get("analysis_buffer_ft", 2000)
    
    # Load the FVI dataset.
    fvi = gpd.read_file(config.FVI_DATA)
    fvi = ensure_crs_vector(fvi, config.crs)
    
    # Build a spatial index for the FVI dataset.
    fvi_sindex = fvi.sindex
    
    # Lists to store the area-weighted averages.
    ss80_values = []
    tid80_values = []
    
    # Process each site.
    for idx, site in gdf.iterrows():
        geom = site.geometry
        if geom is None or geom.is_empty:
            ss80_values.append(np.nan)
            tid80_values.append(np.nan)
            continue
        
        # Create a buffer around the site's centroid.
        centroid = geom.centroid
        buffer_geom = centroid.buffer(buffer_dist)
        
        # Get FVI polygons that intersect the buffer.
        possible_fvi = fvi.iloc[list(fvi_sindex.intersection(buffer_geom.bounds))]
        
        # Compute area-weighted averages for each FVI field.
        ss80_val = area_weighted_average(buffer_geom, possible_fvi, "ss_80s")
        tid80_val = area_weighted_average(buffer_geom, possible_fvi, "tid_80s")
        ss80_values.append(ss80_val)
        tid80_values.append(tid80_val)
    
    # Store the raw values in the GeoDataFrame.
    # The overall raw flood vulnerability is taken as the mean of the two FVI values.
    gdf["ssvul_area"] = ss80_values
    gdf["tivul_area"] = tid80_values
    flood_raw_field = config.dataset_info["Flood_Vulnerability_Index"].get("raw", "flood_vuln")
    gdf[flood_raw_field] = gdf[["ssvul_area", "tivul_area"]].mean(axis=1)
    
    return gdf

def compute_flood_vulnerability_index(gdf, config):
    """
    Compute the Flood Vulnerability Index.
    It uses the raw flood vulnerability value (computed as the mean of the two FVI measures)
    and then applies a low-is-good normalization.
    """
    # Compute raw vulnerability values (if not already computed).
    if "flood_vuln" not in gdf.columns:
        gdf = compute_raw_flood_vulnerability(gdf, config)
    # Normalize using the generic low-is-good normalization.
    gdf = compute_index_for_factor_high(gdf, "Flood_Vulnerability_Index", config)
    return gdf


###############################################################################
# CENSUS ANALYSIS – Raw Population Computation via Raster Method
###############################################################################
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import warnings
from shapely.geometry import Point
from pathlib import Path
import time

def compute_raw_population(gdf, config):
    """
    Compute the total population within a specified buffer for each feature in gdf using a raster-based approach.
    
    Steps:
      1. Load a pre-generated census blocks dataset with population data from a file
         (assumed to be at config.input_dir/nyc_blocks_with_pop.geojson).
      2. Ensure that the blocks are in the target CRS (config.crs) and filter them to NYC counties
         using config.nyc_counties.
      3. Determine the population field (first preference is "P1_001N") and compute each block's area.
      4. Compute a population density (people per square foot) for each block.
      5. Rasterize the blocks using the computed population density.
      6. For each feature in gdf, create a buffer (using config.analysis_params["analysis_buffer_ft"]).
      7. Use zonal statistics over the raster to estimate the total population in the buffer
         (by summing density values over pixels and multiplying by pixel area).
    
    The raw total population is stored in the column defined by config.dataset_info["Service_Population_Index"]["raw"].
    """
    from rasterstats import zonal_stats

    # Ensure input gdf is in the proper CRS.
    gdf = ensure_crs_vector(gdf, config.crs)
    buffer_dist = config.analysis_params.get("analysis_buffer_ft", 2000)

    # Load the pre-generated census blocks with population data.
    blocks_file = os.path.join(config.input_dir, "nyc_blocks_with_pop.geojson")
    if not os.path.exists(blocks_file):
        raise FileNotFoundError(f"Blocks file with population data not found at: {blocks_file}")
    logger.info("Loading census blocks with population data...")
    blocks = gpd.read_file(blocks_file)
    blocks = ensure_crs_vector(blocks, config.crs)
    # Filter to NYC counties (using config.nyc_counties)
    blocks = blocks[blocks['GEOID20'].str[2:5].isin(config.nyc_counties)]
    
    # Determine the population field.
    candidate_fields = ["P1_001N", "population", "pop", "Population", "POPULATION"]
    pop_field = None
    for field in candidate_fields:
        if field in blocks.columns:
            pop_field = field
            break
    if pop_field is None:
        raise KeyError("No population field found in blocks dataset. Available columns: " + str(blocks.columns))
    logger.info(f"Using population field: '{pop_field}'")
    
    # Compute block area (in square feet) and population density (people per square foot)
    blocks = blocks.copy()
    blocks["block_area"] = blocks.geometry.area
    blocks["pop_density"] = blocks.apply(
        lambda row: row[pop_field] / row["block_area"] if row["block_area"] > 0 else 0, axis=1
    )
    
    # Define raster bounds based on the blocks dataset.
    xmin, ymin, xmax, ymax = blocks.total_bounds
    resolution = config.RESOLUTION
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = from_origin(xmin, ymax, resolution, resolution)
    
    # Rasterize the blocks using the computed population density.
    shapes = ((geom, density) for geom, density in zip(blocks.geometry, blocks["pop_density"]))
    population_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32,
        all_touched=True
    )
    pixel_area = resolution * resolution  # in square feet

    # For each feature in gdf, create a buffer and compute zonal statistics.
    pop_estimates = []
    for idx, row in gdf.iterrows():
        buffer_geom = row.geometry.buffer(buffer_dist)
        stats = zonal_stats(
            [buffer_geom],
            population_raster,
            affine=transform,
            stats=['sum'],
            nodata=0,
            all_touched=True
        )
        # Estimated total population = (sum of population density over pixels) * (pixel area)
        estimated_population = stats[0]['sum'] * pixel_area if stats[0]['sum'] is not None else 0.0
        pop_estimates.append(estimated_population)
    
    # Store the raw total population in the designated column.
    raw_field = config.dataset_info["Service_Population_Index"]["raw"]
    gdf[raw_field] = pop_estimates
    logger.info("Raw population estimates computed.")
    return gdf

def compute_service_population_index(gdf, config):
    """
    Compute the Service Population Index.
    First, compute the raw total population within the buffer using the raster-based method,
    then apply high-is-good normalization.
    """
    gdf = compute_raw_population(gdf, config)
    return compute_index_for_factor_high(gdf, "Service_Population_Index", config)

def compute_all_indices(gdf, config, weight_scenario=None):
    """
    Compute all individual indices and then an overall Suitability_Index.
    
    For each factor defined in config.dataset_info, this function:
      - Uses the mapping 'index_directions' to determine whether a high raw value is good ("high")
        or bad ("low").
      - Calls either compute_index_for_factor_high() or compute_index_for_factor_low() accordingly.
      
    Finally, the overall index is computed as a weighted sum using the weights from the chosen
    weight scenario, and a normalized index ('index_norm') is added.
    """
    # Mapping: for each factor, indicate if a higher raw value is "good" or "bad".
    index_directions = {
         "Adaptability_Index": "high",
         "Solar_Energy_Index": "high",
         "Heat_Hazard_Index": "low",
         "Flood_Hazard_Index": "low",
         "Heat_Vulnerability_Index": "low",
         "Flood_Vulnerability_Index": "low",
         "Service_Population_Index": "high"
    }
    
    # Loop over each factor from the data dictionary in the config.
    for factor in config.dataset_info.keys():
        # Skip any factors not specified in the mapping.
        if factor not in index_directions:
            continue
        if index_directions[factor] == "high":
            gdf = compute_index_for_factor_high(gdf, factor, config)
        else:
            gdf = compute_index_for_factor_low(gdf, factor, config)
    
    # Use default weight scenario if none is provided.
    if weight_scenario is None:
        weight_scenario = list(config.weight_scenarios.values())[0]
    
    # Compute the overall Suitability Index as a weighted sum.
    overall = 0.0
    for factor, info in config.dataset_info.items():
        weight_key = f"{factor}_weight"
        overall += gdf[info["alias"]] * weight_scenario.get(weight_key, 0)
    
    gdf["Suitability_Index"] = overall
    gdf["index_norm"] = min_max_normalize(gdf["Suitability_Index"])
    return gdf


def compute_all_indices_ordered(gdf, config):
    """
    Compute each analysis (Adaptability, Solar, Heat, Flood, Vulnerability, and Service Population)
    in the order defined. After each factor is processed, print summary statistics for both the
    raw value and the normalized index.
    """
    # 1. Adaptability Analysis
    print("\n--- Running Adaptability Analysis ---")
    gdf = compute_adaptability_index(gdf, config)
    raw_field = config.dataset_info["Adaptability_Index"]["raw"]      # e.g., "bldg_adapt"
    idx_field = config.dataset_info["Adaptability_Index"]["alias"]      # e.g., "Adaptability"
    print(f"Adaptability raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Adaptability index ({idx_field}):\n", gdf[idx_field].describe())

    # 2. Solar Analysis
    print("\n--- Running Solar Energy Analysis ---")
    gdf = compute_solar_energy_index(gdf, config)
    raw_field = config.dataset_info["Solar_Energy_Index"]["raw"]        # e.g., "solar_pot"
    idx_field = config.dataset_info["Solar_Energy_Index"]["alias"]        # e.g., "Solar"
    print(f"Solar raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Solar index ({idx_field}):\n", gdf[idx_field].describe())

    # 3. Heat Analysis
    print("\n--- Running Heat Hazard Analysis ---")
    gdf = compute_heat_index(gdf, config)
    raw_field = config.dataset_info["Heat_Hazard_Index"]["raw"]           # e.g., "heat_mean"
    idx_field = config.dataset_info["Heat_Hazard_Index"]["alias"]           # e.g., "HeatHaz"
    print(f"Heat raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Heat index ({idx_field}):\n", gdf[idx_field].describe())

    # 4. Flood Analysis
    print("\n--- Running Flood Hazard Analysis ---")
    gdf = compute_flood_hazard_index(gdf, config)
    raw_field = config.dataset_info["Flood_Hazard_Index"]["raw"]          # e.g., "flood_risk"
    idx_field = config.dataset_info["Flood_Hazard_Index"]["alias"]          # e.g., "FloodHaz"
    print(f"Flood raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Flood index ({idx_field}):\n", gdf[idx_field].describe())

    # 5. Vulnerability Analysis
    #    (Heat Vulnerability)
    print("\n--- Running Heat Vulnerability Analysis ---")
    gdf = compute_heat_vulnerability_index(gdf, config)
    # For heat vulnerability, the raw field may have been computed earlier (e.g. in 'hvi_area')
    raw_field = config.dataset_info["Heat_Vulnerability_Index"].get("raw", "hvi_area")
    idx_field = config.dataset_info["Heat_Vulnerability_Index"]["alias"]
    print(f"Heat Vulnerability raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Heat Vulnerability index ({idx_field}):\n", gdf[idx_field].describe())

    #    (Flood Vulnerability)
    print("\n--- Running Flood Vulnerability Analysis ---")
    gdf = compute_flood_vulnerability_index(gdf, config)
    raw_field = config.dataset_info["Flood_Vulnerability_Index"].get("raw", "flood_vuln")
    idx_field = config.dataset_info["Flood_Vulnerability_Index"]["alias"]
    print(f"Flood Vulnerability raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Flood Vulnerability index ({idx_field}):\n", gdf[idx_field].describe())

    # 6. Census / Service Population Analysis
    print("\n--- Running Service Population Analysis ---")
    gdf = compute_service_population_index(gdf, config)
    raw_field = config.dataset_info["Service_Population_Index"]["raw"]
    idx_field = config.dataset_info["Service_Population_Index"]["alias"]
    print(f"Service Population raw values ({raw_field}):\n", gdf[raw_field].describe())
    print(f"Service Population index ({idx_field}):\n", gdf[idx_field].describe())

    # Finally, compute an overall Suitability Index as a weighted sum.
    print("\n--- Computing Overall Suitability Index ---")
    gdf = compute_all_indices(gdf, config)
    print("Overall Suitability Index:\n", gdf["Suitability_Index"].describe())
    print("Normalized Overall Suitability Index:\n", gdf["index_norm"].describe())

    return gdf