# data_preprocessing.py
import os
import geopandas as gpd
import rasterio
from pathlib import Path
import warnings
import multiprocessing as mp
import time
from datetime import timedelta
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Import our configuration
from config import ResilienceHubConfig
config = ResilienceHubConfig()

# Use config attributes
INPUT_FOLDER = Path(config.input_dir)
TARGET_CRS = config.crs
MIN_PARKING_AREA = config.min_parking_area
data_files = config.data_files

# Lists from config:
SCHOOL_TYPES = config.school_types
PRIORITY2_TYPES = config.priority2_types
FACILITIES_REQUIRED_COLUMNS = config.facilities_required_columns
LOT_FIELDS = config.lot_fields
NSI_FIELDS = config.nsi_fields

def print_dataset_info(name, data):
    print(f"\n=== {name} Dataset ===")
    if isinstance(data, gpd.GeoDataFrame):
        print("\nColumns:")
        for col in data.columns:
            print(f"- {col}")
        print("\nCRS Information:")
        print(data.crs)
        print(f"Number of features: {len(data)}")
    elif isinstance(data, rasterio.DatasetReader):
        print("\nRaster Summary:")
        print(f"Width: {data.width}")
        print(f"Height: {data.height}")
        print(f"Bands: {data.count}")
        print(f"Bounds: {data.bounds}")
        print("\nCRS Information:")
        print(data.crs)
    else:
        print("No data loaded or invalid data format.")

def load_geojson(file_path, columns=None):
    try:
        gdf = gpd.read_file(file_path, columns=columns, engine='pyogrio')
        if gdf.crs is None:
            print(f"Warning: {Path(file_path).name} has no CRS defined. Setting to {TARGET_CRS}")
            gdf.set_crs(TARGET_CRS, inplace=True)
        elif gdf.crs.to_string() != TARGET_CRS:
            print(f"Reprojecting {Path(file_path).name} from {gdf.crs} to {TARGET_CRS}")
            gdf = gdf.to_crs(TARGET_CRS)
        return gdf
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_raster(file_path):
    try:
        with rasterio.open(file_path) as src:
            if src.crs.to_string() != TARGET_CRS:
                print(f"Reprojecting {Path(file_path).name} from {src.crs} to {TARGET_CRS}")
                transform, width, height = calculate_default_transform(src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
                temp_path = Path(file_path).parent / f"temp_{Path(file_path).name}"
                kwargs = src.meta.copy()
                kwargs.update({'crs': TARGET_CRS, 'transform': transform, 'width': width, 'height': height})
                with rasterio.open(temp_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.bilinear
                        )
                return rasterio.open(temp_path)
            else:
                return rasterio.open(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clean_and_validate_geometry(gdf):
    gdf.geometry = gdf.geometry.make_valid().buffer(0.01).buffer(-0.01)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    return gdf

def explode_multipart_features(gdf):
    exploded = gdf.explode(index_parts=True).reset_index(drop=True)
    return exploded

def safe_dissolve(gdf, dissolve_field):
    try:
        groups = gdf.groupby(dissolve_field)
        dissolved_parts = []
        for name, group in groups:
            group = clean_and_validate_geometry(group)
            if len(group) > 0:
                unified = group.geometry.unary_union
                dissolved_part = gpd.GeoDataFrame({dissolve_field: [name]}, geometry=[unified], crs=group.crs)
                dissolved_part = explode_multipart_features(dissolved_part)
                dissolved_parts.append(dissolved_part)
        if dissolved_parts:
            result = pd.concat(dissolved_parts, ignore_index=True)
            return clean_and_validate_geometry(result)
        return gdf
    except Exception as e:
        print(f"Error during dissolve operation: {e}")
        return gdf

def process_facilities(gdf):
    print("\nProcessing Facilities...")
    gdf['RH_Priority'] = None
    priority1_mask = (gdf['FACTYPE'] == 'PUBLIC LIBRARY') | (gdf['FACTYPE'].str.startswith('NYCHA COMMUNITY CENTER', na=False))
    gdf.loc[priority1_mask, 'RH_Priority'] = 1
    gdf.loc[gdf['FACTYPE'].isin(PRIORITY2_TYPES), 'RH_Priority'] = 2
    gdf = gdf[gdf['RH_Priority'].notna()].copy()
    gdf['name'] = gdf['FACNAME'].str.title()
    conditions = [
        gdf['FACTYPE'].isin(SCHOOL_TYPES),
        gdf['FACTYPE'].str.contains('COMMUNITY SERVICES', na=False),
        gdf['FACTYPE'] == 'PUBLIC LIBRARY',
        gdf['FACTYPE'] == 'NYCHA COMMUNITY CENTER'
    ]
    choices = ['School', 'Community Services', 'Public Library', 'NYCHA Community Center']
    gdf['fclass'] = np.select(conditions, choices, default=gdf['FACTYPE'].str.title())
    gdf = gdf[FACILITIES_REQUIRED_COLUMNS]
    print("\nFacility counts by fclass:")
    print(gdf['fclass'].value_counts())
    print(f"Total facilities: {len(gdf)}")
    return gdf

def process_pofw(gdf):
    print("\nProcessing Places of Worship...")
    if 'fclass' not in gdf.columns:
        gdf['fclass'] = 'pofw'
    gdf['fclass'] = 'pofw_' + gdf['fclass'].astype(str)
    if 'name' not in gdf.columns:
        gdf['name'] = 'Unknown'
    for col in ['FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE']:
        gdf[col] = 'Unknown'
    gdf['CAPACITY'] = 0
    gdf['RH_Priority'] = 1
    gdf['name'] = gdf['name'].fillna('Unknown')
    keep_cols = ['geometry', 'fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                 'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority']
    gdf = gdf[keep_cols]
    print(f"POFW points: {len(gdf)}")
    return gdf

def merge_point_datasets(datasets):
    print("\nMerging Point Datasets...")
    required_columns = ['fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                        'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 
                        'RH_Priority', 'geometry']
    points_dfs = []
    for key in ['pofw', 'facilities']:
        if key in datasets and datasets[key] is not None:
            df = datasets[key]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: {key} is missing {missing_cols}")
                continue
            points_dfs.append(df[required_columns])
    if not points_dfs:
        print("No valid point datasets found!")
        return gpd.GeoDataFrame(columns=required_columns, crs=TARGET_CRS)
    bldg_pts = pd.concat(points_dfs, ignore_index=True)
    bldg_pts['ObjectID'] = bldg_pts.index + 1
    print("Merged dataset columns:", bldg_pts.columns.tolist())
    print("Counts by fclass:")
    print(bldg_pts['fclass'].value_counts())
    print(f"Total points: {len(bldg_pts)}")
    return gpd.GeoDataFrame(bldg_pts, geometry='geometry', crs=TARGET_CRS)

def full_preprocessing_pipeline():
    """
    Run the full preprocessing pipeline:
      - Load datasets specified in config.data_files
      - Process facilities and places of worship (POFW)
      - Merge point datasets to create an intermediate 'sites.geojson' in the input folder
      - Then proceed with joining with lots, merging with buildings and NSI data,
        and export the final preprocessed sites as a GeoJSON file (shapefile export is omitted)
    """
    start_time = time.time()  # Initialize start_time here!
    
    datasets = {}
    temp_files = []
    for key, file_info in data_files.items():
        file_path = file_info['path']
        columns = file_info['columns']
        print(f"\nLoading {Path(file_path).name}...")
        if str(file_path).endswith('.geojson'):
            datasets[key] = load_geojson(file_path, columns)
        elif str(file_path).endswith('.tif'):
            datasets[key] = load_raster(file_path)
            if datasets[key] is not None and datasets[key].name != str(file_path):
                temp_files.append(Path(datasets[key].name))
        if datasets[key] is not None:
            print(f"{key} loaded with {len(datasets[key]) if hasattr(datasets[key], '__len__') else 'N/A'} features.")
        else:
            print(f"{key} could not be loaded.")
    
    if 'facilities' in datasets and datasets['facilities'] is not None:
        datasets['facilities'] = process_facilities(datasets['facilities'])
    if 'pofw' in datasets and datasets['pofw'] is not None:
        datasets['pofw'] = process_pofw(datasets['pofw'])
    
    # -------------------------------
    # Merge point datasets into a single "sites" dataset.
    # -------------------------------
    sites_geojson = INPUT_FOLDER / "sites.geojson"
    if not sites_geojson.exists():
        print("\nMerging point datasets into 'sites.geojson'...")
        sites = merge_point_datasets(datasets)
        sites.to_file(sites_geojson, driver="GeoJSON")
        print(f"'sites.geojson' created in {INPUT_FOLDER}")
    else:
        print(f"\nUsing precomputed 'sites.geojson' from {INPUT_FOLDER}")
        sites = gpd.read_file(sites_geojson)
    
    # -------------------------------
    # Continue with the remaining preprocessing steps.
    # -------------------------------
    print("\nExtracting data from lots...")
    if 'lots' in datasets and datasets['lots'] is not None and len(sites) > 0:
        sites = gpd.sjoin(sites, datasets['lots'][LOT_FIELDS + ['geometry']], how='left', predicate='within')
        if 'index_right' in sites.columns:
            sites.drop(columns=['index_right'], inplace=True)
        matched = sites[~sites['Address'].isna()].shape[0]
        unmatched = sites[sites['Address'].isna()].shape[0]
        print(f"Lot data matched: {matched}; unmatched: {unmatched}")
    
    print("\nMerging point data with buildings...")
    if 'buildings' in datasets and datasets['buildings'] is not None and len(sites) > 0:
        buildings = datasets['buildings'][['geometry', 'groundelev', 'heightroof', 'lststatype', 'cnstrct_yr']]
        joined = gpd.sjoin(buildings, sites, how='inner', predicate='contains')
        print("Joined columns:", joined.columns)
        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan
        agg_fields = config.agg_fields
        grouped = joined.groupby(joined.index)
        agg_dict = {f: combine_values for f in agg_fields}
        if 'CAPACITY' in joined.columns:
            agg_dict['CAPACITY'] = 'sum'
        if 'RH_Priority' in joined.columns:
            agg_dict['RH_Priority'] = 'min'
        agg_result = grouped.agg(agg_dict)
        buildings = buildings.loc[agg_result.index]
        for f in agg_fields:
            buildings[f] = agg_result[f]
        datasets['buildings'] = buildings
    else:
        print("No overlapping buildings and points found.")
    
    print("\nExtracting NSI data...")
    if 'nsi' in datasets and datasets['nsi'] is not None and 'buildings' in datasets and datasets['buildings'] is not None:
        nsi = datasets['nsi']
        buildings = datasets['buildings']
        joined_nsi = gpd.sjoin(buildings, nsi[NSI_FIELDS + ['geometry']], how='left', predicate='contains')
        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan
        agg_dict_nsi = {f: combine_values for f in NSI_FIELDS}
        grouped_nsi = joined_nsi.groupby(joined_nsi.index).agg(agg_dict_nsi)
        for f in NSI_FIELDS:
            buildings[f] = grouped_nsi[f]
        datasets['buildings'] = buildings
    
    # Export final preprocessed sites as a GeoJSON (omitting shapefile export).
    geojson_output = Path(config.output_dir) / "preprocessed_sites_RH.geojson"
    datasets['buildings'].to_file(geojson_output, driver="GeoJSON")
    print(f"\nPreprocessing complete. Final sites saved to {geojson_output}")
    
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")
    
    total_duration = time.time() - start_time
    print(f"\nTotal processing time: {timedelta(seconds=total_duration)}")

if __name__ == "__main__":
    full_preprocessing_pipeline()