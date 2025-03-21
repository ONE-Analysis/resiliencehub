# main.py
import os
import geopandas as gpd
from datetime import datetime

def main():
    from config import ResilienceHubConfig
    from analysis_modules import compute_all_raw_values, compute_all_indices_from_raw
    import webmap

    # Instantiate configuration.
    config = ResilienceHubConfig()

    # First, ensure that the preprocessed sites exist.
    preprocessed_file = os.path.join(config.output_dir, "preprocessed_sites_RH.geojson")
    if not os.path.exists(preprocessed_file):
        print(f"{preprocessed_file} not found. Running full preprocessing pipeline.")
        from data_preprocessing import full_preprocessing_pipeline
        full_preprocessing_pipeline()
        if not os.path.exists(preprocessed_file):
            print("Error: Preprocessing did not produce the expected file. Exiting.")
            return
        else:
            print("Preprocessing complete.")

    # Load the preprocessed sites.
    gdf = gpd.read_file(preprocessed_file)

    # Now check for a file with the raw values.
    raw_file = os.path.join(config.output_dir, "raw_values_RH.geojson")
    if not os.path.exists(raw_file):
        print(f"Raw values file {raw_file} not found. Computing raw values...")
        gdf_raw = compute_all_raw_values(gdf, config)
        gdf_raw.to_file(raw_file, driver="GeoJSON")
        print("Raw values computed and saved.")
    else:
        print("Using precomputed raw values.")
        gdf_raw = gpd.read_file(raw_file)

    # Get current date (YYMMDD) for naming outputs.
    today_str = datetime.now().strftime("%y%m%d")

    # Loop over each scenario defined in the config.
    # (Note: your new weight scenarios now include separate weights for coastal and stormwater flood hazards.)
    for scenario_name, weight_scenario in config.weight_scenarios.items():
        print(f"\nProcessing scenario: {scenario_name}")
        # Work on a copy of the raw values dataset.
        scenario_gdf = gdf_raw.copy()

        # Compute indices from the raw values using the scenario's weights.
        scenario_gdf = compute_all_indices_from_raw(scenario_gdf, config, weight_scenario=weight_scenario)
        
        # Save the scenario-specific GeoJSON.
        geojson_filename = f"{today_str}_RH_{scenario_name}.geojson"
        geojson_path = os.path.join(config.output_dir, geojson_filename)
        scenario_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"Saved scenario GeoJSON: {geojson_path}")

        # Build the webmap.
        scenario_geojsons = {scenario_name: geojson_path}
        webmap_path = webmap.build_webmap(scenario_geojsons, config)

        # Rename the webmap HTML file to include the date and scenario name.
        new_webmap_filename = f"{today_str}_RH_{scenario_name}.html"
        new_webmap_path = os.path.join(config.output_dir, new_webmap_filename)
        os.rename(webmap_path, new_webmap_path)
        print(f"Saved webmap for scenario {scenario_name}: {new_webmap_path}")

if __name__ == "__main__":
    main()