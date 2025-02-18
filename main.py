import os
import geopandas as gpd

def main():
    from config import ResilienceHubConfig
    from analysis_modules import compute_all_indices_ordered
    import webmap

    # Instantiate the configuration.
    config = ResilienceHubConfig()

    # Define file paths.
    indices_file = os.path.join(config.output_dir, "sites_with_indices.geojson")
    preprocessed_sites_file = os.path.join(config.output_dir, "preprocessed_sites_RH.geojson")
    
    # Check if the preprocessed sites file exists; if not, run the preprocessing pipeline.
    if not os.path.exists(preprocessed_sites_file):
        print(f"{preprocessed_sites_file} not found. Running full preprocessing pipeline.")
        from data_preprocessing import full_preprocessing_pipeline
        full_preprocessing_pipeline()
        # Check again if the file now exists.
        if not os.path.exists(preprocessed_sites_file):
            print("Error: Preprocessing did not produce the expected file. Exiting.")
            return
        else:
            print("Preprocessing complete.")

    # Now check for the indices file.
    if os.path.exists(indices_file):
        print("Found precomputed indices file: sites_with_indices.geojson")
        # Load the GeoDataFrame with precomputed indices.
        gdf = gpd.read_file(indices_file)

        # Recalculate the overall Priority_Index using the precomputed index values.
        # The calculation uses the precomputed values (using each index's alias)
        # multiplied by the corresponding weight from the default weight scenario.
        default_weights = config.weight_scenarios["Default"]
        priority_series = None

        for index_name, index_info in config.dataset_info.items():
            weight_key = f"{index_name}_weight"
            if weight_key in default_weights:
                alias = index_info["alias"]  # the column name for the computed value
                if alias in gdf.columns:
                    if priority_series is None:
                        priority_series = default_weights[weight_key] * gdf[alias]
                    else:
                        priority_series += default_weights[weight_key] * gdf[alias]
                else:
                    print(f"Warning: Column '{alias}' not found in the GeoDataFrame.")
        if priority_series is not None:
            gdf["Priority_Index"] = priority_series
        else:
            print("Warning: No indices found to compute Priority_Index.")

        # Optionally, save the updated GeoDataFrame.
        gdf.to_file(indices_file, driver="GeoJSON")

    else:
        print("sites_with_indices.geojson not found. Running full indices computation pipeline.")
        # Load the preprocessed sites file.
        gdf = gpd.read_file(preprocessed_sites_file)

        gdf = compute_all_indices_ordered(gdf, config)

        # Save the computed indices to the output file.
        gdf.to_file(indices_file, driver="GeoJSON")

    # Build the webmap using the (now ensured) indices file.
    scenario_geojsons = {"ResilienceHub": indices_file}
    webmap_path = webmap.build_webmap(scenario_geojsons, config)
    print("Webmap created at:", webmap_path)

if __name__ == "__main__":
    main()