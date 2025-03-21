"""
webmap.py

This module builds an interactive HTML webmap for the Resilience Hub Prioritization Analysis.
It reads an exported GeoJSON file (with computed indices) and creates a folium map with a true tooltip
that displays the non-zero index values for each analysis factor.
"""

import os
import folium
import geopandas as gpd

def create_tooltip_content(properties, dataset_info, weight_scenario):
    """
    Build an HTML snippet for a true tooltip that displays:
      - The site identifier,
      - The overall Suitability Index,
      - For each analysis factor whose corresponding weight is nonzero,
        display its raw value (with any prefix/suffix) and the normalized index.
    
    Parameters:
      properties: the feature's properties
      dataset_info: a dictionary from config containing display settings for each factor
      weight_scenario: a dict (e.g., config.weight_scenarios[scenario_name]) specifying the weights
                       for each factor (keys are like "Adaptability_Index_weight", etc.)
    """
    html = "<div style='font-family: Verdana; font-size: 12px; border-radius: 8px; padding: 12px;'>"
    
    # Header with site identifier.
    site_id = properties.get('name', properties.get('ID', properties.get('OBJECTID', 'Site')))
    html += f"<h4 style='margin:0 0 4px 0;'>Site: {site_id}</h4>"
    
    # Overall Suitability Index.
    overall = properties.get('Suitability_Index')
    if overall is not None:
        try:
            overall_disp = f"{float(overall):.3f}"
        except Exception:
            overall_disp = overall
        html += f"<p style='margin:2px 0;'><strong>Overall Suitability:</strong> {overall_disp}</p>"
    
    # List all factors for which the weight is nonzero.
    html += "<ul style='list-style-type: none; padding-left:0; margin:0;'>"
    for factor, info in dataset_info.items():
        weight_key = f"{factor}_weight"
        if weight_scenario.get(weight_key, 0) == 0:
            continue  # Skip factors with zero weight.
        alias = info.get('alias')    # normalized index field
        raw_field = info.get('raw')   # raw value field
        if alias not in properties or raw_field not in properties:
            continue
        try:
            raw_val = float(properties[raw_field])
            norm_val = float(properties[alias])
            raw_disp = f"{raw_val:.2f}"
            norm_disp = f"{norm_val:.3f}"
        except Exception:
            raw_disp = properties.get(raw_field, "N/A")
            norm_disp = properties.get(alias, "N/A")
        colored_name = (
            f"<span style='color: {info.get('hex')}; font-weight:bold;'>"
            f"{info.get('name')}</span>"
        )
        html += (
            f"<li style='margin:2px 0;'>{colored_name}: {info.get('prefix','')}{raw_disp}{info.get('suffix','')}"
            f"<br><span style='font-size:10px;'>Index: {norm_disp}/1</span></li>"
        )
    html += "</ul></div>"
    return html

def style_function(feature):
    """
    A basic style function for the GeoJSON layer.
    """
    return {
         "fillColor": "#228B22",  # forest green
         "color": "#228B22",
         "weight": 2,
         "fillOpacity": 0.6
    }

def clean_geojson_properties(geojson_data):
    """
    Remove any property that is a function from all features in the geojson_data.
    """
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        keys_to_remove = [k for k, v in props.items() if callable(v)]
        for key in keys_to_remove:
            del props[key]
    return geojson_data

def interpolate_color(color1, color2, t):
    """
    Linearly interpolate between two hex colors.
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return rgb_to_hex((r, g, b))

def get_style_function(min_val, max_val):
    """
    Returns a style function that interpolates fillColor based on the 'index_norm' property.
    """
    def style_function(feature):
        try:
            index_norm = float(feature['properties'].get('index_norm', 0))
        except Exception:
            index_norm = 0
        if max_val != min_val:
            normalized = (index_norm - min_val) / (max_val - min_val)
        else:
            normalized = 0
        normalized = max(0, min(normalized, 1))
        fill_color = interpolate_color("#f4fa7d", "#00732a", normalized)
        return {
            "fillColor": fill_color,
            "color": fill_color,
            "weight": 2,
            "fillOpacity": 0.6
        }
    return style_function

def build_webmap(scenario_geojsons, config, neighborhood_name=None):
    """
    Build an interactive folium webmap for the Resilience Hub analysis by adding GeoJSON features
    with true tooltips, along with custom title, logo, legend, and donut chart overlay.
    
    Parameters:
      - scenario_geojsons: dict mapping a scenario name to the path of the exported GeoJSON.
      - config: an instance of ResilienceHubConfig.
      - neighborhood_name: (optional) appended to the output filename.
        
    Returns:
      The file path of the saved HTML map.
    """
    # Assume only one scenario is passed.
    scenario_name = list(scenario_geojsons.keys())[0]
    print(f"Building Resilience Hub webmap for scenario: {scenario_name}")
    
    # Get the current weight scenario from the config.
    current_weight_scenario = config.weight_scenarios.get(scenario_name, {})
    
    geojson_path = scenario_geojsons[scenario_name]
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf.set_crs(config.crs, inplace=True)
    gdf = gdf.to_crs("EPSG:4326")
    
    min_index_norm = gdf["index_norm"].min()
    max_index_norm = gdf["index_norm"].max()
    style_func = get_style_function(min_index_norm, max_index_norm)
    
    geojson_data = gdf.__geo_interface__
    geojson_data = clean_geojson_properties(geojson_data)
    
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=12,
                   tiles="CartoDB Positron")
    
    # ---------------------------------------------------------------------
    # Add meta viewport for mobile responsiveness
    meta_viewport = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    m.get_root().html.add_child(folium.Element(meta_viewport))
    
    # ---------------------------
    # Add responsive CSS for fixed overlays
    # The mobile media query now hides the donut chart and top roads lists, and makes the legend smaller.
    responsive_css = """
    <style>
    /* Base styles for overlays */
    .resilience-title, .analysis-text, .legend-box, .donut-overlay {
        position: fixed;
        background-color: white;
        border: 2px solid grey;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-family: Helvetica, sans-serif;
        z-index: 1000;
        padding: 10px;
    }
    .resilience-title {
        top: 20px;
        left: 20px;
        font-size: 30px;
        font-weight: bold;
    }
    .analysis-text {
        bottom: 20px;
        left: 20px;
        font-size: 30px;
    }
    .legend-box {
        bottom: 20px;
        right: 20px;
        padding: 12px;
        font-size: 12px;
        width: 200px;
    }
    .donut-overlay {
        top: 100px;
        left: 20px;
        font-size: 25px;
        text-align: center;
    }

    /* Mobile adjustments: hide donut chart and make legend smaller */
    @media (max-width: 600px) {
        .resilience-title {
            top: 5px;
            left: 5px;
            font-size: 18px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
        }
        .analysis-text {
            bottom: 5px;
            left: 5px;
            font-size: 12px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
        }
        .legend-box {
            top: 60px;
            left: 5px;
            font-size: 9px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
            width: 150px;
            height: 270px;
        }
        .donut-overlay {
            display: none !important;
        }
    }

    /* Tooltip CSS for donut legend */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    .tooltip {
        visibility: hidden;
        position: absolute;
        left: 25px;
        background-color: white;
        color: #333;
        padding: 10px;
        border-radius: 20px;
        font-size: 8pt;
        width: 200px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #ddd;
        z-index: 1001;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip-container:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .legend-item {
        position: relative;
        white-space: nowrap;
    }
    .info-icon {
        color: inherit;
        font-weight: bold;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(responsive_css))

    
    # ---------------------------------------------------------------------
    # Add tooltip custom CSS (kept as in your original code)
    tooltip_custom_css = """
    <style>
    .leaflet-tooltip {
        border-radius: 8px !important;
        padding: 4px 8px !important;
        font-size: 12px !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(tooltip_custom_css))
    
    # ---------------------------------------------------------------------
    # Add Data Layers
    # Add FOZ layer
    foz_path = os.path.join(config.input_dir, 'FOZ_NYC_Merged.geojson')
    if os.path.exists(foz_path):
        try:
            foz_gdf = gpd.read_file(foz_path)
            if foz_gdf.crs is None or foz_gdf.crs != "EPSG:4326":
                foz_gdf = foz_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                foz_gdf,
                name="Federal Opportunity Zones",
                style_function=lambda x: {
                    'fillColor': 'MediumOrchid',
                    'color': 'MediumOrchid',
                    'weight': 0.5,
                    'fillOpacity': 0.07,
                    'opacity': 1.0
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process FOZ file: {str(e)}")

    # Add Persistent Poverty layer
    poverty_path = os.path.join(config.input_dir, 'nyc_persistent_poverty.geojson')
    if os.path.exists(poverty_path):
        try:
            poverty_gdf = gpd.read_file(poverty_path)
            if poverty_gdf.crs is None or poverty_gdf.crs != "EPSG:4326":
                poverty_gdf = poverty_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                poverty_gdf,
                name="Persistent Poverty Areas",
                style_function=lambda x: {
                    'fillColor': 'SandyBrown',
                    'color': 'SandyBrown',
                    'weight': 0.5,
                    'fillOpacity': 0.07,
                    'opacity': 1.0
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process persistent poverty file: {str(e)}")

    # Add neighborhoods layer
    csc_path = os.path.join(config.input_dir, 'CSC_Neighborhoods.geojson')
    if os.path.exists(csc_path):
        try:
            csc_gdf = gpd.read_file(csc_path)
            if csc_gdf.crs is None or csc_gdf.crs != "EPSG:4326":
                csc_gdf = csc_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                csc_gdf,
                name="CSC_Neighborhoods",
                style_function=lambda x: {
                    "color": "gray",
                    "weight": 1,
                    "fillOpacity": 0.07
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process neighborhoods file: {str(e)}")
    
    # ---------------------------------------------------------------------
    # Create the feature group for site features with tooltips.
    sites_group = folium.FeatureGroup(name="Sites")
    
    # For each feature, build a tooltip using our function that now considers weight.
    for feature in geojson_data.get('features', []):
        tooltip_content = create_tooltip_content(feature['properties'], config.dataset_info, current_weight_scenario)
        gj_feature = folium.GeoJson(feature, style_function=style_func,
                                    tooltip=folium.Tooltip(tooltip_content, sticky=True))
        gj_feature.add_to(sites_group)
    sites_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    # ---------------------------------------------------------------------
    # Add Title overlay (using CSS class for responsiveness)
    title_html = f'''
    <div class="resilience-title">
        Resilience Hub Analysis - {scenario_name} Scenario
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # ---------------------------------------------------------------------
    # Add ONE Analysis logo text box
    analysis_text_html = f'''
    <div class="analysis-text">
        <span style="font-family: 'Futura', sans-serif; font-weight: bold; color: #4c5da4;">one</span>
        <span style="font-family: 'Futura', sans-serif; font-weight: 300; color: #4c5da4;"> analysis</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(analysis_text_html))
    
    # ---------------------------------------------------------------------
    # Add Legend overlay (wrapped in a responsive container)
    legend_html = """
    <div class="legend-box">
        <h4 style="margin-bottom: 5px; font-size: 17px; font-weight: bold;">Legend</h4>
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <!-- Priority Score Section -->
            <div>
                <h5 style="margin: 5px 0; font-weight: bold;">Resilience Hub Priority Score</h5>
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <div style="
                        width: 120px; 
                        height: 10px; 
                        border-radius: 5px;
                        background: linear-gradient(to right, #f4fa7d, #00732a);
                    "></div>
                    <div style="display: flex; justify-content: space-between; width: 120px;">
                        <span style="font-size: 12px;">lower</span>
                        <span style="font-size: 12px;">higher</span>
                    </div>
                </div>
            </div>
            <!-- Area Overlays Section -->
            <div>
                <h5 style="margin: 5px 0; font-weight: bold;">Area Overlays</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: rgba(128, 128, 128, 0.2); border: 2px solid gray; border-radius: 3px;"></div>
                        <span style="font-size: 12px;">CSC Neighborhoods</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: rgba(244, 164, 96, 0.2); border: 1px solid sandybrown; border-radius: 3px;"></div>
                        <span style="font-size: 12px;">Persistent Poverty Areas</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: rgba(186, 85, 211, 0.2); border: 1px solid MediumOrchid; border-radius: 3px;"></div>
                        <span style="font-size: 12px;">Federal Opportunity Zones</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ---------------------------------------------------------------------
    # Add Donut Chart Overlay with Interactive Info Icons in Legend
    # Use the default weight scenario from the config.
    scenario_weights = config.weight_scenarios.get("Default", {})

    # Build active_items list based on dataset_info keys and corresponding weight.
    active_items = []
    for index_key in config.dataset_info:
        weight_key = f"{index_key}_weight"
        weight_val = scenario_weights.get(weight_key, 0)
        if weight_val != 0:
            active_items.append((index_key, weight_val))
    
    if active_items:
        sizes = [weight for _, weight in active_items]
        colors = [config.dataset_info[index_key]['hex'] for index_key, _ in active_items]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 3), dpi=90)
        wedges, texts, autotexts = ax.pie(
             sizes,
             labels=None,
             autopct=lambda pct: f"{int(round(pct))}%",
             pctdistance=0.75,
             startangle=90,
             wedgeprops={'width': 0.5, 'edgecolor': 'white'},
             colors=colors
        )
        ax.set_aspect("equal")
        plt.setp(autotexts, size=12, fontfamily='Verdana', weight='bold', color='black', va='center', ha='center')
        
        import io
        svg_buf = io.BytesIO()
        plt.savefig(svg_buf, format='svg', transparent=True, bbox_inches='tight')
        svg_buf.seek(0)
        svg_data = svg_buf.read().decode('utf-8')
        svg_buf.close()
        plt.close(fig)
        
        # Build a custom HTML legend with interactive tooltips.
        donut_legend_html = '<div style="margin-top:10px;">'
        for index_key, weight in active_items:
            info = config.dataset_info[index_key]
            label = info['name']
            description = info.get('description', 'No description available.')
            tooltip_content = f"""
                <div class='tooltip-content'>
                    <strong>{label}</strong><br>
                    {description}
                </div>
            """
            donut_legend_html += f"""
                <div class="legend-item" style="font-size:12pt; color:{info['hex']}; margin-bottom:10px; margin-left:30px; display:flex; align-items:center;">
                    <div class="tooltip-container">
                        <span class="info-icon" style="cursor:pointer; margin-right:5px; font-size:12pt;">&#9432;</span>
                        <div class="tooltip">{tooltip_content}</div>
                    </div>
                    <span>{label}</span>
                </div>
            """
        donut_legend_html += '</div>'
        
        tooltip_css = """
            <style>
                .tooltip-container {
                    position: relative;
                    display: inline-block;
                }
                .tooltip {
                    visibility: hidden;
                    position: absolute;
                    left: 25px;
                    background-color: white;
                    color: #333;
                    padding: 10px;
                    border-radius: 20px;
                    font-size: 8pt;
                    width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    border: 1px solid #ddd;
                    z-index: 1001;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .tooltip-content {
                    line-height: 1.4;
                }
                .tooltip-container:hover .tooltip {
                    visibility: visible;
                    opacity: 1;
                }
                .legend-item {
                    position: relative;
                    white-space: nowrap;
                }
                .info-icon {
                    color: inherit;
                    font-weight: bold;
                }
            </style>
        """
        
        donut_combined = tooltip_css + svg_data + donut_legend_html
    else:
        donut_combined = "<svg></svg>"
    
    donut_html = f'''
    <div class="donut-overlay">
        <h4 style="margin-top: 0; margin-bottom: 0; font-weight: bold; font-size: 25px; text-align: center;">
            Analysis Weights
        </h4>
        {donut_combined}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(donut_html))
    
    # ---------------------------------------------------------------------
    # Build output filename and save the map.
    fname = f"{scenario_name}_resilience_hub_webmap"
    if neighborhood_name:
        fname += f"_{neighborhood_name.replace(' ', '_')}"
    html_map_path = os.path.join(config.output_dir, f"{fname}.html")
    
    m.save(html_map_path)
    print(f"Webmap saved to: {html_map_path}")
    return html_map_path

# Minimal testing snippet (if running webmap.py directly)
if __name__ == "__main__":
    from config import ResilienceHubConfig
    config = ResilienceHubConfig()
    exported_geojson = os.path.join(config.output_dir, "sites_with_indices.geojson")
    scenario_geojsons = {"ResilienceHub": exported_geojson}
    build_webmap(scenario_geojsons, config)