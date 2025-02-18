"""
webmap.py

This module builds an interactive HTML webmap for the Resilience Hub Prioritization Analysis.
It reads an exported GeoJSON file (with computed indices) and creates a folium map with popups
that display analysis metrics based on the configuration in config.py.
"""

import os
import folium
import geopandas as gpd

def create_popup_content(properties, dataset_info):
    """
    Build an HTML snippet that displays:
      - A header with the site identifier,
      - The overall Suitability Index,
      - For each analysis factor: the raw value (with any prefix/suffix) and the normalized index.
    
    Uses the mapping in dataset_info (from config) to obtain display settings.
    This content is now used for tooltips.
    """
    # Added border-radius and padding to make rounded corners look better
    html = "<div style='font-family: Verdana; font-size: 12px; border-radius: 8px; padding: 12px;'>"
    # Use a provided identifier if available.
    site_id = properties.get('name', properties.get('ID', properties.get('OBJECTID', 'Site')))
    html += f"<h4>Site: {site_id}</h4>"
    
    overall = properties.get('Suitability_Index')
    if overall is not None:
        try:
            overall_disp = f"{float(overall):.3f}"
        except Exception:
            overall_disp = overall
        html += f"<p><strong>Overall Suitability:</strong> {overall_disp}</p>"
    
    html += "<ul style='list-style-type: none; padding: 3;'>"
    for factor, info in dataset_info.items():
        alias = info.get('alias')    # normalized index field
        raw_field = info.get('raw')   # raw value field
        if alias in properties and raw_field in properties:
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
                f"<li>{colored_name}: {info.get('prefix','')}{raw_disp}{info.get('suffix','')}"
                f"<br>Index: {norm_disp}/1</li>"
            )
    html += "</ul></div>"
    return html
def style_function(feature):
    """
    A basic style function for the GeoJSON layer.
    (This function is not used when applying a gradient style based on index_norm.)
    """
    return {
         "fillColor": "#228B22",  # forest green
         "color": "#228B22",
         "weight": 2,
         "fillOpacity": 0.6
    }

class OnEachFeatureCallback:
    """
    (Unused in this version)
    A callable class that previously bound a popup to each feature.
    """
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info

    def __call__(self, feature, layer):
        popup_content = create_popup_content(feature['properties'], self.dataset_info)
        layer.bindPopup(popup_content)

    def __json__(self):
        # Prevent JSON serialization of this callback.
        return None


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

    Parameters:
      - color1: Hex string for the start color (e.g., "#f4fa7d").
      - color2: Hex string for the end color (e.g., "#16A34A").
      - t: A value between 0 and 1.
      
    Returns:
      A hex string representing the interpolated color.
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
    Returns a style function that interpolates the fillColor based on the
    'index_norm' property of each feature.
    """
    def style_function(feature):
        try:
            index_norm = float(feature['properties'].get('index_norm', 0))
        except Exception:
            index_norm = 0
        # Normalize the index_norm value between 0 and 1.
        if max_val != min_val:
            normalized = (index_norm - min_val) / (max_val - min_val)
        else:
            normalized = 0
        # Clamp t between 0 and 1.
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
    Build an interactive folium webmap for the Resilience Hub analysis
    by manually adding GeoJSON features and binding tooltips (instead of popups),
    along with custom title, logo, legend, and donut chart overlay.
    
    Parameters:
      - scenario_geojsons: dict mapping a scenario name (e.g., "ResilienceHub")
                           to the path of the exported GeoJSON file with computed indices.
      - config: an instance of ResilienceHubConfig.
      - neighborhood_name: (optional) if provided, appended to the output filename.
      
    Returns:
      The file path of the saved HTML map.
    """
    # For this analysis, assume only one scenario is passed.
    scenario_name = list(scenario_geojsons.keys())[0]
    print(f"Building Resilience Hub webmap for scenario: {scenario_name}")
    
    geojson_path = scenario_geojsons[scenario_name]
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf.set_crs(config.crs, inplace=True)
    # Reproject to EPSG:4326 for folium.
    gdf = gdf.to_crs("EPSG:4326")
    
    # Compute global min and max for the index_norm column.
    min_index_norm = gdf["index_norm"].min()
    max_index_norm = gdf["index_norm"].max()
    
    # Create a style function that colors features based on index_norm.
    style_func = get_style_function(min_index_norm, max_index_norm)
    
    # Get the geo_interface and remove any function-valued properties.
    geojson_data = gdf.__geo_interface__
    geojson_data = clean_geojson_properties(geojson_data)
    
    # Determine map center from the total bounds.
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create the base map.
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=12,
                   tiles="CartoDB Positron")
    
    # Add custom CSS to round the tooltip corners
    tooltip_custom_css = """
    <style>
    .leaflet-tooltip {
        border-radius: 20px !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(tooltip_custom_css))

    # ---------------------------------------
    # Add Data Layers
    # ---------------------------------------
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

    # ---------------------------------------
    # Create the feature group for site features with tooltips.
    # ---------------------------------------
    sites_group = folium.FeatureGroup(name="Sites")
    
    # Loop over each feature and add it to the map with the new style.
    sites_group = folium.FeatureGroup(name="Sites")
    for feature in geojson_data.get('features', []):
        tooltip_content = create_popup_content(feature['properties'], config.dataset_info)
        gj_feature = folium.GeoJson(feature, style_function=style_func)
        # Bind the tooltip instead of a popup.
        gj_feature.add_child(folium.Tooltip(tooltip_content))
        gj_feature.add_to(sites_group)
    sites_group.add_to(m)
    folium.LayerControl().add_to(m)

    # ---------------------------------------
    # Add title
    # ---------------------------------------
    title_html = f'''
    <div style="position: fixed; 
                top: 30px; 
                left: 30px; 
                z-index: 1000;
                background-color: white;
                padding: 10px;
                border: 2px solid grey;
                border-radius: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                font-size: 30px;
                font-weight: bold;
                font-family: Futura Bold, sans-serif;">
        {list(scenario_geojsons.keys())[0]} Prioritization Tool
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # ---------------------------------------
    # Add ONE Analysis logo text box
    # ---------------------------------------
    analysis_text_html = f'''
    <div style="
        position: fixed;
        bottom: 30px;  
        left: 30px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-size: 30px;
    ">
        <span style="font-family: Futura Bold, sans-serif; color: #4c5da4;">one</span>
        <span style="font-family: Futura Light, sans-serif; color: #4c5da4;"> analysis</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(analysis_text_html))
    
    # ---------------------------------------
    # Add Legend
    # ---------------------------------------  
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 32px; 
        right: 32px; 
        z-index: 9999;
        background-color: white; 
        padding: 12px; 
        border: 2px solid rgb(156 163 175);
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    ">
        <h4 style="margin-bottom: 12px; font-size: 20px; font-weight: bold;">Legend</h4>
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <!-- Priority Score Section -->
            <div>
                <h5 style="margin: 4px 0; font-size: 14px; font-weight: 500;">Resilience Hub Priority Score</h5>
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <!-- Gradient bar -->
                    <div style="
                        width: 128px; 
                        height: 8px; 
                        border-radius: 2px;
                        background: linear-gradient(to right, #f4fa7d, #00732a);
                    "></div>
                    <!-- Score labels -->
                    <div style="display: flex; justify-content: space-between; width: 128px;">
                        <span style="font-size: 12px;">0</span>
                        <span style="font-size: 12px;">1</span>
                    </div>
                </div>
            </div>

            <!-- Area Overlays Section -->
            <div>
                <h5 style="margin: 4px 0; font-size: 14px; font-weight: 500;">Area Overlays</h5>
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: rgba(128, 128, 128, 0.2); border: 2px solid gray; border-radius: 2px;"></div>
                        <span style="font-size: 14px;">CSC Neighborhoods</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: rgba(244, 164, 96, 0.2); border: 1px solid sandybrown; border-radius: 2px;"></div>
                        <span style="font-size: 14px;">Persistent Poverty Areas</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: rgba(186, 85, 211, 0.2); border: 1px solid MediumOrchid; border-radius: 2px;"></div>
                        <span style="font-size: 14px;">Federal Opportunity Zones</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ---------------------------------------
    # Add Donut Chart Overlay with Interactive Info Icons in Legend
    # ---------------------------------------    
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
        plt.tight_layout()
        
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
    <div style="
        position: fixed;
        top: 110px;
        left: 30px;
        z-index: 1000;
        background: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-family: 'Verdana', sans-serif;
    ">
        <h4 style="margin-top: 0; margin-bottom: 0; font-weight: bold; font-size: 25px; text-align: center;">
            Analysis Weights
        </h4>
        {donut_combined}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(donut_html))
    
    # ---------------------------------------
    # Build output filename and save the map.
    # ---------------------------------------
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
    # Assume the exported GeoJSON is saved as "sites_with_indices.geojson" in the output directory.
    exported_geojson = os.path.join(config.output_dir, "sites_with_indices.geojson")
    scenario_geojsons = {"ResilienceHub": exported_geojson}
    build_webmap(scenario_geojsons, config)