# config.py
import os
import copy

class ResilienceHubConfig:
    def __init__(self, analysis_type='citywide'):
        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.script_dir = os.getcwd()

        # Directories
        self.input_dir = os.path.join(self.script_dir, 'input')
        self.output_dir = os.path.join(self.script_dir, 'output')
        self.temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # CRS and base parameters
        self.crs = 'EPSG:6539'
        self.min_parking_area = 5000

        # Data files dictionary
        self.data_files = {
            'nsi': {'path': os.path.join(self.input_dir, 'NYC_NSI.geojson'), 'columns': None},
            'lots': {'path': os.path.join(self.input_dir, 'MapPLUTO.geojson'), 'columns': None},
            'pofw': {'path': os.path.join(self.input_dir, 'NYC_POFW.geojson'), 'columns': None},
            'buildings': {'path': os.path.join(self.input_dir, 'NYC_Buildings.geojson'), 'columns': None},
            'facilities': {'path': os.path.join(self.input_dir, 'NYC_Facilities.geojson'), 'columns': None},
        }

        # Lists of strings used in preprocessing
        self.school_types = [
            'OTHER SCHOOL - NON-PUBLIC', 'ELEMENTARY SCHOOL - PUBLIC',
            'SECONDARY SCHOOL - CHARTER', 'HIGH SCHOOL - PUBLIC', 'CHARTER SCHOOL',
            'COMPASS ELEMENTARY', 'LICENSED PRIVATE SCHOOLS',
            'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL - PUBLIC', 'K-8 SCHOOL - PUBLIC',
            'PRE-K CENTER', 'ELEMENTARY SCHOOL - CHARTER', 'K-8 SCHOOL - CHARTER',
            'PRE-SCHOOL FOR STUDENTS WITH DISABILITIES', 'K-12 ALL GRADES SCHOOL - PUBLIC, SPECIAL EDUCATION',
            'HIGH SCHOOL - CHARTER', 'K-12 ALL GRADES SCHOOL - CHARTER', 'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL'
        ]
        # self.priority2_types = [
        #     'OTHER SCHOOL - NON-PUBLIC', 'COMMUNITY SERVICES', 'ELEMENTARY SCHOOL - PUBLIC',
        #     'FOOD PANTRY', 'HIGH SCHOOL - PUBLIC', 'CHARTER SCHOOL', 'COMPASS ELEMENTARY',
        #     'LICENSED PRIVATE SCHOOLS', 'SENIOR CENTER', 'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL - PUBLIC',
        #     'AFTERSCHOOL PROGRAMS; COMMUNITY SERVICES; EDUCATIONAL SERVICES; FAMILY SUPPORT',
        #     'HOSPITAL EXTENSION CLINIC', 'FIREHOUSE', 'NURSING HOME', 'K-8 SCHOOL - PUBLIC',
        #     'COMMUNITY SERVICES; FAMILY SUPPORT; HOUSING SUPPORT; IMMIGRANT SERVICES',
        #     'SENIOR SERVICES', 'PRE-K CENTER', 'SOUP KITCHEN', 'ELEMENTARY SCHOOL - CHARTER',
        #     'K-8 SCHOOL - CHARTER', 'PRE-SCHOOL FOR STUDENTS WITH DISABILITIES', 'HOSPITAL',
        #     'ADULT HOME', 'SENIORS', 'AFTERSCHOOL PROGRAMS; COMMUNITY SERVICES; EDUCATIONAL SERVICES; FAMILY SUPPORT; IMMIGRANT SERVICES',
        #     'K-12 ALL GRADES SCHOOL - PUBLIC, SPECIAL EDUCATION', 'HIGH SCHOOL - CHARTER',
        #     'K-12 ALL GRADES SCHOOL - CHARTER', 'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL',
        #     'SECONDARY SCHOOL - CHARTER', 'BOROUGH OFFICE'
        # ]

        self.priority2_types = []
        
        self.facilities_required_columns = [
            'geometry', 'fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
            'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority'
        ]
        self.lot_fields = [
            'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 'ComArea',
            'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR'
        ]
        self.nsi_fields = ['bldgtype', 'num_story', 'found_type', 'found_ht']

        # List of fields to aggregate when merging building footprints with point data
        self.agg_fields = [
            'fclass', 'name', 'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea',
            'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR', 'FACTYPE', 'FACSUBGRP',
            'FACGROUP', 'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority'
        ]

        # Analysis type–specific parameters
        base_params = {'analysis_buffer_ft': 2000, 'min_resilience_score': 0.5}
        citywide_specific = {'top_site_count': 50, 'min_site_area_ft2': 1000}
        neighborhood_specific = {'top_site_count': 10, 'min_site_area_ft2': 500}
        self.analysis_params = base_params.copy()
        if analysis_type.lower() == 'citywide':
            self.analysis_params.update(citywide_specific)
        elif analysis_type.lower() == 'neighborhood':
            self.analysis_params.update(neighborhood_specific)

        # Weight scenarios for the indices
        self.weight_scenarios = {
            'Default': {
                'Adaptability_Index_weight': 0.10,
                'Solar_Energy_Index_weight': 0.15,
                'Heat_Hazard_Index_weight': 0.05,
                'Flood_Hazard_Index_weight': 0.10,
                'Heat_Vulnerability_Index_weight': 0.20,
                'Flood_Vulnerability_Index_weight': 0.15,
                'Service_Population_Index_weight': 0.25,
                'Adaptability_Index_components': {
                    'RS_Priority': 0.0,
                    'CAPACITY': 0.35,
                    'BldgArea': 0.50,
                    'StrgeArea': 0.15
                },
                'Solar_Energy_Index_components': {'peak_power': 1.0},
                'Heat_Hazard_Index_components': {'heat_mean': 1.0},
                'Flood_Hazard_Index_components': {
                    'Cst_500_in': 0.05,
                    'Cst_500_nr': 0.05,
                    'Cst_100_in': 0.15,
                    'Cst_100_nr': 0.15,
                    'StrmShl_in': 0.05,
                    'StrmShl_nr': 0.05,
                    'StrmDp_in': 0.10,
                    'StrmDp_nr': 0.10,
                    'StrmTid_in': 0.15,
                    'StrmTid_nr': 0.15
                },
                'Heat_Vulnerability_Index_components': {'hvi_area': 1.0},
                'Flood_Vulnerability_Index_components': {'ssvul_area': 0.50, 'tivul_area': 0.50},
                'Service_Population_Index_components': {'pop_est': 1.0}
            },
            'AggressiveSolar': {
                'Adaptability_Index_weight': 0.08,
                'Solar_Energy_Index_weight': 0.30,
                'Heat_Hazard_Index_weight': 0.04,
                'Flood_Hazard_Index_weight': 0.08,
                'Heat_Vulnerability_Index_weight': 0.18,
                'Flood_Vulnerability_Index_weight': 0.10,
                'Service_Population_Index_weight': 0.22,
                'Adaptability_Index_components': {
                    'RS_Priority': 0.0,
                    'CAPACITY': 0.40,
                    'BldgArea': 0.45,
                    'StrgeArea': 0.15
                },
                'Solar_Energy_Index_components': {'peak_power': 1.0},
                'Heat_Hazard_Index_components': {'heat_mean': 1.0},
                'Flood_Hazard_Index_components': {
                    'Cst_500_in': 0.04,
                    'Cst_500_nr': 0.04,
                    'Cst_100_in': 0.12,
                    'Cst_100_nr': 0.12,
                    'StrmShl_in': 0.04,
                    'StrmShl_nr': 0.04,
                    'StrmDp_in': 0.08,
                    'StrmDp_nr': 0.08,
                    'StrmTid_in': 0.16,
                    'StrmTid_nr': 0.16
                },
                'Heat_Vulnerability_Index_components': {'hvi_area': 1.0},
                'Flood_Vulnerability_Index_components': {'ssvul_area': 0.55, 'tivul_area': 0.45},
                'Service_Population_Index_components': {'pop_est': 1.0}
            }
        }

        self.COMPONENTS = {
            "Adaptability_Index_components": self.weight_scenarios['Default']['Adaptability_Index_components'],
            "Solar_Energy_Index_components": self.weight_scenarios['Default']['Solar_Energy_Index_components'],
            "Heat_Hazard_Index_components": self.weight_scenarios['Default']['Heat_Hazard_Index_components'],
            "Flood_Hazard_Index_components": self.weight_scenarios['Default']['Flood_Hazard_Index_components'],
            "Heat_Vulnerability_Index_components": self.weight_scenarios['Default']['Heat_Vulnerability_Index_components'],
            "Flood_Vulnerability_Index_components": self.weight_scenarios['Default']['Flood_Vulnerability_Index_components'],
            "Service_Population_Index_components": self.weight_scenarios['Default']['Service_Population_Index_components']
        }        
        
        # ---------------------------------------------------------------
        # Additional analysis-specific files and parameters
        # ---------------------------------------------------------------
        # For Heat Analysis:
        self.HEAT_FILE = os.path.join(self.input_dir, 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif')  # heat raster file
        self.RESOLUTION = 30  # Resolution in the units of your CRS (e.g., feet or meters)

        # For Flood Analysis:
        self.FEMA_RASTER = os.path.join(self.input_dir, 'FEMA_FloodHaz_Raster.tif')  # FEMA flood raster
        self.STORM_RASTER = os.path.join(self.input_dir, 'Stormwater2080_Raster.tif')  # Storm flood raster
        self.NUM_WORKERS = 4  # Number of worker processes (adjust as needed)

        # For Vulnerability Analysis:
        self.HVI_DATA = os.path.join(self.input_dir, 'HVI.geojson')  # HVI dataset
        self.FVI_DATA = os.path.join(self.input_dir, 'FVI.geojson')  # FVI dataset

        # For Census Analysis:
        self.CENSUS_BLOCKS_FILE = os.path.join(self.input_dir, 'nyc_blocks_with_pop.geojson')
        self.nyc_counties = ['005', '047', '061', '081', '085']  # FIPS codes for NYC counties

        # ---------------------------------------------------------------
        # Data dictionary for indices
        # ---------------------------------------------------------------
        self.dataset_info = {
            "Adaptability_Index": {
                "alias": "Adaptability",
                "raw": "bldg_adapt",
                "name": "Building Adaptability",
                "description": "Evaluates building characteristics that support adaptive reuse.",
                "prefix": "",
                "suffix": "",
                "hex": "#99789D"
            },
            "Solar_Energy_Index": {
                "alias": "SolarEnergy",
                "raw": "solar_pot",
                "name": "Solar Energy Potential",
                "description": "Estimates potential for solar power generation.",
                "prefix": "~",
                "suffix": " kW",
                "hex": "#E2C85D"
            },
            "Heat_Hazard_Index": {
                "alias": "HeatHaz",
                "raw": "heat_mean",
                "name": "Heat Hazard",
                "description": "Measures heat based on average summer temperatures.",
                "prefix": "",
                "suffix": " °F",
                "hex": "#C26558"
            },
            "Flood_Hazard_Index": {
                "alias": "FloodHaz",
                "raw": "flood_risk",
                "name": "Flood Hazard",
                "description": "Assesses flood risk using historical flood data and proximity.",
                "prefix": "",
                "suffix": "",
                "hex": "#6988B0"
            },
            "Heat_Vulnerability_Index": {
                "alias": "HeatVuln",
                "raw": "hvi_area",
                "name": "Heat Vulnerability",
                "description": "Quantifies social vulnerability to heat hazards.",
                "prefix": "",
                "suffix": "",
                "hex": "#C77851"
            },
            "Flood_Vulnerability_Index": {
                "alias": "FloodVuln",
                "raw": "flood_vuln",
                "name": "Flood Vulnerability",
                "description": "Quantifies social vulnerability to flood hazards.",
                "prefix": "",
                "suffix": "",
                "hex": "#6BADC9"
            },
            "Service_Population_Index": {
                "alias": "ServicePop",
                "raw": "pop_est",
                "name": "Service Population",
                "description": "Estimates the number of people served within a defined area.",
                "prefix": "~",
                "suffix": " people",
                "hex": "#777C77"
            }
        }

    def copy(self):
        return copy.deepcopy(self)

# For convenience, you can instantiate and use this config in other modules:
# config = ResilienceHubConfig()
# print(config.dataset_info)
        
if __name__ == "__main__":
    config = ResilienceHubConfig()
    print("Config dataset info:")
    for key, info in config.dataset_info.items():
        print(f"{key}: {info}")