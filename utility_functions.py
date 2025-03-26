import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pyincore import Mapping, MappingSet, FragilityCurveSet
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from scipy.stats import norm, lognorm

########################################################################################
################ HAZARD EXPOSURE UTILITY FUNCTIONS #####################################
########################################################################################

# Function to check if a point is within bounds
def is_within_bounds(geometry, bounds):
    return (
        bounds.left <= geometry.x <= bounds.right and
        bounds.bottom <= geometry.y <= bounds.top
    )

def extract_raster_value(geometry, raster, transform):
    try:
        row, col = rasterio.transform.rowcol(transform, geometry.x, geometry.y)
        row, col = int(row), int(col)  # Ensure indices are integers
        return raster[row, col]
    except IndexError:
        # Return a placeholder (e.g., NaN) for points outside the raster bounds
        return np.nan

def attach_flood_hazard(gdf, raster_path, raster_value = 'inundationDepth'):

    # Load the raster file
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs  # Get CRS of the raster
        raster_data = src.read(1)  # Read the raster values
        transform = src.transform  # Get the affine transform
    
    # Reproject GeoDataFrame to match raster CRS, if needed
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    
    # Apply the extraction logic
    if 'Point' in gdf.geom_type.unique():
        # Filter points outside the raster bounds
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
        gdf = gdf[gdf['geometry'].apply(lambda geom: is_within_bounds(geom, bounds))]
        
        gdf[raster_value] = gdf['geometry'].apply(
            lambda geom: extract_raster_value(geom, raster_data, transform)
        )
    
    elif 'LineString' in gdf.geom_type.unique():
        def get_average_raster_value(line):
            # Sample raster values at evenly spaced points along the line
            sample_points = [
                line.interpolate(i / 10.0, normalized=True) for i in range(11)
            ]
            values = [
                extract_raster_value(point, raster_data, transform)
                for point in sample_points
                if point.within(gpd.GeoSeries([line]).unary_union)
            ]
            # Return the mean of valid raster values
            return np.nanmean(values) if values else np.nan
            
        gdf[raster_value] = gdf['geometry'].apply(get_average_raster_value)
    
    return gdf

def create_networkx_graph(nodes_gdf, 
                          edges_gdf,
                          fromnode = 'fromnode',
                          tonode = 'tonode',
                          node_id = 'nodenwid'
                         
):
    # Ensure node_id is present in nodes_gdf
    if node_id not in nodes_gdf.columns:
        raise ValueError("Ensure that the correct Node ID Column in the input nodes geodataframe is used")

    # Ensure source and target are present in edges_gdf
    if fromnode not in edges_gdf.columns or tonode not in edges_gdf.columns:
        raise ValueError("Ensure that the correct fromnode and tonode columns in the input nodes geodataframe is used")

    # Create a directed graph (change to nx.Graph() for undirected)
    # G = nx.DiGraph()
    G = nx.Graph()

    # Add nodes with attributes
    for _, row in nodes_gdf.iterrows():
        G.add_node(
            row[node_id],
            **row.drop(['geometry', node_id]).to_dict(),  # Add all columns except geometry and node_id
            geometry=row['geometry']  # Add geometry separately
        )

    # Add edges with attributes
    for _, row in edges_gdf.iterrows():
        G.add_edge(
            row[fromnode],  # Source node
            row[tonode],  # Target node
            **row.drop(['geometry', fromnode, tonode]).to_dict(),  # Add all columns except geometry, source, and target
            geometry=row['geometry']  # Add geometry separately
        )

    return G

def sample_raster_along_line(raster, line, num_samples=100):
    """
    Samples raster values along a line at equally spaced points.
    
    Parameters:
    - raster (rasterio.io.DatasetReader): Opened rasterio dataset.
    - line (shapely.geometry.LineString): Line geometry to sample along.
    - num_samples (int): Number of equally spaced points to sample along the line.
    
    Returns:
    - List of sampled raster values.
    """
    # Generate equally spaced points along the line
    distances = np.linspace(0, line.length, num_samples)
    points = [line.interpolate(distance) for distance in distances]

    # Convert points to raster coordinates and sample
    sampled_values = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        row, col = int(row), int(col)  # Ensure indices are integers
        try:
            value = raster.read(1)[row, col]
            sampled_values.append(value)
        except IndexError:
            sampled_values.append(np.nan)  # Handle out-of-bounds cases
    
    return sampled_values

# Utility function: Classify based on cumulative probabilities
def classify_numbers(numbers, limits):
    classes = np.zeros(len(numbers), dtype=int)  # Array to store class labels
    for i in range(1,len(limits)):
        # Classify numbers between the current and next limit
        classes[(numbers > limits[i-1]) & (numbers <= limits[i])] = i
    return classes

def find_nearest_network_node(building, network_nodes):
    nearest_geom = nearest_points(building.geometry, network_nodes.geometry.union_all())[1]
    nearest_node = network_nodes[network_nodes.geometry == nearest_geom]
    return nearest_node.iloc[0].name  # Assuming node IDs are in the index or a unique column

########################################################################################
################ FRAGILITY MODEL DEFINITION ############################################
########################################################################################

def get_building_fragility():

    # create a fragility curve set with three limit state
    fragility_curve_basement = {
        "description": "Basement Submerged",
        "rules": [
            {
                "condition": [
                    "inundationDepth > 0"
                ],
                "expression": "scipy.stats.norm.cdf((math.log(inundationDepth) - 0.5)/(0.05))"
            }
        ],
        "returnType": {
            "type": "Limit State",
            "unit": "",
            "description": "basement_flooded"
        }
    }
    
    fragility_curve_FL_1 = {
        "description": "First Floor Submerged",
        "rules": [
            {
                "condition": [
                    "inundationDepth > 0"
                ],
                "expression": "scipy.stats.norm.cdf((math.log(inundationDepth) - 1.0)/(0.2))"
            }
        ],
        "returnType": {
            "type": "Limit State",
            "unit": "",
            "description": "Floor1_flooded"
        }
    }
    
    fragility_curve_FL_2 = {
        "description": "Second Floor Submerged",
        "rules": [
            {
                "condition": [
                    "inundationDepth > 0"
                ],
                "expression": "scipy.stats.norm.cdf((math.log(inundationDepth) - 2.565)/(0.15))"
            }
        ],
        "returnType": {
            "type": "Limit State",
            "unit": "",
            "description": "Floor2_flooded"
        }
    }

    ## Fragility curve for building 2-storey or more buildings
    # place three curves into a set with extra metadata
    frag1_metadata = {
        "id":"local_fragility_curve_set",
        "description": "Flood Fragility of Buildings - 2 or more floors",
        "demandTypes": ["inundationDepth"],
        "demandUnits": ["ft"],
        "resultType":"Limit State",
        "hazardType":"flood",
        "inventoryType":"building",
        "fragilityCurves":[
            fragility_curve_basement,
            fragility_curve_FL_1,
            fragility_curve_FL_2,
        ],
        "curveParameters": [
            {
                "name": "inundationDepth",
                "unit": "ft",
                "description": "Inundation Level at Building Location from Hazard Service",
                "fullName": "inundationDepth",
            },
            {
                "name": "Floor",
                "unit": "",
                "description": "Number of floors in the building",
                "expression": "2"                                                   # Default value
            }
        ]
    }

    # construct the fragility curve object to use
    fragility_curve_set1 = FragilityCurveSet(frag1_metadata)
    fragility_curve_set1

    ## Fragility curve for building 1-storey buildings
    # place two curves into a set with extra metadata
    frag2_metadata = {
        "id":"local_fragility_curve_set",
        "description": "Flood Fragility of Buildings - 1 floor",
        "demandTypes": ["inundationDepth"],
        "demandUnits": ["ft"],
        "resultType":"Limit State",
        "hazardType":"flood",
        "inventoryType":"building",
        "fragilityCurves":[
            fragility_curve_basement,
            fragility_curve_FL_1
        ],
        "curveParameters": [
            {
                "name": "inundationDepth",
                "unit": "ft",
                "description": "Inundation Level at Building Location from Hazard Service",
                "fullName": "waterSurfaceElevation",
            },
            {
                "name": "Floor",
                "unit": "",
                "description": "Number of floors in the building",
                "expression": "1"                                                   # Default value
            }
        ]
    }
    
    # construct the fragility curve object to use
    fragility_curve_set2 = FragilityCurveSet(frag2_metadata)
    fragility_curve_set2

    entry_1 = {"Fragility ID Code": fragility_curve_set1}
    rules_1 = [["int Floors GE 2"]]
    mapping_1 = Mapping(entry_1, rules_1)
    
    entry_2 = {"Fragility ID Code": fragility_curve_set2}
    rules_2 = [["int Floors GE 1", "int Floors LE 1"]]
    mapping_2 = Mapping(entry_2, rules_2)
    
    mapping_set = {
        'id': 'local placeholder',
        'name': 'testing local mapping object creation',
        'hazardType': 'flood',
        'inventoryType': 'building',
        'mappings': [
            mapping_1,
            mapping_2,
        ],
        'mappingType': 'fragility'
    }
    local_building_mapping_set = MappingSet(mapping_set)

    return fragility_curve_set1, fragility_curve_set2, local_building_mapping_set

def get_road_fragility():

    # create a fragility curve set with three limit state
    fragility_curve_roadway = {
        "description": "Road Submerged",
        "rules": [
            {
                "condition": [
                    "inundationDepth > 0"
                ],
                "expression": "scipy.stats.norm.cdf((math.log(inundationDepth) - 0.5)/(0.05))"
            }
        ],
        "returnType": {
            "type": "Limit State",
            "unit": "",
            "description": "road_overtopped"
        }
    }

    frag_road_metadata = {
        "id":"local_fragility_curve_set",
        "description": "Flood Fragility of Roads",
        "demandTypes": ["inundationDepth"],
        "demandUnits": ["ft"],
        "resultType":"Limit State",
        "hazardType":"flood",
        "inventoryType":"roads",
        "fragilityCurves":[fragility_curve_roadway],
        "curveParameters": [
            {
                "name": "inundationDepth",
                "unit": "ft",
                "description": "Maximum Depth from Hazard Service",
                "fullName": "inundationDepth",
            }
        ]
    }
    
    # construct the fragility curve object to use
    fragility_curve_set = FragilityCurveSet(frag_road_metadata)

    return fragility_curve_set

def get_substation_fragility():

    # create a fragility curve set with three limit state
    fragility_curve_substation = {
        "description": "Substation Submerged",
        "rules": [
            {
                "condition": [
                    "inundationDepth > 0"
                ],
                "expression": "scipy.stats.norm.cdf((math.log(inundationDepth) - 2.5)/(0.05))"
            }
        ],
        "returnType": {
            "type": "Limit State",
            "unit": "",
            "description": "water_in_substation"
        }
    }

    frag_road_metadata = {
        "id":"local_fragility_curve_set",
        "description": "Flood Fragility of Substations",
        "demandTypes": ["inundationDepth"],
        "demandUnits": ["ft"],
        "resultType":"Limit State",
        "hazardType":"flood",
        "inventoryType":"substations",
        "fragilityCurves":[fragility_curve_substation],
        "curveParameters": [
            {
                "name": "inundationDepth",
                "unit": "ft",
                "description": "Maximum Depth from Hazard Service",
                "fullName": "inundationDepth",
            }
        ]
    }
    
    # construct the fragility curve object to use
    fragility_curve_set = FragilityCurveSet(frag_road_metadata)

    return fragility_curve_set

########################################################################################
################ DAMAGE ANALYSIS #######################################################
########################################################################################
def run_building_damage_analysis(bldg_data, local_building_mapping_set):
    pf0 = []
    pf1 = []
    pf2 = []
    
    for row in bldg_data.iterrows():
        im = row[1]['inundationDepth']
        floors = row[1]['Floors']

        if floors == 1:
            mapping = local_building_mapping_set.mappings[1]
        else:
            mapping = local_building_mapping_set.mappings[0]

        fragility_set = mapping.entry['Fragility ID Code']
        probs = fragility_set.calculate_limit_state(hazard_values={'inundationDepth': im}, Floor = int(floors))
        
        pf_basement = probs['LS_0']
        pf_floor1 = probs['LS_1']
        
        if floors == 1:
            pf_floor2 = 0.0
        else:
            pf_floor2 = probs['LS_2']

        pf0.append(pf_basement)
        pf1.append(pf_floor1)
        pf2.append(pf_floor2)

    bldg_data['LS_0'] = pf0
    bldg_data['LS_1'] = pf1
    bldg_data['LS_2'] = pf2

    bldg_data['DS_0'] = 1 - bldg_data['LS_0']
    bldg_data['DS_1'] = bldg_data['LS_0'] - bldg_data['LS_1']
    bldg_data['DS_2'] = bldg_data['LS_1'] - bldg_data['LS_2']
    bldg_data['DS_3'] = bldg_data['LS_2']
    
    return bldg_data


def run_substation_damage_analysis(subs_data, fragility_curve_set_substations):
    pf = []
    
    for row in subs_data.iterrows():
        im = row[1]['inundationDepth']

        probs = fragility_curve_set_substations.calculate_limit_state(hazard_values={'inundationDepth': im})
        
        pf_subs = probs['LS_0']
        pf.append(pf_subs)

    subs_data['LS_0'] = pf

    subs_data['DS_0'] = 1 - subs_data['LS_0']
    subs_data['DS_1'] = subs_data['LS_0']
    
    return subs_data

def run_buildings_power(substations_df, buildings_df, edges_bldg_subs_df):
    buildings_df['DS_nopower'] = 1.0
    buildings_df['DS_power'] = 0.0
    
    for sub_id, sub_ds in zip(substations_df['ID'], substations_df['DS_1']):
        edges_from_sub = edges_bldg_subs_df[edges_bldg_subs_df['inode_substation'] == sub_id]
        for bldg_id in edges_bldg_subs_df['jnode_building']:
            building_row = buildings_df.index[buildings_df['ID'] == bldg_id]
            buildings_df.loc[building_row, 'DS_nopower'] = 1 - sub_ds
            buildings_df.loc[building_row, 'DS_power'] = sub_ds
    
    return buildings_df
    

def run_road_damage_analysis(edges_gdf, 
                             fragility_curve_set_roads,
                             elev_data_loc = 'files/USGS_1_n39w077_20220713.tif',
                             flood_data_loc = 'files/cat4_high_raster.tif',
                             num_samples=10
):
    """
    Assigns the maximum flood depth along each edge in a GeoDataFrame using a .tif raster file.
    
    Parameters:
    - edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing graph edges with 'geometry' column as line geometries.
    - raster_tif_path (str): Path to the .tif raster file with flood depth values.
    - num_samples (int): Number of points to sample along each line.
    
    Returns:
    - gpd.GeoDataFrame: Updated GeoDataFrame with a new column 'max_flood_depth'.
    """
    
    # Open the raster file
    with rasterio.open(flood_data_loc) as flood_raster, rasterio.open(elev_data_loc) as elev_raster:
        # Ensure the GeoDataFrame is in the same CRS as the raster
        if edges_gdf.crs != flood_raster.crs:
            edges_gdf = edges_gdf.to_crs(flood_raster.crs)
        # if edges_gdf.crs != elev_raster.crs:
        #     elev_raster = elev_raster.to_crs(edges_gdf.crs)
        
        # Compute the maximum flood depth for each edge
        max_depths = []
        for line in edges_gdf['geometry']:
            if isinstance(line, LineString):
                sampled_depths = sample_raster_along_line(flood_raster, line, num_samples)
                sampled_elevs = sample_raster_along_line(elev_raster, line, num_samples)
                actual_depths = [sampled_depths[i]-sampled_elevs[i] for i in range(len(sampled_elevs))]
                max_depth = np.nanmax(actual_depths)  # Handle NaNs
                max_depths.append(max_depth if not np.isnan(max_depth) else 0)
            else:
                max_depths.append(0)  # Handle invalid geometries
        
    # Add the max flood depth as a new column
    edges_gdf['inundationDepth'] = max_depths
    edges_gdf['inundationDepth'] = edges_gdf['inundationDepth'].apply(lambda x: max(x, 0))

    pf = []
    
    for row in edges_gdf.iterrows():
        im = row[1]['inundationDepth']
        probs = fragility_curve_set_roads.calculate_limit_state(hazard_values={'inundationDepth': im})
        pf_roads = probs['LS_0']
        pf.append(pf_roads)
        
    edges_gdf['LS_0'] = pf

    edges_gdf['DS_0'] = 1 - edges_gdf['LS_0']
    edges_gdf['DS_1'] = edges_gdf['LS_0']

    return edges_gdf

########################################################################################
######################## RECOVERY MODEL ################################################
########################################################################################
restoration_building = {"DS_0": 
                        {
                            "mu": 0.0,
                            "std": 0.0
                        },
                        "DS_1": 
                        {
                            "mu": 7.0,
                            "std": 1.0
                        },
                        "DS_2": 
                        {
                            "mu": 15.0,
                            "std": 3.0
                        },
                        "DS_3": 
                        {
                            "mu": 60.0,
                            "std": 6.0
                        }
                       }

restoration_power = {"DS_nopower": 
                        {
                            "mu": 0.0,
                            "std": 0.0
                        },
                        "DS_power": 
                        {
                            "mu": 3.0,
                            "std": 0.2
                        }
                       }

restoration_road = {"DS_0": 
                        {
                            "mu": 0.0,
                            "std": 0.0
                        },
                        "DS_1": 
                        {
                            "mu": 3.0,
                            "std": 0.2
                        }
                       }

# def building_recovery_model(buildings_df):
#     damage_states = ['DS_0', 'DS_1', 'DS_2', 'DS_3']

#     exp_downtime = np.zeros_like(np.array(buildings_df['DS_0']))
    
#     for DS in damage_states:
#         median = restoration_building[DS]["mu"]
#         disp = restoration_building[DS]["std"]

#         exp_downtime += median * np.array(buildings_df[DS])

#     buildings_df['downtime_expected'] = exp_downtime

#     return buildings_df

# def building_power_recovery_model(buildings_df):
#     damage_states = ['DS_nopower', 'DS_power']

#     exp_downtime = np.zeros_like(np.array(buildings_df['DS_nopower']))
    
#     for DS in damage_states:
#         median = restoration_power[DS]["mu"]
#         disp = restoration_power[DS]["std"]

#         exp_downtime += median * np.array(buildings_df[DS])

#     buildings_df['downtime_power'] = exp_downtime

#     return buildings_df

# def road_recovery_model(edges_df):
#     damage_states = ['DS_0', 'DS_1']

#     exp_downtime = np.zeros_like(np.array(edges_df['DS_0']))
    
#     for DS in damage_states:
#         median = restoration_road[DS]["mu"]
#         disp = restoration_road[DS]["std"]

#         exp_downtime += median * np.array(edges_df[DS])

#     edges_df['downtime_expected'] = exp_downtime

#     return edges_df

# Compute Recovery Given Sa
def get_recovery_curve_buildings(data_df: pd.DataFrame):

    time = np.arange(0.0, 120.0, 1.0)
    
    DSs = [1,2,3] #All damage states
    
    data_df['time'] = None
    data_df['recov'] = None

    data_df['mean_downtime'] = 0
    data_df['std_downtime'] = 0
    
    data_df = data_df.astype({'time': 'object',
                             'recov': 'object'})
    
    for i, row in data_df.iterrows():
        
        Pf = [0]*4 #presizing including the no damage state
        Pf[0] = row['DS_0'] #None
        Pf[1] = row['DS_1'] #DS1 
        Pf[2] = row['DS_2'] #DS2 
        Pf[3] = row['DS_3'] #DS2                                   #complete
        
        Pfn = Pf[0] 
        for ds in DSs:
            mean = restoration_building['DS_{}'.format(ds)]['mu']
            std = restoration_building['DS_{}'.format(ds)]['std']
            recov_given_failure = norm.cdf(time, mean, std)
            
            Pfn = Pfn + recov_given_failure * Pf[ds]
     
        #reset the variable names so that they are easy to be understood by the ENG team
        time_days = time
        functionality = Pfn
    
        data_df.at[i,'time'] = time
        data_df.at[i,'recov'] = functionality

        #Compute mean downtime
        expected_downtime = np.trapz(1-Pfn,time)  
        
        #Compute standard deviation of downtime
        expected_square_downtime = np.trapz(2*time*(1-Pfn), time)  
        variance_downtime = expected_square_downtime - expected_downtime ** 2
        standard_deviation = np.sqrt(variance_downtime) 
        
        #Final output in days (not converted to hours unlike WS and floods)
        mean_downtime_days = expected_downtime
        stdev_downtime_days = standard_deviation

        data_df.loc[i, 'mean_downtime'] = mean_downtime_days
        data_df.loc[i, 'std_downtime'] = stdev_downtime_days

    return data_df

# Compute Recovery Given Sa
def get_recovery_curve_power(data_df: pd.DataFrame):

    time = np.arange(0.0, 10.0, 1.0)
    
    DSs = [1] #All damage states
    
    data_df['time_power'] = None
    data_df['recov_power'] = None

    data_df['mean_downtime_power'] = 0
    data_df['std_downtime_power'] = 0
    
    data_df = data_df.astype({'time': 'object',
                             'recov_power': 'object'})
    
    for i, row in data_df.iterrows():
        
        Pf = [0]*2 #presizing including the no damage state
        Pf[0] = row['DS_nopower'] #None
        Pf[1] = row['DS_power'] #DS1
        
        Pfn = Pf[0] 
        for ds in DSs:
            mean = restoration_power['DS_power']['mu']
            std = restoration_power['DS_power']['std']
            recov_given_failure = norm.cdf(time, mean, std)
            
            Pfn = Pfn + recov_given_failure * Pf[ds]
     
        #reset the variable names so that they are easy to be understood by the ENG team
        time_days = time
        functionality = Pfn
    
        data_df.at[i,'time_power'] = time
        data_df.at[i,'recov_power'] = functionality

        #Compute mean downtime
        expected_downtime = np.trapz(1-Pfn,time)  
        
        #Compute standard deviation of downtime
        expected_square_downtime = np.trapz(2*time*(1-Pfn), time)  
        variance_downtime = expected_square_downtime - expected_downtime ** 2
        standard_deviation = np.sqrt(variance_downtime) 
        
        #Final output in days (not converted to hours unlike WS and floods)
        mean_downtime_days = expected_downtime
        stdev_downtime_days = standard_deviation

        data_df.loc[i, 'mean_downtime_power'] = mean_downtime_days
        data_df.loc[i, 'std_downtime_power'] = stdev_downtime_days
        
    return data_df

# Compute Recovery Given Sa
def get_recovery_curve_roads(data_df: pd.DataFrame):

    time = np.arange(0.0, 10.0, 1.0)
    
    DSs = [1] #All damage states
    
    data_df['time'] = None
    data_df['recov'] = None

    data_df['mean_downtime'] = 0
    data_df['std_downtime'] = 0
    
    data_df = data_df.astype({'time': 'object',
                             'recov': 'object'})
    
    for i, row in data_df.iterrows():
        
        Pf = [0]*2 #presizing including the no damage state
        Pf[0] = row['DS_0'] #None
        Pf[1] = row['DS_1'] #DS1
        
        Pfn = Pf[0] 
        for ds in DSs:
            mean = restoration_road['DS_1']['mu']
            std = restoration_road['DS_1']['std']
            recov_given_failure = norm.cdf(time, mean, std)
            
            Pfn = Pfn + recov_given_failure * Pf[ds]
     
        #reset the variable names so that they are easy to be understood by the ENG team
        time_days = time
        functionality = Pfn
    
        data_df.at[i,'time'] = time
        data_df.at[i,'recov'] = functionality

        #Compute mean downtime
        expected_downtime = np.trapz(1-Pfn,time)  
        
        #Compute standard deviation of downtime
        expected_square_downtime = np.trapz(2*time*(1-Pfn), time)  
        variance_downtime = expected_square_downtime - expected_downtime ** 2
        standard_deviation = np.sqrt(variance_downtime) 
        
        #Final output in days (not converted to hours unlike WS and floods)
        mean_downtime_days = expected_downtime
        stdev_downtime_days = standard_deviation

        data_df.loc[i, 'mean_downtime'] = mean_downtime_days
        data_df.loc[i, 'std_downtime'] = stdev_downtime_days

    return data_df

########################################################################################
################ MONTE CARLO SIMULATION ################################################
########################################################################################

# Monte Carlo simulation for failure analysis
def montecarloFailureSim(data_df, damage_states_keys, numsim):

    Nstates = len(damage_states_keys)

    def get_mcs_samples(row):
        probs = np.array(row[damage_states_keys])
        r = np.random.uniform(0, 1, numsim)
        states = classify_numbers(r, np.cumsum(probs))
        states = list(states)
        return states

    data_df['samples'] = data_df.apply(get_mcs_samples, axis = 1)

    return data_df

########################################################################################
################ NETWORK RESILIENCE ANALYSIS ###########################################
########################################################################################

# Compute graph shortest paths using Dijkstra's algorithm and given weight and store results in a dataframe
def compute_shortest_paths(G, source_nodes, target_nodes, weight = 'length'):
    results = []
    for source in source_nodes:
        for target in target_nodes:
            try:
                # Shortest path
                path = nx.shortest_path(G, source=source, target=target, weight=weight)
                # Shortest path length
                path_length = nx.shortest_path_length(G, source=source, target=target, weight=weight)
                results.append((source, target, np.round(path_length,3)))
            except nx.NetworkXNoPath:
                results.append((source, target, float('inf')))  # No path found

    results = pd.DataFrame(results, columns = ['from', 'to', 'Weight'])
    
    return results

# Compute the number of connected node pairs in the given graph
def count_connected_pairs(G):
    N = len(G)
    if N < 2:
        return 0  # No pairs in a single-node or empty graph

    # Total possible node pairs
    total_pairs = N * (N - 1) // (1 if G.is_directed() else 2)

    # Count disconnected pairs
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))  # Use weakly_connected_components if needed
    else:
        components = list(nx.connected_components(G))

    disconnected_pairs = sum(len(comp1) * len(comp2) for i, comp1 in enumerate(components) for comp2 in components[i+1:])

    # Connected pairs
    connected_pairs = total_pairs - disconnected_pairs

    frac_connected_pairs = connected_pairs/total_pairs

    return frac_connected_pairs
    

