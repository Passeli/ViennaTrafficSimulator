import osmnx as ox
import geopandas as gpd
import networkx as nx
import json
import struct

# ==========================================
# 1. SETUP & MAP DOWNLOAD
# ==========================================
print("1. Loading Vienna Polygon and downloading street network...")
apocalypse_filter = (
    '["area"!~"yes"]'
    '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|service|living_street|track|construction"]'
)

# Load the polygon you created
vienna_combined_gdf = gpd.read_file("vienna_combined.geojson")
vienna_combined_polygon = vienna_combined_gdf.geometry.union_all()

# Download the graph
G = ox.graph_from_polygon(vienna_combined_polygon, custom_filter=apocalypse_filter)

# ==========================================
# 1.5. PRE-CALCULATE TRAVEL TIMES
# ==========================================
print("1.5. Cleaning speeds and calculating travel times for fastest routing...")

for u, v, key, data in G.edges(keys=True, data=True):
    # --- Clean maxspeed ---
    max_speed_kmh = data.get('maxspeed', 50.0)

    if isinstance(max_speed_kmh, list):
        max_speed_kmh = max(map(float, max_speed_kmh))
    elif isinstance(max_speed_kmh, str):
        max_speed_kmh = 10.0 if max_speed_kmh.lower() == 'walk' else float(max_speed_kmh)

    max_speed_ms = max_speed_kmh / 3.6

    # Save the cleaned speed back to the graph
    data['max_speed_ms'] = max_speed_ms

    # Calculate travel time in seconds: Time = Distance / Speed
    length = float(data.get('length', 1.0))
    data['travel_time'] = length / max_speed_ms

# ==========================================
# 2. DIJKSTRA ROUTING (FASTEST PATH)
# ==========================================
print("2. Calculating fastest Dijkstra routes to exit nodes...")
exit_nodes = [
    10262271241, 2490526838, 291465820, 291189453, 60658616,
    8111487139, 9847098826, 1595333305, 7987059451, 73978867,
    1917025540, 127619206, 620959, 61861493, 92420972, 620993,
    621022, 621033, 9306538374, 6828939513, 60214544, 29006505,
    518494754, 127637242, 144786481, 2066084081, 29006459, 21584324,
    309895678, 7223381116, 602215461, 815629460, 140452737, 140452972,
    110328669, 140453333, 110328359, 110277054, 337564997, 602228158,
    29667403, 970706685, 1780418469, 6813429057, 945996008, 305183428,
    302772812, 6744282110, 6604011411, 117676269, 5866530260, 3811670788,
    316283227, 3578797943, 316283399, 98685755,

    127773769, 90192169,
    112381139, 90190491, 112377495, 112372340, 129942361, 129942360,
    283550896, 31437581, 1392925568, 316373422, 286298588, 392273855,
    174881967, 282534201, 1475372855, 59993684, 59993752, 59993688,
    59993691, 57735685, 253439159, 60016851, 60016855, 60016858,
    60016859, 60016860, 1973858591, 59860668, 59836928, 59836878,
    59836843, 59836837, 59836834, 59836819, 59836815, 59836793,
    59836788, 59836729, 59836730, 59836733, 59836736, 59836739,
    59836742, 59836961, 59836962, 247561921, 59836971, 175449509,
    4824551684, 9900794, 175450473, 276112831, 59843116, 1859311891,
    129944529, 59843118, 386026776, 2535111699, 2505961258, 59989570,
    81605395, 81625281, 9217237142, 6282971918, 9829196616, 130069703,
    59372041
]

G_reversed = G.reverse(copy=True)
# --- CHANGED WEIGHT TO 'travel_time' ---
lengths, paths = nx.multi_source_dijkstra(G_reversed, sources=exit_nodes, weight='travel_time')

# Initialize all nodes to -1
nx.set_node_attributes(G, -1, 'next_node_to_exit')

# Apply paths to graph
for node, path in paths.items():
    if len(path) > 1:
        G.nodes[node]['next_node_to_exit'] = path[-2]  # The next step towards the exit
    elif len(path) == 1:
        G.nodes[node]['next_node_to_exit'] = -1  # It IS the exit
    else:
        raise RuntimeError("Path is empty")

# ==========================================
# 3. PARKING & CAR SPAWN CAPACITY
# ==========================================
print("3. Fetching parking and building features...")

accommodation_buildings = ['apartments', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe', 'dormitory', 'farm',
                           'ger', 'hotel', 'house', 'houseboat', 'residential', 'semidetached_house', 'static_caravan',
                           'stilt_house', 'terrace', 'tree_house', 'trullo']
cars_buildings = ['carport', 'garage', 'garages', 'parking']

# Expanded dictionary to capture ALL residential building types
parking_tags = {
    'amenity': ['parking', 'parking_space'],
    'building': accommodation_buildings + cars_buildings
}
parking_features = ox.features_from_polygon(vienna_combined_polygon, tags=parking_tags)

print("-> Exporting raw OSM features to GeoJSON for visualization...")

# 1. Make a copy so we don't accidentally ruin the simulation data
export_gdf = parking_features.copy()

# 2. Pick only the columns you care about seeing in your map viewer
columns_to_keep = ['geometry', 'amenity', 'building', 'parking']

# 3. Safely filter the dataframe (in case 'amenity' or 'building' didn't download)
existing_columns = [col for col in columns_to_keep if col in export_gdf.columns]
export_gdf = export_gdf[existing_columns]

# 4. The Magic Fix: Force all non-geometry columns into strings to prevent GeoJSON crashes
for col in export_gdf.columns:
    if col != 'geometry':
        export_gdf[col] = export_gdf[col].astype(str)

# 5. Export!
export_gdf.to_file("vienna_buildings_and_parking.geojson", driver='GeoJSON')
print("✅ Saved vienna_buildings_and_parking.geojson!")

# Project to UTM Zone 33N for Vienna to calculate accurate metric areas and lengths
parking_features_proj = parking_features.to_crs(epsg=32633)

print("-> Applying explicit capacity rules based on tags and geometry...")

valid_parking = parking_features_proj.copy()
valid_parking['capacity'] = 0.0

# Pre-calculate geometry types, areas, and lengths for fast math
geom_types = valid_parking.geometry.geom_type
area = valid_parking.geometry.area
lengths = valid_parking.geometry.length

# --- CRASH-PROOF COLUMN EXTRACTION ---
if 'amenity' not in valid_parking.columns: valid_parking['amenity'] = ''
if 'building' not in valid_parking.columns: valid_parking['building'] = ''
if 'parking' not in valid_parking.columns: valid_parking['parking'] = ''

amenity_col = valid_parking['amenity'].fillna('').astype(str)
building_col = valid_parking['building'].fillna('').astype(str)
parking_col = valid_parking['parking'].fillna('').astype(str)
# -------------------------------------

# Create specific boolean masks
is_parking_lot = amenity_col == 'parking'
is_parking_space = amenity_col == 'parking_space'

# Underground/Multi-storey garages
is_underground_or_multi = parking_col.isin(['underground', 'multi-storey'])
is_surface_lot = is_parking_lot & ~is_underground_or_multi

is_garage_single = building_col.isin(['garage', 'carport'])
is_garages_row = building_col == 'garages'

# Catch the lazy parking decks! (amenity=parking AND building=yes)
is_lazy_parking_deck = (amenity_col == 'parking') & (building_col == 'yes')
is_parking_deck = (building_col == 'parking') | is_lazy_parking_deck

# Ensure lazy parking decks are NOT accidentally counted as residential 'yes' buildings
is_residential = building_col.isin(accommodation_buildings + ['yes']) & ~is_lazy_parking_deck

# ---------------------------------------------------------
# RULE 1: POLYGONS (Areas)
# ---------------------------------------------------------
is_poly = geom_types.isin(['Polygon', 'MultiPolygon'])

# All Houses / Apartments / Residential -> Area / 50 (NO Vertical Density Tweak)
valid_parking.loc[is_poly & is_residential, 'capacity'] = area / 50.0

# Surface Parking Lots & 1-Story Garages -> Area / 25
valid_parking.loc[is_poly & (is_surface_lot | is_garages_row), 'capacity'] = area / 25.0

# Multi-Story & Underground Parking Decks -> Area / 5
valid_parking.loc[is_poly & (is_parking_deck | is_underground_or_multi), 'capacity'] = area / 5.0

# Single Garage & Single Parking Space -> Flat rate of 1 car
valid_parking.loc[is_poly & (is_garage_single | is_parking_space), 'capacity'] = 1.0

# ---------------------------------------------------------
# RULE 2: LINES (Lengths - mostly parallel street parking)
# ---------------------------------------------------------
is_line = geom_types.isin(['LineString', 'MultiLineString'])

# Any parking mapped as a line -> 1 car per 5 meters
valid_parking.loc[is_line & (is_parking_lot | is_parking_space), 'capacity'] = lengths / 5.0

# ---------------------------------------------------------
# RULE 3: POINTS (Nodes with no physical dimensions)
# ---------------------------------------------------------
is_point = geom_types.isin(['Point', 'MultiPoint'])

# A point tagged simply as 'parking' -> 10 cars
valid_parking.loc[is_point & is_parking_lot, 'capacity'] = 10.0

# A point tagged as a single space or single garage -> 1 car
valid_parking.loc[is_point & (is_parking_space | is_garage_single), 'capacity'] = 1.0

# ---------------------------------------------------------
# CLEANUP & TOTAL CALCULATION
# ---------------------------------------------------------
# Convert to integer (this rounds down, e.g., 2.9 becomes 2)
valid_parking['capacity'] = valid_parking['capacity'].fillna(0).astype(int)

# Filter out the zeroes
valid_parking = valid_parking[valid_parking['capacity'] > 0]

# --- THE GRAND TOTAL ---
total_spawn_capacity = valid_parking['capacity'].sum()
print(f"🔥 TOTAL CARS GENERATED: {total_spawn_capacity:,} 🔥")
# -----------------------

# To find nearest edges, we need centroids in Lat/Lon
print("-> Snapping features to the street graph...")
centroids_latlon = valid_parking.geometry.centroid.to_crs(epsg=4326)

# Vectorized nearest-edge lookup
nearest_edges = ox.distance.nearest_edges(G, centroids_latlon.x, centroids_latlon.y)

# Aggregate the capacities onto the specific edges
edge_spawn_capacities = {}
for i, (u, v, key) in enumerate(nearest_edges):
    edge_id = f"{u}_{v}_{key}"
    edge_spawn_capacities[edge_id] = edge_spawn_capacities.get(edge_id, 0) + valid_parking.iloc[i]['capacity']

# ==========================================
# 4. CLEAN DATA & PREPARE EXPORTS
# ==========================================
print("4. Formatting JSONs...")
nodes_export = []
edges_export = []

# Process Nodes
for node_id, data in G.nodes(data=True):
    nodes_export.append({
        "id": node_id,
        "x": data['x'],
        "y": data['y'],
        "next_node_to_exit": data.get('next_node_to_exit', -1)
    })

# Process Edges
for u, v, key, data in G.edges(keys=True, data=True):
    edge_id = f"{u}_{v}_{key}"
    length = data['length']
    max_speed_ms = data['max_speed_ms']  # Calculated in Step 1.5!

    # Fetch the spawn capacity we calculated in Step 3
    spawn_capacity = edge_spawn_capacities.get(edge_id, 0)
    G.edges[u, v, key]['spawn_capacity'] = spawn_capacity

    edges_export.append({
        "u": int(u),
        "v": int(v),
        "length": float(length),
        "max_speed_ms": float(max_speed_ms),
        "spawn_capacity": int(spawn_capacity)
    })

# ==========================================
# 5. SAVE FILES
# ==========================================
print("5. Saving all output files...")

# --- Save Flat JSONs for Engine ---
with open("vienna_nodes.json", "w") as f:
    json.dump(nodes_export, f)

with open("vienna_edges.json", "w") as f:
    json.dump(edges_export, f)

# --- Save GeoJSONs for QGIS ---
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

gdf_nodes.to_file("vienna_nodes.geojson", driver='GeoJSON')
gdf_edges.to_file("vienna_edges.geojson", driver='GeoJSON')

exit_nodes_gdf = gdf_nodes[gdf_nodes.index.isin(exit_nodes)]
exit_nodes_gdf.to_file("vienna_exit_nodes.geojson", driver='GeoJSON')

print("✅ Complete! You now have fully processed data ready for simulation.")

print("\n--- INITIATING VULKAN DATA COMPILATION ---")

# ==========================================
# 1. ID MAPPING (64-bit to 0-Indexed Arrays)
# ==========================================
print("1. Compiling 0-indexed memory maps...")

node_to_index = {}
edge_to_index = {}
index_to_edge = {}

uv_to_first_edge_index = {}

# Map Nodes
for i, node_id in enumerate(G.nodes()):
    node_to_index[node_id] = i

# Map Edges (WITH KEY!)
for i, (u, v, key) in enumerate(G.edges(keys=True)):
    edge_id = f"{u}_{v}_{key}"
    edge_to_index[edge_id] = i
    index_to_edge[i] = (u, v, key)

    uv_str = f"{u}_{v}"
    if uv_str not in uv_to_first_edge_index:
        uv_to_first_edge_index[uv_str] = i

num_nodes = len(node_to_index)
num_edges = len(edge_to_index)

print(f"   -> Mapped {num_nodes:,} Nodes and {num_edges:,} Edges.")

# ==========================================
# 2. PRE-COMPUTE GPU ROUTING (The "Next Edge" trick)
# ==========================================
print("2. Baking Dijkstra routes directly into Edge memory...")

edge_next_indices = {}

for i in range(num_edges):
    u, v, key = index_to_edge[i]
    end_node_id = v

    next_node_id = G.nodes[end_node_id].get('next_node_to_exit', -1)

    if next_node_id == -1:
        edge_next_indices[i] = -1
    else:
        uv_str = f"{end_node_id}_{next_node_id}"
        edge_next_indices[i] = uv_to_first_edge_index.get(uv_str, -1)

# ==========================================
# 3. EXPORT RAW BINARY FILES FOR C++
# ==========================================
print("3. Packing flat binary files for Vulkan RWStructuredBuffers...")

# --- A. NODES BINARY (16 Bytes: ffii) ---
print("   -> Packing Nodes...")
with open("vulkan_nodes.bin", "wb") as f:
    for node_id in G.nodes():
        data = G.nodes[node_id]
        x = float(data['x'])
        y = float(data['y'])
        lock = -1
        padding = 0
        f.write(struct.pack('ffii', x, y, lock, padding))

# --- B. EDGES BINARY (32 Bytes: iiiffiii) ---
print("   -> Packing Edges...")
with open("vulkan_edges.bin", "wb") as f:
    for i in range(num_edges):
        u, v, key = index_to_edge[i]
        data = G.edges[u, v, key]

        start_node_idx = node_to_index[u]
        end_node_idx = node_to_index[v]
        next_edge_idx = edge_next_indices[i]
        length = data['length']
        max_speed_ms = data['max_speed_ms']  # Fetched clean

        f.write(struct.pack('iiiffiii',
                            start_node_idx, end_node_idx, next_edge_idx,
                            float(length), max_speed_ms, -1, 0, 0))

# --- C. CARS BINARY (32 Bytes: iffiiiii) ---
print("   -> Packing Virtual Garages (Cars)...")
total_cars_packed = 0

with open("vulkan_cars.bin", "wb") as f:
    for i in range(num_edges):
        u, v, key = index_to_edge[i]
        spawn_cap = int(G.edges[u, v, key].get('spawn_capacity', 0))

        # Packed out the 100k+ cars using your real capacity!
        for _ in range(spawn_cap):
            edge_idx = i
            position = 0.0
            speed = 0.0
            state = 3  # 3 = In Garage

            f.write(struct.pack('iffiiiii',
                                edge_idx, position, speed, state,
                                -1, 0, 0, 0))
            total_cars_packed += 1

print(f"✅ VULKAN PIPELINE COMPLETE!")
print(f"   Saved {num_nodes} nodes to vulkan_nodes.bin")
print(f"   Saved {num_edges} edges to vulkan_edges.bin")
print(f"   Saved {total_cars_packed:,} cars to vulkan_cars.bin")
