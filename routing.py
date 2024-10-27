from geopy.distance import geodesic
import itertools
import networkx as nx

def estimate_travel_time(vol_a, vol_b, dist, speed_limit=60, intersection_delay=30):
    avg_volume = (vol_a + vol_b) / 2
    base_time = dist / (speed_limit / 60)
    total_time = base_time + intersection_delay + (avg_volume / 100)
    return total_time

def build_graph(data, vol_data):
    G = nx.Graph()
    for (i, row_a), (j, row_b) in itertools.combinations(data.iterrows(), 2):
        # Only consider pairs of different nodes to avoid self-loops
        if row_a['SCATS Number'] != row_b['SCATS Number']:
            dist = geodesic((row_a['NB_LATITUDE'], row_a['NB_LONGITUDE']),
                            (row_b['NB_LATITUDE'], row_b['NB_LONGITUDE'])).km
            if dist <= 1:
                travel_time = estimate_travel_time(vol_data[i], vol_data[j], dist)
                G.add_edge(row_a['SCATS Number'], row_b['SCATS Number'], weight=travel_time)
    
    # Print all nodes and edges to verify no self-loops remain
    print("Graph nodes:", list(G.nodes))
    print("Graph edges:", list(G.edges(data=True)))
    return G


def find_routes(G, origin, destination):
    # Ensure origin and destination are integers to match node types
    origin = int(origin)
    destination = int(destination)
    
    # Check if both nodes are in the graph
    if not G.has_node(origin):
        print(f"Error: Origin node {origin} not found in graph.")
        return []
    if not G.has_node(destination):
        print(f"Error: Destination node {destination} not found in graph.")
        return []
    
    try:
        routes = list(nx.shortest_simple_paths(G, source=origin, target=destination, weight='weight'))[:5]
    except nx.NetworkXNoPath:
        print(f"No path between {origin} and {destination}")
        routes = []
    return routes