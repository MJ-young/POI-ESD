import networkx as nx

# 计算poi能否在DT跳内被覆盖的变量rho
def calculate_rho(environment):
    rho = {}
    for server_id, server in environment.edge_servers.items():
        short_hops = nx.single_source_shortest_path_length(environment.network_topology, server_id, cutoff=environment.DT)
        dT_neighbors = set(short_hops.keys())
        dT_pois = set()
        for neighbor_id in dT_neighbors:
            dT_pois.update(environment.edge_servers[neighbor_id].covered_pois)
        for poi_id in environment.pois.keys():
            rho[(server_id, poi_id)] = 1 if poi_id in dT_pois else 0
    return rho