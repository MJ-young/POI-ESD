import networkx as nx
import math
import random

def calculate_distance(server1, server2):
    """计算两个服务器之间的距离"""
    return math.sqrt((server1.location[0] - server2.location[0])**2 + (server1.location[1] - server2.location[1])**2)

def build_edge_network_topology(edge_servers, max_avg_neighbors=3):
    """
    根据新策略构建边缘网络拓扑图。
    edge_servers: 边缘服务器列表。
    max_avg_neighbors: 平均每个服务器最多连接的邻居个数。
    """
    G = nx.Graph()
    
    # 添加节点
    for server in edge_servers:
        G.add_node(server.server_id, pos=server.location)
    
    # 为每个服务器分配初始的邻居连接数（随机选择一部分最大限制）
    initial_neighbors = max(1, max_avg_neighbors // 2)
    
    # 服务器之间的距离列表
    distances = []
    for i, server1 in enumerate(edge_servers):
        for j, server2 in enumerate(edge_servers[i+1:], start=i+1):
            distance = calculate_distance(server1, server2)
            distances.append((distance, server1.server_id, server2.server_id))
    
    # 根据距离排序，以便后续操作
    distances.sort(key=lambda x: x[0])
    
    # 随机连接邻居
    for server in edge_servers:
        possible_neighbors = [d for d in distances if server.server_id in d[1:]]
        random.shuffle(possible_neighbors)  # 打乱可能的邻居列表，以便随机选择
        
        # 从可能的邻居中选择一部分进行连接
        for _, server1_id, server2_id in possible_neighbors[:initial_neighbors]:
            if G.degree(server1_id) < max_avg_neighbors and G.degree(server2_id) < max_avg_neighbors:
                G.add_edge(server1_id, server2_id)
    
    # 获取邻接矩阵
    adjacency_matrix = nx.adjacency_matrix(G).todense()
                
    return G