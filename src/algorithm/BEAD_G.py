import time
from utils.rho import calculate_rho

# 迭代选择每次增加被直接或间接覆盖的POI数量最多的服务器
def BEAD_G(environment,service):
    """
    采用贪心算法进行服务部署，优先选择覆盖POI数量最多的服务器。
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param budget: 预算限制。
    :return: 部署方案和实验运行时间。
    """
    start = time.perf_counter()
    server_ids = list(environment.edge_servers.keys())
    total_cost = 0
    deployment_plan = {server_id: 0 for server_id in server_ids}  # 初始化部署方案
    covered_pois = set()

    # 提前计算每个服务器及其邻居覆盖的POI集合
    all_covered_pois = {}
    # for server_id in server_ids:
    #     server = environment.edge_servers[server_id]
    #     self_and_neighbors_covered_pois = set(server.covered_pois)
    #     for neighbor_id in server.neighbors:
    #         self_and_neighbors_covered_pois.update(environment.edge_servers[neighbor_id].covered_pois)
    #     all_covered_pois[server_id] = self_and_neighbors_covered_pois

    rho = calculate_rho(environment)
    # 优化版本：提前计算每个服务器及其邻居覆盖的POI集合,使用rho矩阵
    for server_id in server_ids:
        self_and_neighbors_covered_pois = set()
        for poi_id in environment.pois.keys():
            if rho.get((server_id, poi_id), 0):
                self_and_neighbors_covered_pois.add(poi_id)
        all_covered_pois[server_id] = self_and_neighbors_covered_pois


    while total_cost < service.budget:
        max_covered_pois = 0
        best_server_id = None
        # 选择delta最大的服务器，下面的计算效率不高，可以优化
        # 优化版本：
        # 1. 找出未部署服务的服务器
        # 2. 计算每个服务器的delta
        # 3. 选择delta最大的服务器
        unselected_server_ids = [server_id for server_id in server_ids if deployment_plan[server_id] == 0]
        for server_id in unselected_server_ids:
            new_covered_pois = len(all_covered_pois[server_id] - covered_pois)
            if new_covered_pois > max_covered_pois and total_cost + environment.edge_servers[server_id].price_per_unit * service.size <= service.budget:
                max_covered_pois = new_covered_pois
                best_server_id = server_id
        if best_server_id is not None:
            deployment_plan[best_server_id] = 1  # 部署服务
            total_cost += environment.edge_servers[best_server_id].price_per_unit * service.size
            covered_pois.update(all_covered_pois[best_server_id])
        else:
            break
    end = time.perf_counter()
    run_time = (end - start)*1000
    return deployment_plan, run_time