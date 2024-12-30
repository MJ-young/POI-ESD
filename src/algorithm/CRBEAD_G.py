# Joint Coverage-Reliability for Budgeted Edge Application Deployment in Mobile Edge Computing Environment[J]. IEEE Transactions on Parallel and Distributed Systems, 2022, 33(12): 3760–3771. 

import time
from utils.rho import calculate_rho


def CRBEAD_G(environment, service):
    # 为函数计时
    start = time.perf_counter()

    beta = 0.5  # 覆盖和可靠性的权重
    deployed_servers = set()  # 已部署服务器集合
    total_coverage = set()  # 总覆盖的POI集合
    total_cb_size = 0  # 总覆盖和可靠性收益
    improvement = True  # 收益是否还在增加
    remaining_budget = service.budget  # 剩余预算

    rho = calculate_rho(environment) # 计算rho矩阵
    
    while improvement:
        improvement = False
        best_increase = 0
        best_server = None
        
        for server_id in environment.edge_servers.keys():
            if server_id in deployed_servers:
                continue  # 跳过已部署的服务器

            # 计算部署该服务器的成本
            deployment_cost = environment.edge_servers[server_id].price_per_unit * service.size
            if deployment_cost > remaining_budget:
                continue  # 跳过超出预算的服务器
                
            # 计算部署该服务器带来的增量
            new_coverage = set()
            for poi_id in environment.pois.keys():
                if rho.get((server_id, poi_id), 0):
                    new_coverage.add(poi_id)
            
            # 计算增量的覆盖和可靠性收益
            new_total_coverage = total_coverage.union(new_coverage)
            increase = (1 - 2 * beta) * (len(new_total_coverage) - len(total_coverage)) + beta * len(new_coverage)
            
            # 选择增量最大的服务器
            if increase > best_increase:
                best_increase = increase
                best_server = server_id
        
        # 如果找到了增量最大的服务器，则部署该服务器
        if best_server is not None and best_increase > 0:
            deployed_servers.add(best_server)
            for poi_id in environment.pois.keys():
                if rho.get((best_server, poi_id), 0):
                    total_coverage.add(poi_id)
            total_cb_size += best_increase
            improvement = True
            remaining_budget -= environment.edge_servers[best_server].price_per_unit * service.size
    
    deployment_plan = {sid: 1 if sid in deployed_servers else 0 for sid in environment.edge_servers.keys()}

    end = time.perf_counter()
    run_time = (end - start)*1000

    
    return deployment_plan, run_time