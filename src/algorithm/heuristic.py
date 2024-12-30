# 编写启发式生成初始种群解的算法

import numpy as np
import random

def heuristic(environment, service, solution_num):
    if solution_num <= 0:
        return []
    # 生成初始种群解，每个解是一个部署方案
    server_ids = list(environment.edge_servers.keys())
    match_weights = [environment.edge_servers[server_id].match_revenue for server_id in server_ids]
    cover_weights = [len(environment.edge_servers[server_id].covered_pois) for server_id in server_ids]

    # 生成solution_num个部署方案,在预算约束下，按照匹配度和覆盖度的权重随机选择服务器
    solutions = []
    mean_num = service.budget / 1
    # mean_num = service.budget / environment.mean_price_per_unit
    for _ in range(solution_num):
        total_cost = 0
        deployment_plan = []
        while total_cost < service.budget:
            if random.random() < 0.5:
                server_id = random.choices(server_ids, weights=match_weights)[0]
            else:
                server_id = random.choices(server_ids, weights=cover_weights)[0]
            if server_id in deployment_plan:
                continue
            server = environment.edge_servers[server_id]
            deployment_plan.append(server_id)
            total_cost += server.price_per_unit * service.size
        # 判断是否超过预算，如果超过预算则随机删除直到满足预算
        # 将部署方案按照价格升序排序
        deployment_plan.sort(key=lambda x: -environment.edge_servers[x].price_per_unit)
        while total_cost > service.budget:
            server_id = deployment_plan.pop()
            server = environment.edge_servers[server_id]
            total_cost -= server.price_per_unit * service.size
        
        solutions.append(deployment_plan)
    return solutions

def heuristicM(environment, service, solution_num):
    if solution_num <= 0:
        return []
    # 生成初始种群解，每个解是一个部署方案
    server_ids = list(environment.edge_servers.keys())
    match_weights = [environment.edge_servers[server_id].match_revenue for server_id in server_ids]
    cover_weights = [len(environment.edge_servers[server_id].rho_pois.difference(environment.edge_servers[server_id].covered_pois)) for server_id in server_ids]

    # 生成solution_num个部署方案,在预算约束下，按照匹配度和覆盖度的权重随机选择服务器
    solutions = []
    mean_num = service.budget / 1
    # mean_num = service.budget / environment.mean_price_per_unit
    for _ in range(solution_num):
        total_cost = 0
        deployment_plan = []
        while total_cost < service.budget:
            if random.random() < 0.5:
                server_id = random.choices(server_ids, weights=match_weights)[0]
            else:
                server_id = random.choices(server_ids, weights=cover_weights)[0]
            if server_id in deployment_plan:
                continue
            server = environment.edge_servers[server_id]
            deployment_plan.append(server_id)
            total_cost += server.price_per_unit * service.size
        # 判断是否超过预算，如果超过预算则随机删除直到满足预算
        # 将部署方案按照价格升序排序
        deployment_plan.sort(key=lambda x: -environment.edge_servers[x].price_per_unit)
        while total_cost > service.budget:
            server_id = deployment_plan.pop()
            server = environment.edge_servers[server_id]
            total_cost -= server.price_per_unit * service.size
        
        solutions.append(deployment_plan)
    return solutions

def heuristic_cover(environment, service, solution_num):
    if solution_num <= 0:
        return []
    # 生成初始种群解，每个解是一个部署方案
    server_ids = list(environment.edge_servers.keys())
    cover_weights = [len(environment.edge_servers[server_id].covered_pois) for server_id in server_ids]

    # 生成solution_num个部署方案,在预算约束下，按照匹配度和覆盖度的权重随机选择服务器
    solutions = []
    for _ in range(solution_num):
        total_cost = 0
        deployment_plan = []
        while total_cost < service.budget:
            server_id = random.choices(server_ids, weights=cover_weights)[0]
            if server_id in deployment_plan:
                continue
            server = environment.edge_servers[server_id]
            deployment_plan.append(server_id)
            total_cost += server.price_per_unit * service.size
        # 判断是否超过预算，如果超过预算则随机删除直到满足预算
        # 将部署方案按照价格升序排序
        deployment_plan.sort(key=lambda x: -environment.edge_servers[x].price_per_unit)
        while total_cost > service.budget:
            server_id = deployment_plan.pop()
            server = environment.edge_servers[server_id]
            total_cost -= server.price_per_unit * service.size
        
        solutions.append(deployment_plan)
    return solutions

def heuristic_match(environment, service, solution_num):
    if solution_num <= 0:
        return []
    # 生成初始种群解，每个解是一个部署方案
    server_ids = list(environment.edge_servers.keys())
    match_weights = [environment.edge_servers[server_id].match_revenue for server_id in server_ids]

    # 生成solution_num个部署方案,在预算约束下，按照匹配度和覆盖度的权重随机选择服务器
    solutions = []
    for _ in range(solution_num):
        total_cost = 0
        deployment_plan = []
        while total_cost < service.budget:
            server_id = random.choices(server_ids, weights=match_weights)[0]
            if server_id in deployment_plan:
                continue
            server = environment.edge_servers[server_id]
            deployment_plan.append(server_id)
            total_cost += server.price_per_unit * service.size
        # 判断是否超过预算，如果超过预算则随机删除直到满足预算
        # 将部署方案按照价格升序排序
        deployment_plan.sort(key=lambda x: -environment.edge_servers[x].price_per_unit)
        while total_cost > service.budget:
            server_id = deployment_plan.pop()
            server = environment.edge_servers[server_id]
            total_cost -= server.price_per_unit * service.size
        
        solutions.append(deployment_plan)
    return solutions

def heuristic_in_cover(environment, service, solution_num):
    if solution_num <= 0:
        return []
    # 生成初始种群解，每个解是一个部署方案
    server_ids = list(environment.edge_servers.keys())
    cover_weights = [len(environment.edge_servers[server_id].rho_pois.difference(environment.edge_servers[server_id].covered_pois)) for server_id in server_ids]

    # 生成solution_num个部署方案,在预算约束下，按照匹配度和覆盖度的权重随机选择服务器
    solutions = []
    for _ in range(solution_num):
        total_cost = 0
        deployment_plan = []
        while total_cost < service.budget:
            server_id = random.choices(server_ids, weights=cover_weights)[0]
            if server_id in deployment_plan:
                continue
            server = environment.edge_servers[server_id]
            deployment_plan.append(server_id)
            total_cost += server.price_per_unit * service.size
        # 判断是否超过预算，如果超过预算则随机删除直到满足预算
        # 将部署方案按照价格升序排序
        deployment_plan.sort(key=lambda x: -environment.edge_servers[x].price_per_unit)
        while total_cost > service.budget:
            server_id = deployment_plan.pop()
            server = environment.edge_servers[server_id]
            total_cost -= server.price_per_unit * service.size
        
        solutions.append(deployment_plan)
    return solutions
