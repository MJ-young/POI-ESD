# Optimal Application Deployment in Mobile Edge Computing Environment[C]//2020 IEEE 13th International Conference on Cloud Computing (CLOUD). 
# 考虑当前和邻居服务器可以覆盖的poi集合大小

import time
from gurobipy import Model, GRB, max_, quicksum
import networkx as nx
from utils.rho import calculate_rho

def BEAD(environment, service):
    # 为函数计时
    start = time.perf_counter()
    model = Model("Budgeted Edge Application Deployment")

    # 计算rho矩阵，如果POI在rho_pois，则rho为1，否则为0
    rho = calculate_rho(environment)

    # 决策变量：服务器是否部署服务
    x = model.addVars(environment.edge_servers.keys(), vtype=GRB.BINARY, name="x")

    # 中间变量：服务器或其邻居是否覆盖每个POI
    cover = model.addVars(environment.pois.keys(), vtype=GRB.BINARY, name="cover")

    # POI覆盖情况的约束
    # for poi_id, poi in environment.pois.items():
    #     all_neighbors = set(poi.covered_by)
    #     if len(all_neighbors) == 0:
    #         model.addConstr(cover[poi_id] == 0)
    #         continue
    #     for server_id in poi.covered_by:
    #         all_neighbors.update(environment.edge_servers[server_id].neighbors)
    #     # 为每个POI创建一个约束，确保如果任何一个覆盖它的服务器或邻居部署了服务，则cover[poi_id]为1
    #     model.addConstr(cover[poi_id] <= quicksum(x[server_id] for server_id in all_neighbors))
    #     model.addConstr(cover[poi_id] >= quicksum(x[server_id] for server_id in all_neighbors) / len(all_neighbors))  

    # 约束：如果POI被至少一个部署的服务器覆盖，则covered[poi]必须为1
    for poi in environment.pois.keys():
        model.addConstr(cover[poi] <= quicksum(x[si] * rho.get((si, poi), 0) for si in environment.edge_servers.keys()))
        # 确保如果POI被覆盖，则covered[poi]为1
        model.addConstr(quicksum(x[si] * rho.get((si, poi), 0) for si in environment.edge_servers.keys()) <= cover[poi] * len(environment.edge_servers.keys()))
    


    # 目标：全局覆盖收益
    global_coverage_revenue = quicksum(cover[poi_id] for poi_id in environment.pois)
    # 只最大化全局覆盖收益
    model.setObjective(global_coverage_revenue, GRB.MAXIMIZE)

    # 添加其他必要的约束（例如，预算约束等）
    model.addConstr(quicksum(environment.edge_servers[server_id].price_per_unit * service.size * x[server_id] for server_id in environment.edge_servers) <= service.budget)
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        deployment_plan = model.getAttr('x', x)
        print("Optimal deployment plan:")
        for server_id, deploy in deployment_plan.items():
            if deploy > 0.5:
                print(f"Deploy service on server {server_id}")
    # model.write("model.lp")
    end = time.perf_counter()
    run_time = (end - start)*1000

    return deployment_plan, run_time
