from gurobipy import Model, GRB, quicksum
import time
from utils.rho import calculate_rho

def RBEAD(environment,service):
    # 为函数计时
    start = time.perf_counter()

    rho = calculate_rho(environment)

    # 初始化模型
    model = Model("MaximizeReliabilityGain")
    
    # 决策变量：服务器是否部署服务
    x = model.addVars(environment.edge_servers.keys(), vtype=GRB.BINARY, name="x")

    # 定义每个POI被覆盖的辅助变量
    cover = model.addVars(environment.pois.keys(), vtype=GRB.BINARY, name="covered")

    # 约束：如果POI被至少一个部署的服务器覆盖，则covered[poi]必须为1
    for poi in environment.pois.keys():
        model.addConstr(cover[poi] <= quicksum(x[si] * rho.get((si, poi), 0) for si in environment.edge_servers.keys()))
        # 确保如果POI被覆盖，则covered[poi]为1
        model.addConstr(quicksum(x[si] * rho.get((si, poi), 0) for si in environment.edge_servers.keys()) <= cover[poi] * len(environment.edge_servers.keys()))
    
    # 计算每个服务器及其邻居覆盖的POI集合大小 |CB(si)|
    cb_size = {si: sum(rho.get((si, poi), 0) for poi in environment.pois.keys()) for si in environment.edge_servers.keys()}
    
    # 构建目标函数
    individual_covered_size = quicksum(cb_size[si] * x[si] for si in environment.edge_servers.keys())
    joint_covered_size = quicksum(cover[poi] for poi in environment.pois.keys())
    
    # 设置目标函数
    model.setObjective(individual_covered_size - joint_covered_size, GRB.MAXIMIZE)
    
    # 优化模型
    model.optimize()
    
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
        print("Objective value: ", model.objVal)
    else:
        deployment_plan = {server_id: 0 for server_id in environment.edge_servers.keys()}
        print('status: ', model.status)
    # model.write("RBEAD.lp")

    end = time.perf_counter()
    run_time = (end - start)*1000
    
    return deployment_plan, run_time
