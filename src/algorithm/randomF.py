import random
import time

def random_deployment(environment, service):
    """
    随机选择边缘服务器进行服务部署。
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param service: Service对象。
    :return: 随机部署方案和估算的收益。
    """
    start = time.perf_counter()
    server_ids = list(environment.edge_servers.keys())
    random.shuffle(server_ids)  # 随机打乱服务器顺序
    
    total_cost = 0
    deployment_plan = {}
    
    for server_id in server_ids:
        server = environment.edge_servers[server_id]
        if total_cost + server.price_per_unit * service.size <= service.budget:
            deployment_plan[server_id] = 1  # 部署服务
            total_cost += server.price_per_unit * service.size
        else:
            deployment_plan[server_id] = 0  # 不部署服务
    end = time.perf_counter()
    run_time = (end - start)*1000

    return deployment_plan, run_time