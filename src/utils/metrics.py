import numpy as np
import pandas as pd
import math
from .matchBen import calculate_matching_benefit
from .coverBen import calculate_cover_benefit

def calculate_metrics(environment,service,deployment_plan):
    """
    输入待部署场景和部署策略，输出各项指标值，包括匹配度收益、全局覆盖收益、总体收益、部署成本、单位成本总收益等
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param deployment_plan: 部署方案。
    :return: 匹配度收益、全局覆盖收益、总体收益、部署成本、单位成本总收益等指标值。
    """
    server_ids = list(environment.edge_servers.keys())
    total_cost = 0
    match_revenue = 0
    cover_revenue = 0
    hop_revenue = 0
    rho_pois = set()
    if not deployment_plan:
        return 0,0,0,0,0,0,0,rho_pois
    for server_id in server_ids:
        server = environment.edge_servers[server_id]
        if deployment_plan[server_id] == 1:
            rho_pois.update(server.rho_pois)
            total_cost += server.price_per_unit * service.size
            # match_revenue += server.match_revenue
            match_revenue += calculate_matching_benefit(environment,server_id)
            hop_revenue += server.cover_revenue

    # 计算部署策略折损覆盖收益
    # cover_revenue = calculate_cover_revenue(environment,deployment_plan)
    cover_revenue, all_cover = calculate_cover_benefit(environment,deployment_plan) 
    # 0跳收益
    zero_hop_revenue = all_cover - cover_revenue
    # 计算总体收益
    # estimated_revenue = match_revenue*environment.alpha + cover_revenue*environment.beta
    # estimated_revenue = math.sqrt(match_revenue * cover_revenue)
 
    # 精准匹配质量
    deploy_num = sum(deployment_plan.values())
    match_revenue = match_revenue / deploy_num if deploy_num > 0 else 0

    # 计算公平性收益
    estimated_revenue = 2*match_revenue*(all_cover/60)/(match_revenue+cover_revenue) if match_revenue+cover_revenue > 0 else 0
    # estimated_revenue = match_revenue + cover_revenue
    cover_pois, cover_poi_num = calculate_cover_poi_num(environment,deployment_plan)
    high_rels_num = calculate_high_rels_num(environment,cover_pois)
    
    return match_revenue, cover_revenue, estimated_revenue, total_cost ,cover_poi_num, hop_revenue, high_rels_num, rho_pois, all_cover, zero_hop_revenue


def calculate_cover_revenue(environment,deployment_plan):
    """
    计算部署策略折损覆盖收益
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param deployment_plan: 部署方案。
    :return: 每个折损层的poi排序加权之后加和的收益。
    """
    global_match_revenue = 0
    deployment_server_ids = [sid for sid in deployment_plan if deployment_plan[sid] == 1]
    deployment_num = len(deployment_server_ids)
    print('=='*20)
    print(deployment_server_ids,deployment_num)
    if deployment_num == 0:
        return 0
    for h in range(environment.DT+1):
    # for h in range(1,environment.DT+1):
        h_pids = [pid for pid,poi in environment.pois.items() if min(poi.shortest_hops.get(sid,float('inf')) for sid in deployment_server_ids) == h]
        pois_rel_dict = {pid: environment.pois[pid].correlation * (environment.gamma ** h) for pid in h_pids}
        global_match_revenue +=  calculate_matching_benefit(pois_rel_dict)
    return global_match_revenue * deployment_num

def calculate_cover_poi_num(environment,deployment_plan):
    """
    计算部署策略覆盖POI数量
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param deployment_plan: 部署方案。
    :return: 覆盖POI数量。
    """
    cover_pois = set()
    deloyment_server_ids = [sid for sid in deployment_plan if deployment_plan[sid] == 1]
    if not deloyment_server_ids:
        return cover_pois,0
    for pid,poi in environment.pois.items():
        min_hops = min(poi.shortest_hops.get(sid,float('inf')) for sid in deloyment_server_ids)
        if min_hops <= environment.DT:
            cover_pois.add(pid)
    return cover_pois,len(cover_pois)

def calculate_high_rels_num(environment,cover_pois):
    """
    计算部署策略覆盖高相关性POI数量
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param cover_pois: 覆盖POI集合。
    :return: 覆盖高相关性POI数量。
    """
    # 统计rho_pois中的相关性超过75分位数的POI
    # 读取environment.poi_info中的rel,是第三个元素
    rels = [poi[2] for poi in environment.pois_info]
    rels = np.array(rels)
    rels = np.sort(rels)
    # 75分位数
    threshold = np.percentile(rels, 70)
    print('threshold:',threshold)
    high_rels_pois = [pid for pid in cover_pois if environment.pois[pid].correlation >= threshold]
    return len(high_rels_pois)