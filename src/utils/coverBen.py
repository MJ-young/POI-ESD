from .matchBen import calculate_matching_benefit

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
    if deployment_num == 0:
        return 0
    for h in range(1,environment.DT+1):
        h_pids = [pid for pid,poi in environment.pois.items() if min(poi.shortest_hops.get(sid,float('inf')) for sid in deployment_server_ids) == h]
        pois_rel_dict = {pid: environment.pois[pid].correlation * (environment.gamma ** h) for pid in h_pids}
        global_match_revenue +=  calculate_matching_benefit(pois_rel_dict)
    return global_match_revenue * deployment_num

def calculate_cover_benefit(environment,deployment_plan):
    """
    计算部署策略覆盖收益
    
    :param environment: EdgeDeploymentEnvironment对象。
    :param deployment_plan: 部署方案。
    :return: 折损之后的poi相关性加和的收益。
    """
    deployment_server_ids = [sid for sid in deployment_plan if deployment_plan[sid] == 1]
    deployment_num = len(deployment_server_ids)
    pois_rel_dict = {}
    if deployment_num == 0:
        return 0, 0
    for h in range(1,environment.DT+1):
    # for h in range(environment.DT+1):
        h_pids = [pid for pid,poi in environment.pois.items() if min(poi.shortest_hops.get(sid,float('inf')) for sid in deployment_server_ids) == h]
        pois_rel_dict.update({pid: environment.pois[pid].correlation * (environment.gamma ** h) for pid in h_pids})
    
    cover_quality = sum(pois_rel_dict.values())
    # 除0跳外的收益
    # 0跳收益
    zero_hop_pois = [pid for pid,poi in environment.pois.items() if min(poi.shortest_hops.get(sid,float('inf')) for sid in deployment_server_ids) == 0]
    zero_hop_cover = sum(environment.pois[pid].correlation * (environment.gamma ** 0) for pid in zero_hop_pois)
    all_cover = cover_quality + zero_hop_cover


    return cover_quality,all_cover