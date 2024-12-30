import os
from utils.metrics import calculate_metrics
from .randomF import *
from .BEAD import *
from .RBEAD import *
from .CRBEAD import *
from .GA_Match import *
from .GA_CoverM import *
from .GA_CoverMM import *
from .GA_CoverMM0 import *
from .GA_Graph_N import *
from .GA_Graph_N0 import *

import time
import csv

def run(environment,service,algorithm,file_name,dic_name=None):
    # 获取实验时间
    exp_time = time.strftime("%m%d%H%M", time.localtime())
    # file_path = f"results/metrics/{file_name}.csv"
    # if dic_name is None:
        # file_path = f"results/metrics/{file_name}.csv"
    # 创建文件夹
    if os.path.exists(f"results/{dic_name}") == False:
        os.makedirs(f"results/{dic_name}", exist_ok=True)
    # else:
    file_path = f"results/{dic_name}/{file_name}.csv"


    # 对algorithm中的函数依次进行调用，并计算各项指标值
    for algo in algorithm:
        # 根据算法名称调用相应的函数
        deployment_plan, run_time = algo(environment, service)
        # 计算各项指标值
        match_revenue, cover_revenue, estimated_revenue, total_cost, cover_poi_num, hop_revenue, high_rels_num, rho_pois, all_cover, zero_hop_revenue = calculate_metrics(environment,service,deployment_plan)
        # 打印部署方案
        print(f"{algo.__name__} Deployment plan: {deployment_plan}")
        # 计算部署的服务器列表
        servers = [server for server in deployment_plan if deployment_plan[server] == 1]
        # print(f"Algorithm: {algo.__name__}, Estimated revenue: {estimated_revenue} (match: {match_revenue}, cover: {cover_revenue}), Total cost: {total_cost}, Revenue per cost: {revenue_per_cost}, reliability: {reliability}, hm: {hm}, run time: {run_time}")
        print(f"Algorithm: {algo.__name__}, Estimated revenue: {estimated_revenue} (match: {match_revenue}, cover: {cover_revenue}), Total cost: {total_cost}, run time: {run_time}, cover_poi_num: {cover_poi_num}, hop_revenue: {hop_revenue}, high_rels_num: {high_rels_num}, all_cover:{all_cover}, zero_hop_revenue:{zero_hop_revenue}")
        # 将结果保存到csv文件中,如果没有列名则为其添加列名：exp_tmie,algorithm,run_time,match_revenue, cover_revenue, estimated_revenue, total_cost, revenue_per_cost
        # 如果文件不存在，则添加列名
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["exp_time","algorithm","run_time","match_revenue", "zero_hop_revenue", "cover_revenue","all_cover", "estimated_revenue","cover_poi_num","servers","rho_pois"])
        # 将结果写入文件
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([exp_time,algo.__name__,run_time,match_revenue,zero_hop_revenue, cover_revenue,all_cover,estimated_revenue,cover_poi_num,servers,rho_pois])


    
    # 将environment对象保存到文件
    env_file = f"results/{dic_name}/{file_name}_env.txt"
    if not os.path.exists(env_file):
        os.makedirs(os.path.dirname(env_file), exist_ok=True)
    with open(env_file, "w") as f:
        f.write("Servers info:\n")
        f.write(str(environment.edge_servers))
        f.write("\n")
        f.write("POIs info:\n")
        f.write(str(environment.pois))
        f.write("\n")
        