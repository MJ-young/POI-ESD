import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from .environment import EdgeDeploymentEnvironment
from .sigmoid import transform_relevance_adaptive

def read_server_info(filename):
    df = pd.read_csv(filename)
    df = df.convert_dtypes()
    servers_info = [(row['id'], (row['x'], row['y']), row['radius'],row['budget']) for i, row in df.iterrows()]
    return servers_info

# def read_poi_info(filename):
#     df = pd.read_csv(filename)
#     df = df.convert_dtypes()
#     pois_info = [(row['id'], (row['x'], row['y']), row['rel']) for i, row in df.iterrows()]
#     return pois_info

def read_poi_info(service_name,poi_file,rel_file, top_percentile=30, k = 12):
    # 读取rel文件,第一行是表头，第一列是poi的name，后面是50列rel代表当前poi与50个服务的相关性值
    rel_info = pd.read_csv(rel_file,header=0)
        # 读取poi文件
    df = pd.read_csv(poi_file)
    # 根据service_name获取对应的一列
    # rel = rel_info[service_name].tolist()
    # 将rel添加到df中最后一列
    # df['rel'] = rel
    # 将rel进行sigmoid变换
    df['rel'] = transform_relevance_adaptive(rel_info[service_name],top_percentile=30, k = 12)
    # 如果poi文件中没有id列，则添加一列id
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    df = df.convert_dtypes()
    pois_info = [(row['id'], (row['x'], row['y']), row['rel']) for i, row in df.iterrows()]
    return pois_info

def read_edge_info(filename, servers_info):
    df = pd.read_csv(filename)
    df = df.convert_dtypes()
    network_topology = nx.Graph()
    # 添加节点
    for es in servers_info:
        network_topology.add_node(es[0])
    # 添加边
    for _, row in df.iterrows():
        network_topology.add_edge(row['s1'], row['s2'])
    return network_topology

# def generate_environment(service, server_file, poi_file,rel_file, edge_file,DT=1,random_rate=0,service_budget=1, Psi = 1, alpha=1, beta=1):
#     servers_info = read_server_info(server_file)
#     pois_info = read_poi_info(service,poi_file,rel_file)
#     network_topology = read_edge_info(edge_file, servers_info)
    
#     environment = EdgeDeploymentEnvironment(servers_info, pois_info, network_topology,DT=DT,random_rate=random_rate,service_budget=service_budget,Psi = Psi, alpha=alpha, beta=beta)
#     environment.setup_environment()
#     return environment

def generate_environment(service, service_file, server_file, poi_file,rel_file, edge_file, DT=2, alpha=1, beta=1, gamma=0.5, top_percentile=30, k=12):
    # # 读取service文件
    # service_df = pd.read_csv(service_file)
    # 根据service(是id)获取对应的服务的name
    service_name = service.name
    servers_info = read_server_info(server_file)
    pois_info = read_poi_info(service_name,poi_file,rel_file, top_percentile, k)
    network_topology = read_edge_info(edge_file, servers_info)
    
    environment = EdgeDeploymentEnvironment(servers_info, pois_info, network_topology,DT=DT,alpha=alpha, beta=beta, gamma=gamma)
    environment.setup_environment()
    return environment

# 可视化环境
def visualize_environment(environment,save_path=None):
    # 可视化服务器和POI的位置，以及边缘网络拓扑，以及服务器半径
    fig, ax = plt.subplots()
    for server_id, location, radius, budget in environment.servers_info:
        circle = plt.Circle(location, radius, color='r', fill=False)
        ax.add_artist(circle)
        # 画出服务器的位置,marker设置为三角形
        ax.plot(location[0], location[1], 'r^')
        ax.text(location[0], location[1], f'e{server_id}', fontsize=8, ha='center')
    for poi_id, location, rel in environment.pois_info:
        ax.plot(location[0], location[1], 'bo')
        # ax.text(location[0], location[1], f'p{poi_id}({rel})', fontsize=8, ha='center')
        ax.text(location[0], location[1], f'p{poi_id}', fontsize=8, ha='center')

    for edge in environment.network_topology.edges:
        print(edge)
        # print(edge,environment.servers_info[edge[0]])
        # ax.plot([environment.servers_info[edge[0]][1][0], environment.servers_info[edge[1]][1][0]], 
        #         [environment.servers_info[edge[0]][1][1], environment.servers_info[edge[1]][1][1]], 'g--')
        ax.plot([environment.edge_servers[edge[0]].location[0], environment.edge_servers[edge[1]].location[0]], 
                [environment.edge_servers[edge[0]].location[1], environment.edge_servers[edge[1]].location[1]], 'g--')
    ax.set_aspect('equal')
    # plt.show()

    # 如果save_path给定则将图片保存，同时将environment对象保存到文件
    if save_path:
        img_file = f"{save_path}.png"
        plt.savefig(img_file)
        env_file = f"{save_path}.txt"
        if not os.path.exists(env_file):
            os.makedirs(os.path.dirname(env_file), exist_ok=True)
        with open(env_file, "w") as f:
            f.write("Servers info:\n")
            f.write(str(environment.edge_servers))
            f.write("\n")
            f.write("POIs info:\n")
            f.write(str(environment.pois))
            f.write("\n")


if __name__ == "__main__":

    # server_file = "data/example/server.csv"
    # poi_file = "data/example/poi.csv"
    # edge_file = "data/example/edge.csv"

    # server_file = "data/huangpu_all/new/server.csv"
    # poi_file = "data/huangpu_all/new/poi.csv"
    # edge_file = "data/huangpu_all/new/edge.csv"

    place = "small"
    server_file = f"data/monitor_sample/{place}/server_0.2.csv"
    poi_file = f"data/monitor_sample/{place}/poi.csv"
    edge_file = f"data/monitor_sample/{place}/edge_k1_0.2.csv"
    environment = generate_environment(server_file, poi_file, edge_file,DT=1)
    # 这里可以进一步操作或检查返回的environment对象
    print("Environment generated successfully.")
    # 输出服务器和POI的信息
    print("Servers info:")
    print(environment.edge_servers)
    print("POIs info:")
    print(environment.pois)
    # 将生成的环境保存到文件
    env_file = f"data/monitor_sample/{place}/env_k1_0.2.txt"
    # 将environment.edge_servers和environment.pois输出到env_file文件中
    # 没有这个文件的话会自动创建
    if not os.path.exists(env_file):
        os.makedirs(os.path.dirname(env_file), exist_ok=True)
    with open(env_file, "w") as f:
        f.write("Servers info:\n")
        f.write(str(environment.edge_servers))
        f.write("\n")
        f.write("POIs info:\n")
        f.write(str(environment.pois))


    # 可视化环境
    visualize_environment(environment)