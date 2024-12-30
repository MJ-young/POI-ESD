import networkx as nx
import numpy as np
from .edge_server import EdgeServer
from .poi import POI
from .service import Service
import math

class EdgeDeploymentEnvironment:
    def __init__(self, servers_info, pois_info, network_topology, DT=1, alpha=1, beta=1, gamma= 0.5):
        """
        初始化环境构造函数。
        
        :param servers_info: 服务器信息列表，每个元素是一个元组(server_id, location (x,y), coverage_radius, price_per_unit)。
        :param pois_info: POI信息列表，每个元素是一个元组(poi_id, location (x,y), correlation_with_service)
        :param network_topology: 边缘网络拓扑，一个networkx.Graph对象。
        :param DT: 服务器之间的最大跳数。
        :param alpha: 用于计算匹配收益的参数。
        :param beta: 用于计算覆盖收益的参数。
        :param gamma: 用于计算覆盖收益的折损参数
        """
        self.servers_info = servers_info
        self.pois_info = pois_info
        self.network_topology = network_topology
        self.DT = DT
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma

        self.pois = {}
        self.edge_servers = {}
        self.rho = {}

        self.match_revenue_upper = 0
        self.cover_revenue_upper = 0
       

    def calculate_distance(self, location1, location2):
        return math.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)

    def get_covered_pois(self,server_id, location, coverage_radius):
        covered_pois = []
        poi_rels = {}
        revenue = 0
        for poi_id, poi in self.pois.items():
            distance = self.calculate_distance(location, poi.location)
            if distance <= coverage_radius:
                covered_pois.append(poi_id)
                poi_rels[poi_id] = poi.correlation
                # 更新POI对象的covered_by属性
                poi.add_covered_by(server_id)
        revenue = self.calculate_matching_benefit(poi_rels)
        # revenue = sum(poi_rels.values())
        return covered_pois, revenue

    def calculate_matching_benefit(self, pois_rel: dict) -> float:
        """
        计算服务器的匹配收益
        
        Args:
            pois_rel: 字典,key为POI的id,value为POI与服务的相关性值
            
        Returns:
            float: 服务器的匹配收益值
        """
        if not pois_rel:
            return 0.0
            
        # 按相关性值降序排序
        sorted_pois = sorted(pois_rel.items(), key=lambda x: x[1], reverse=True)
        
        n = len(sorted_pois)
        total_benefit = 0.0
        
        # 计算排序加权的匹配收益
        for k, (_, rel) in enumerate(sorted_pois, 1):
            # 权重为剩余POI数量
            weight = n - k + 1
            total_benefit += weight * rel
            
        # 归一化处理,除以POI数量和权重和
        # normalization = n * (n + 1)/2
        normalization = (n + 1)/2
        if normalization == 0:
            return 0.0
            
        return total_benefit / normalization

    
    def generate_pois(self):
        self.pois = {poi_id: POI(poi_id, location, correlation_with_service) for poi_id, location, correlation_with_service in self.pois_info}

    def generate_servers(self):
        # 为每个服务器创建一个EdgeServer对象，先计算覆盖的POIs和邻居服务器再创建对象
        for server_id, location, coverage_radius, price_per_unit in self.servers_info:
            poi_list, match_revenue = self.get_covered_pois(server_id, location, coverage_radius)
            # neighbors = list(self.network_topology.neighbors(server_id))
            # 获取k跳邻居
            if self.DT > 0:
                neighbors = nx.single_source_shortest_path_length(self.network_topology, server_id, cutoff=self.DT)
                neighbors = set(neighbors.keys())
                neighbors.remove(server_id)
            else:
                neighbors = set()
            es_info = {
                'server_id': server_id,
                'location': location,
                'poi_list': poi_list,
                'neighbors': neighbors,
                'match_revenue': match_revenue,
                'coverage_radius': coverage_radius,
                'price_per_unit': price_per_unit
            }
            self.edge_servers[server_id] = EdgeServer(**es_info)
    
    def calculate_rho(self):
        # 提前计算rho为服务器是否能直接或通过邻居覆盖到poi的0-1值
        rho = {}
        # 遍历每个服务器和每个POI来计算rho
        for server_id, server_info in self.edge_servers.items():
            rho_pois = set(server_info.covered_pois)
            # 更新k跳邻居的覆盖POI
            for neighbor_id in server_info.neighbors:
                rho_pois.update(self.edge_servers[neighbor_id].covered_pois)
            self.edge_servers[server_id].rho_pois = rho_pois
            # 计算rho矩阵，如果POI在rho_pois，则rho为1，否则为0
            for poi_id in self.pois.keys():
                rho[(server_id, poi_id)] = 1 if poi_id in rho_pois else 0
        self.rho = rho

    def precompute_network_info(self):
        """
        预计算并存储网络相关的固定信息
        
        Args:
            network: 网络拓扑图
            servers: 服务器字典 {server_id: Server}
            pois: POI字典 {poi_id: POI} 
            service_info: 待部署服务信息
        """
        # 1. 计算服务器间最短跳数
        shortest_hops = dict(nx.all_pairs_shortest_path_length(self.network_topology, cutoff=self.DT))
        for esid, es in self.edge_servers.items():
            es.shortest_hops = shortest_hops[esid]
            # 删除自己到自己的距离
            del es.shortest_hops[esid]
        
        # 2. 计算POI的服务相关性
        # 这个已经存储在poi的correlation属性中了
            
        # 3. 确定POI-服务器的覆盖关系
        # 这个在生成服务器时已经完成了
                    
        # 4. 计算POI到所有服务器的最短跳数
        for poi in self.pois.values():
            # 可能存在无法覆盖的POI
            if not poi.covered_by:
                continue
            # 可能存在poi被多个服务器覆盖的情况,需要计算到所有服务器的最短距离
            for esid in poi.covered_by:
                poi.shortest_hops[esid] = 0
            for esid in self.edge_servers.keys():
                if esid not in poi.shortest_hops:
                    # 未覆盖的POI到服务器的最短距离
                    hops = min([self.edge_servers[crid].shortest_hops.get(esid, float('inf')) for crid in poi.covered_by])
                    if hops <= self.DT:
                        poi.shortest_hops[esid] = hops
            
        # 5. 预计算每个服务器的匹配收益
        # 这个在生成服务器时已经完成了

        # 6. 计算每个服务器的覆盖收益
        # 根据poi.relevance * (gamma ** min_hops)计算自己在DT跳内的覆盖收益
        for esid, es in self.edge_servers.items():
            cr = 0
            # for h in range(1,self.DT+1):
            for h in range(self.DT+1):
                # 求解恰好DT跳可达的POI
                # dt_pois_ids = [poi_id for poi_id, poi in self.pois.items() if poi.shortest_hops[esid] == h] 
                # 上面的方法可能存在没有覆盖的POI，需要改进
                dt_pois_ids = [poi_id for poi_id, poi in self.pois.items() if poi.shortest_hops.get(esid, float('inf')) == h]
                cr += sum([self.pois[poi_id].correlation* (self.gamma ** h) for poi_id in dt_pois_ids])
            es.cover_revenue = cr

        # for esid, es in self.edge_servers.items():
        #     for h in range(1,self.DT+1):
        #         dt_pois_ids = [poi_id for poi_id, poi in self.pois.items() if poi.shortest_hops.get(esid, float('inf')) == h]
        #         h_pois = {poi_id: self.pois[poi_id].correlation * (self.gamma ** h) for poi_id in dt_pois_ids}
        #         h_cr = self.calculate_matching_benefit(h_pois)
        #         es.cover_revenue += h_cr

    def setup_environment(self):
        """
        设置整个待部署边缘场景的环境。
        
        由于服务器和POI的信息以及边缘网络拓扑已经作为输入提供，这里可以进行进一步的设置或初始化工作，
        例如根据边缘网络拓扑
        生成服务器和POI对象，计算服务器之间的最短跳数，计算POI到服务器的最短距离等。
        """
        # 计算相关参数
        self.generate_pois()
        self.generate_servers()
        self.precompute_network_info()
        self.calculate_rho()
        
        # 可以在这里添加其他设置或初始化工作...

        print(f"Environment setup complete with {len(self.edge_servers)} edge servers and {len(self.pois)} POIs.")