def calculate_matching_benefit(env, sid):
        """
        计算服务器的匹配收益
        
        Args:
            env: EdgeDeploymentEnvironment对象。
            sid: 服务器id
            
        Returns:
            float: 服务器的匹配收益值
        """

        server = env.edge_servers[sid]
        pois_rel = {pid: env.pois[pid].correlation for pid in server.covered_pois}

            
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
        normalization = n * (n + 1)/2
        # normalization = (n + 1)/2
        if normalization == 0:
            return 0.0
            
        return total_benefit / normalization

