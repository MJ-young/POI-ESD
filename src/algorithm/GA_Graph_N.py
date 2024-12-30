import networkx as nx
import random
import copy
import time
import math
from multiprocessing import Pool
from .heuristic import heuristic
from utils.coverBen import *


class Individual:
    def __init__(self, active_nodes=None, added_nodes=None, removed_nodes=None, 
                 cost=0, fitness=0, h_hop_pois=None, match_revenue=0, cover_revenue=0):
        # 初始化时为None的可变参数赋予新的空对象
        self.active_nodes = set() if active_nodes is None else copy.deepcopy(active_nodes)
        self.added_nodes = set() if added_nodes is None else copy.deepcopy(added_nodes)
        self.removed_nodes = set() if removed_nodes is None else copy.deepcopy(removed_nodes)
        self.h_hop_pois = {} if h_hop_pois is None else copy.deepcopy(h_hop_pois)
        self.fitness = fitness
        self.cost = cost
        self.match_revenue = match_revenue
        self.cover_revenue = cover_revenue

    def calculate_fitness(self, environment, service):
        # 如果added_nodes为空和removed_nodes为空，则返回缓存的fitness
        if not self.added_nodes and not self.removed_nodes:
            return self.fitness
            
        # 统计受影响的POI-添加的poi
        affected_rho_pois = set()
        for node in self.added_nodes | self.removed_nodes:
            affected_rho_pois.update(environment.edge_servers[node].rho_pois)

        # 更新active_nodes
        self.active_nodes.update(self.added_nodes)
        self.active_nodes.difference_update(self.removed_nodes)
        self.added_nodes.clear()
        self.removed_nodes.clear()
        
        # 更新cost
        self.cost = sum(environment.edge_servers[n].price_per_unit * service.size 
                       for n in self.active_nodes)
        
        # 计算后的h_hop_pois
        if not self.active_nodes:
            self.h_hop_pois.clear()
            self.fitness = 0
            self.match_revenue = 0
            self.cover_revenue = 0
            return 0
        
        for pid in affected_rho_pois:
            new_h = min(environment.pois[pid].shortest_hops.get(sid,float('inf')) 
                       for sid in self.active_nodes)
            if new_h <= environment.DT:
                self.h_hop_pois[pid] = new_h
            else:
                self.h_hop_pois.pop(pid,None)
        # 目前的计算方法
        # # 计算变化后的收益
        # cover_revenue = sum(environment.pois[pid].correlation * (environment.gamma ** h) 
        #              for pid,h in self.h_hop_pois.items() if h>0)

        # # 计算匹配收益:所有被选择部署的服务器匹配收益之和
        # match_revenue = sum(environment.edge_servers[n].match_revenue for n in self.active_nodes)
        # # 最终适应度综合考虑两种收益
        # # weighted_revenue = match_revenue * environment.alpha + cover_revenue * environment.beta
        # # 计算二者公平比例
        # weighted_revenue = 2 * match_revenue * cover_revenue/(match_revenue+cover_revenue)
        # # weighted_revenue = math.sqrt(match_revenue * cover_revenue)
        # # Graph_X
        # # weighted_revenue = match_revenue*environment.alpha + len(self.h_hop_pois)*environment.beta

        # self.fitness = weighted_revenue
        # self.match_revenue = match_revenue
        # self.cover_revenue = cover_revenue

        # 计算精准匹配质量
        match_sum = sum(environment.edge_servers[n].match_revenue for n in self.active_nodes)
        deploy_num = len(self.active_nodes)
        match_quality = match_sum / deploy_num if deploy_num > 0 else 0
        self.match_revenue = match_quality
            
        # 计算全局覆盖匹配质量
        # rel_pois = {pid: environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items()}
        # print('GA_Graph_N:',rel_pois)
        # cover_quality = calculate_matching_benefit(rel_pois)
        # 平均每个poi的覆盖质量
        # poi_num = len(self.h_hop_pois)
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items()) / poi_num if poi_num > 0 else 0
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items() if h>0) / 70
        
        cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items() if h>0) #  if h>0
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items() if h>0) /120
        # 对cover_quality取log10
        # cover_quality = math.log10(cover_quality+1)
        cover_quality = cover_quality / 150
        self.cover_revenue = cover_quality

        # 计算适应度
        self.fitness = 2 * match_quality * cover_quality / (match_quality + cover_quality) if match_quality + cover_quality > 0 else 0
        # self.fitness = cover_quality

    def mutate(self, graph, environment, service):
        """执行变异操作,根据已部署节点的match_revenue选择变异节点"""
        # 假设选择len(self.active_nodes)*0.2个节点进行变异
        mutate_num = int(len(self.active_nodes) * 0.3)
        # 根据match_revenue作为权重选择值低的节点进行变异
        node_scores = {n: 1/(environment.edge_servers[n].match_revenue+1) for n in self.active_nodes}
        # 候选节点的match_revenue值越大,变异为部署服务的概率越大,相当于将值小的节点变异为值大的节点
        candidates = {n: environment.edge_servers[n].match_revenue for n in graph.nodes() if n not in self.active_nodes}
        for _ in range(mutate_num):
            if not node_scores:
                break
            mutate_node = random.choices(
                list(node_scores.keys()),
                weights=list(node_scores.values()),
                k=1
            )[0]
            # mutate_node = min(node_scores, key=node_scores.get)
            del node_scores[mutate_node]
            # 选择一个match_revenue值大于mutate_node的节点进行替换
            replace_node = random.choices(
                list(candidates.keys()),
                weights=list(candidates.values()),
                k=1
            )[0]
            candidates.pop(replace_node)
            self.removed_nodes.add(mutate_node)
            self.added_nodes.add(replace_node)

        self.calculate_fitness(environment, service)

    def repair(self, environment, service):
        cost = copy.deepcopy(self.cost)
        cur_active_nodes = copy.deepcopy(self.active_nodes)
        cur_indirect_pois = {pid for pid,h in self.h_hop_pois.items() if h>0}
        repair_type = 'cover'
        if random.random() < 0.5:
            repair_type = 'match'
        while cost > service.budget:
            if not cur_active_nodes:
                break
            # worst_node = min(self.active_nodes, key=lambda n: environment.edge_servers[n].match_revenue+len(environment.edge_servers[n].rho_pois))
            # worst_node = min(self.active_nodes, key=lambda n: environment.edge_servers[n].match_revenue * environment.alpha + len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)) * environment.beta)
            if repair_type == 'match':
                worst_node = min(cur_active_nodes, key=lambda n: environment.edge_servers[n].match_revenue)
            else:
                # worst_weight = [1/(1+len(environment.edge_servers[n].rho_pois.difference(environment.edge_servers[n].covered_pois))) for n in cur_active_nodes]
                worst_weight = [environment.edge_servers[n].price_per_unit/(environment.edge_servers[n].cover_revenue+1e-6) for n in cur_active_nodes]
                worst_node = random.choices(list(cur_active_nodes),weights=worst_weight,k=1)[0]
            # worst_node = min(cur_active_nodes, key=lambda n: environment.edge_servers[n].cover_revenue)
            self.removed_nodes.add(worst_node)
            cost -= environment.edge_servers[worst_node].price_per_unit * service.size
            cur_active_nodes.remove(worst_node)
            # self.covered_pois.difference_update(environment.edge_servers[worst_node].rho_pois)
        
        # 寻找预算范围内所有的服务器
        rest_budget = service.budget - cost
        candidate_nodes = set(environment.edge_servers.keys()) - cur_active_nodes
        rest_nodes = [n for n in candidate_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]
        while rest_nodes:
            if repair_type == 'match':
                best_node = max(rest_nodes, key=lambda n: environment.edge_servers[n].match_revenue)
            else: 
                best_weight = [((environment.edge_servers[n].cover_revenue+1e-6)/environment.edge_servers[n].price_per_unit) for n in rest_nodes]
                best_node = random.choices(rest_nodes,weights=best_weight,k=1)[0]
                # best_node = max(rest_nodes, key=lambda n: len(environment.edge_servers[n].rho_pois.difference(environment.edge_servers[n].covered_pois).difference(cur_indirect_pois)))
            # best_node = max(rest_nodes, key=lambda n: environment.edge_servers[n].cover_revenue)  # 默认
            # best_node = max(rest_nodes, key=lambda n: len(environment.edge_servers[n].rho_pois.difference(self.h_hop_pois.keys())))
            # best_weight = [len(environment.edge_servers[n].rho_pois.difference(self.h_hop_pois.keys()))+1 for n in rest_nodes]
            # best_node = random.choices(rest_nodes,weights=best_weight,k=1)[0]
            add_cost = environment.edge_servers[best_node].price_per_unit * service.size
            if add_cost > rest_budget:
                rest_nodes.remove(best_node)
                continue
            if best_node in self.removed_nodes:
                self.removed_nodes.remove(best_node)
            else:
                self.added_nodes.add(best_node)
            rest_nodes.remove(best_node)
            rest_budget -= add_cost
            rest_nodes = [n for n in rest_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]
        
        self.calculate_fitness(environment, service)

    @staticmethod
    def topology_crossover(child1, child2, environment, service):
        """执行拓扑交叉操作，基于DT跳邻居选择交叉点，并在交叉点的DT跳邻居内交换节点的激活状态。"""
        # 根据cover_revenue值高的地方进行交叉
        # node_scores = {n: environment.edge_servers[n].cover_revenue for n in child1.active_nodes}
        # node_scores = {n: environment.edge_servers[n].cover_revenue/(len(environment.edge_servers[n].neighbors.intersection(child1.active_nodes))+1) for n in child1.active_nodes}
        node_scores = {n: environment.edge_servers[n].rho_pois for n in child1.active_nodes}
        if not node_scores:
            return child1, child2
        crossover_center = max(node_scores, key=node_scores.get)
        # 获取交叉子图(包含中心点的DT跳邻居)
        crossover_subgraph = {crossover_center}
        for node in environment.edge_servers[crossover_center].neighbors:
            crossover_subgraph.add(node)
        # 在子图范围内交换节点的激活状态
        for node in crossover_subgraph:
            if node in child1.active_nodes and node not in child2.active_nodes:
                child1.removed_nodes.add(node)
                child2.added_nodes.add(node)
            elif node in child2.active_nodes and node not in child1.active_nodes:
                child2.removed_nodes.add(node)
                child1.added_nodes.add(node)
        child1.calculate_fitness(environment, service)
        child2.calculate_fitness(environment, service)
        return child1, child2
    
    @staticmethod
    def crossover(child1, child2, graph, environment, service):
        """执行交叉操作，仅在交叉点的激活状态不同时交换一个随机选择的节点的激活状态。"""
        crossover_node = random.choice(list(graph.nodes()))
        switch_nodes = environment.edge_servers[crossover_node].neighbors.union({crossover_node})

        for node in switch_nodes:
            if (node in child1.active_nodes) != (node in child2.active_nodes):
                if node in child1.active_nodes:
                    child1.removed_nodes.add(node)
                    child2.added_nodes.add(node)
                else:
                    child2.removed_nodes.add(node)
                    child1.added_nodes.add(node)
        
        child1.calculate_fitness(environment, service)
        child2.calculate_fitness(environment, service)
        return child1, child2

def GA_Graph_N(environment, service):

    best_fitness_history = []
    average_fitness_history = []

    same_times = 0
    last_best_fitness = 0
    last_best_individual = None

    start_time = time.perf_counter()
    base_graph = environment.network_topology.copy()
    population_size = 16
    generations = 100
    cross_rate = 0.8
    mutation_rate = 0.4

    # Generate initial population
    solutions = heuristic(environment, service,population_size)
    population = [Individual(added_nodes=set(solution)) for solution in solutions]
    for ind in population:
        ind.calculate_fitness(environment, service)

    # Evolution process
    for generation in range(generations):
        # Selection
        selected = tournament_selection(population, k=6)
        # selected = roulette_selection(population)

        # Crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = [selected.pop(random.randint(0, len(selected) - 1)) for _ in range(2)]
            child1 = Individual(active_nodes=parent1.active_nodes,added_nodes=parent1.added_nodes,removed_nodes=parent1.removed_nodes,cost = parent1.cost,fitness=parent1.fitness,h_hop_pois=parent1.h_hop_pois, match_revenue=parent1.match_revenue, cover_revenue=parent1.cover_revenue)
            child2 = Individual(active_nodes=parent2.active_nodes,added_nodes=parent2.added_nodes,removed_nodes=parent2.removed_nodes,cost = parent2.cost,fitness=parent2.fitness,h_hop_pois=parent2.h_hop_pois, match_revenue=parent2.match_revenue, cover_revenue=parent2.cover_revenue)
            if random.random() < cross_rate:
                child1, child2 = Individual.topology_crossover(child1, child2, environment, service)
                # child1, child2 = Individual.crossover(child1, child2, base_graph, environment, service)
            if random.random() < mutation_rate:
                child1.mutate(base_graph,environment,service)
            if random.random() < mutation_rate:
                child2.mutate(base_graph,environment,service)
            child1.repair(environment, service)
            child2.repair(environment, service)

            next_generation.extend([child1, child2])

        population = next_generation

        # Gather statistics
        cur_best_individual = max(population, key=lambda ind: ind.fitness)
        best_fitness = cur_best_individual.fitness
        average_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
        if best_fitness > last_best_fitness:
                last_best_fitness = best_fitness
                last_best_individual = cur_best_individual
        if generation > 0:
            # 判断是否收敛以及找到过更大的值即使收敛也继续
            if best_fitness == last_best_fitness and best_fitness == best_fitness_history[-2]:
                same_times += 1
            else:
                same_times = 0
        if same_times > 10:
            break

    best_individual = max(population, key=lambda ind: ind.fitness)
    # best_individual = last_best_individual
    deployment_plan = {node: 1 if node in best_individual.active_nodes else 0 for node in base_graph.nodes()}
    end_time = time.perf_counter()
    run_time = (end_time - start_time)*1000
    # plot_fitness(best_fitness_history,average_fitness_history)
    return deployment_plan, run_time

def tournament_selection(population, k=3):
    """锦标赛选择"""
    selected = []
    for _ in range(len(population)):
        participants = random.sample(population, k)
        winner = max(participants, key=lambda ind: ind.fitness)
        # if random.random() < 0.5:
        #     winner = max(participants, key=lambda ind: ind.match_revenue)
        # else:
        #     winner = max(participants, key=lambda ind: ind.cover_revenue)
        selected.append(winner)
    return selected

# 轮盘赌选择
def roulette_selection(population):
    """轮盘赌选择"""
    fitnesses = [ind.fitness for ind in population]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected = random.choices(population, probabilities, k=len(population))
    return selected

def plot_fitness(best_fitness_history,average_fitness_history):
    import matplotlib.pyplot as plt
    plt.plot(range(len(best_fitness_history)),best_fitness_history,label='best_fitness')
    plt.plot(range(len(average_fitness_history)),average_fitness_history,label='average_fitness')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()