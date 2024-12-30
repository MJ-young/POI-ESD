import networkx as nx
import random
import copy
import time
from .heuristic import heuristic_cover

class Individual:
    def __init__(self, active_nodes=None, added_nodes=None, removed_nodes=None, 
                 cost=0, fitness=0, h_hop_pois=None):
        # 初始化时为None的可变参数赋予新的空对象
        self.active_nodes = set() if active_nodes is None else copy.deepcopy(active_nodes)
        self.added_nodes = set() if added_nodes is None else copy.deepcopy(added_nodes)
        self.removed_nodes = set() if removed_nodes is None else copy.deepcopy(removed_nodes)
        self.h_hop_pois = {} if h_hop_pois is None else copy.deepcopy(h_hop_pois)
        self.fitness = fitness
        self.cost = cost

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
            return 0
        
        for pid in affected_rho_pois:
            new_h = min(environment.pois[pid].shortest_hops.get(sid, float('inf')) 
                       for sid in self.active_nodes)
            if new_h <= environment.DT:
                self.h_hop_pois[pid] = new_h
            else:
                self.h_hop_pois.pop(pid, None)
        
        # 计算变化后的收益
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid, h in self.h_hop_pois.items())
        cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid, h in self.h_hop_pois.items() if h > 0)
        self.fitness = cover_quality
        return self.fitness

    def mutate(self, graph, environment, service):
        # 改进变异策略，基于节点对收益的贡献来决定是否进行变异
        switch_nodes = random.sample(list(graph.nodes()), k=int(len(graph.nodes()) * 0.2))
        for n in switch_nodes:
            if n in self.active_nodes and random.random() < 0.5:
                self.removed_nodes.add(n)
            elif n not in self.active_nodes and random.random() < 0.5:
                self.added_nodes.add(n)
        self.calculate_fitness(environment, service)

    def repair(self, environment, service):
        # 修复个体以确保其符合预算约束
        cost = copy.deepcopy(self.cost)
        cur_active_nodes = copy.deepcopy(self.active_nodes)

        while cost > service.budget:
            if not cur_active_nodes:
                break
            worst_node = min(cur_active_nodes, key=lambda n: environment.edge_servers[n].cover_revenue / (environment.edge_servers[n].match_revenue + 1e-6))
            cur_active_nodes.remove(worst_node)
            self.removed_nodes.add(worst_node)
            cost -= environment.edge_servers[worst_node].price_per_unit * service.size
        
        # 寻找预算范围内的所有服务器
        rest_budget = service.budget - cost
        candidate_nodes = set(environment.edge_servers.keys()) - cur_active_nodes
        rest_nodes = [n for n in candidate_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]

        while rest_nodes:
            best_weight = [(environment.edge_servers[n].cover_revenue + 1) / (environment.edge_servers[n].match_revenue + 1) for n in rest_nodes]
            best_node = random.choices(rest_nodes, weights=best_weight, k=1)[0]

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
        # 改进拓扑交叉操作，基于DT跳邻居选择交叉点
        node_scores = {n: environment.edge_servers[n].cover_revenue / (environment.edge_servers[n].match_revenue + 1) for n in child1.active_nodes}
        if not node_scores:
            return child1, child2
        crossover_center = max(node_scores, key=node_scores.get)
        crossover_subgraph = {crossover_center}
        for node in environment.edge_servers[crossover_center].neighbors:
            crossover_subgraph.add(node)
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

def GA_CoverMM(environment, service):
    best_fitness_history = []
    average_fitness_history = []
    same_times = 0
    last_best_fitness = 0
    start_time = time.perf_counter()
    base_graph = environment.network_topology.copy()
    population_size = 16
    generations = 100
    cross_rate = 0.8
    mutation_rate = 0.6

    # Generate initial population
    solutions = heuristic_cover(environment, service, population_size)
    population = [Individual(added_nodes=set(solution)) for solution in solutions]
    for ind in population:
        ind.calculate_fitness(environment, service)

    for generation in range(generations):
        selected = tournament_selection(population, k=6)
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = [selected.pop(random.randint(0, len(selected) - 1)) for _ in range(2)]
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            if random.random() < cross_rate:
                child1, child2 = Individual.topology_crossover(child1, child2, environment, service)
            if random.random() < mutation_rate:
                child1.mutate(base_graph, environment, service)
            if random.random() < mutation_rate:
                child2.mutate(base_graph, environment, service)
            child1.repair(environment, service)
            child2.repair(environment, service)
            next_generation.extend([child1, child2])
        population = next_generation

        best_fitness = max(ind.fitness for ind in population)
        average_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
        if generation > 0:
            last_best_fitness = max(last_best_fitness, best_fitness)
            if best_fitness == last_best_fitness and best_fitness == best_fitness_history[-2]:
                same_times += 1
            else:
                same_times = 0
        if same_times > 15:
            break

    best_individual = max(population, key=lambda ind: ind.fitness)
    deployment_plan = {node: 1 if node in best_individual.active_nodes else 0 for node in base_graph.nodes()}
    end_time = time.perf_counter()
    run_time = (end_time - start_time) * 1000
    # plot_fitness(best_fitness_history, average_fitness_history)
    return deployment_plan, run_time

def tournament_selection(population, k=3):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(population, k)
        winner = max(participants, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected

def plot_fitness(best_fitness_history, average_fitness_history):
    import matplotlib.pyplot as plt
    plt.plot(range(len(best_fitness_history)), best_fitness_history, label='best_fitness')
    plt.plot(range(len(average_fitness_history)), average_fitness_history, label='average_fitness')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()
