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
            new_h = min(environment.pois[pid].shortest_hops.get(sid,float('inf')) 
                       for sid in self.active_nodes)
            if new_h <= environment.DT:
                self.h_hop_pois[pid] = new_h
            else:
                self.h_hop_pois.pop(pid,None)
        
        # 计算变化后的收益
        # fitness = sum(environment.pois[pid].correlation * (environment.gamma ** h) 
        #              for pid,h in self.h_hop_pois.items() if h>0)
        # fitness = sum(environment.pois[pid].correlation * (environment.gamma ** h) 
        #         for pid,h in self.h_hop_pois.items())
        
        # 全局匹配性
        # rel_pois = {pid: environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items()}
        # fitness = calculate_matching_benefit(rel_pois)poi_num = len(self.h_hop_pois)
        # poi_num = len(self.h_hop_pois)
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items()) / poi_num if poi_num > 0 else 0
        cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items() if h>0)
        # cover_quality = sum(environment.pois[pid].correlation * (environment.gamma ** h) for pid,h in self.h_hop_pois.items())
        fitness = cover_quality
        
        self.fitness = fitness


    def mutate(self, graph, environment, service):
        # node = random.choice(list(graph.nodes()))
        # switch_nodes = [node] + list(graph.neighbors(node))
        switch_nodes = random.sample(list(graph.nodes()), k=int(len(graph.nodes())*0.2))
        for n in switch_nodes:
            # if random.random() < 0.5:
            #     continue
            if n in self.active_nodes:
                self.removed_nodes.add(n)
            else:
                self.added_nodes.add(n)

        self.calculate_fitness(environment, service)

    def repair(self, environment, service):
        cost  = copy.deepcopy(self.cost)
        cur_active_nodes = copy.deepcopy(self.active_nodes)
        cur_indirect_pois = {pid for pid,h in self.h_hop_pois.items() if h>0}

        while cost > service.budget:
            if not cur_active_nodes:
                break
            worst_node = min(cur_active_nodes, key=lambda n: environment.edge_servers[n].cover_revenue/(environment.edge_servers[n].match_revenue+1e-6))
            # worst_node = random.choices(list(cur_active_nodes), weights=[1 / (environment.edge_servers[n].cover_revenue + 1) for n in cur_active_nodes], k=1)[0]
            cur_active_nodes.remove(worst_node)
            self.removed_nodes.add(worst_node)
            cost -= environment.edge_servers[worst_node].price_per_unit * service.size
        
        # 寻找预算范围内所有的服务器
        rest_budget = service.budget - cost
        candidate_nodes = set(environment.edge_servers.keys()) - cur_active_nodes
        rest_nodes = [n for n in candidate_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]
        # 当前间接覆盖的poi，通过h_hop_pois中值>0的poi

        while rest_nodes:
            # best_node = max(rest_nodes, key=lambda n: environment.edge_servers[n].cover_revenue/environment.edge_servers[n].price_per_unit)
            # 目前使用下面的方式
            # best_weight = [(environment.edge_servers[n].cover_revenue+1)/environment.edge_servers[n].price_per_unit for n in rest_nodes]
            best_weight = [(environment.edge_servers[n].cover_revenue+1)/(environment.edge_servers[n].match_revenue+1) for n in rest_nodes]
            best_node = random.choices(rest_nodes,weights=best_weight,k=1)[0]
            # 采用间接覆盖的poi个数作为权重
            # best_weight = [1+len(environment.edge_servers[n].rho_pois.difference(environment.edge_servers[n].covered_pois).difference(cur_indirect_pois)) for n in rest_nodes]
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
        # 目前使用的指标
        # node_scores = {n: environment.edge_servers[n].cover_revenue/(len(environment.edge_servers[n].neighbors.intersection(child1.active_nodes))+1) for n in child1.active_nodes}
        # 为避免match太高，降低选择match高的节点的概率
        node_scores = {n: environment.edge_servers[n].cover_revenue/(environment.edge_servers[n].match_revenue+1) for n in child1.active_nodes}
        # node_scores = {n: environment.edge_servers[n].cover_revenue+1 for n in child1.active_nodes}
        # 采用间接覆盖的poi个数作为权重
        # indirect_pois = {pid for pid,h in child1.h_hop_pois.items() if h>0}
        # node_scores = {n: 1+len(environment.edge_servers[n].rho_pois.difference(environment.edge_servers[n].covered_pois)) for n in child1.active_nodes}
        if not node_scores:
            return child1, child2
        crossover_center = max(node_scores, key=node_scores.get)
        # 获取交叉子图(包含中心点的DT跳邻居)
        crossover_subgraph = {crossover_center}
        for node in environment.edge_servers[crossover_center].neighbors:
            crossover_subgraph.add(node)
        # 执行子图交换
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

def GA_CoverM(environment, service):

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
    solutions = heuristic_cover(environment, service,population_size)
    # population = [Individual(active_nodes=set(solution)) for solution in solutions]
    population = [Individual(added_nodes=set(solution)) for solution in solutions]
    for ind in population:
        ind.calculate_fitness(environment, service)

    # with Pool(processes=4) as pool:
    #     population = pool.starmap(fitness_calculation, [(ind, environment, service) for ind in population])

    # Evolution process
    for generation in range(generations):
        # Selection
        selected = tournament_selection(population, k=6)
        # selected = roulette_selection(population)

        # Crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = [selected.pop(random.randint(0, len(selected) - 1)) for _ in range(2)]
            child1 = Individual(active_nodes=parent1.active_nodes,added_nodes=parent1.added_nodes,removed_nodes=parent1.removed_nodes,cost = parent1.cost,fitness=parent1.fitness,h_hop_pois=parent1.h_hop_pois)
            child2 = Individual(active_nodes=parent2.active_nodes,added_nodes=parent2.added_nodes,removed_nodes=parent2.removed_nodes,cost = parent2.cost,fitness=parent2.fitness,h_hop_pois=parent2.h_hop_pois)
            if random.random() < cross_rate:
                child1, child2 = Individual.topology_crossover(child1, child2, environment, service)
                # child1, child2 = Individual.crossover(child1, child2, base_graph, environment, service)
            if random.random() < mutation_rate:
                child1.mutate(base_graph, environment, service)
            if random.random() < mutation_rate:
                child2.mutate(base_graph, environment, service)
            child1.repair(environment, service)
            child2.repair(environment, service)
            next_generation.extend([child1, child2])

        # Calculate fitness in parallel
        # with Pool(processes=4) as pool:
        #     next_generation = pool.starmap(fitness_calculation, [(ind, environment, service) for ind in next_generation])
        
        population = next_generation

        # Gather statistics
        best_fitness = max(ind.fitness for ind in population)
        average_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
        if generation > 0:
            # 判断是否收敛以及找到过更大的值即使收敛也继续
            last_best_fitness = max(last_best_fitness,best_fitness)
            if best_fitness == last_best_fitness and best_fitness == best_fitness_history[-2]:
                same_times += 1
            else:
                same_times = 0
        if same_times > 10:
            break

    best_individual = max(population, key=lambda ind: ind.fitness)
    if best_individual.fitness > environment.cover_revenue_upper:
        environment.cover_revenue_upper = best_individual.fitness
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
        # selected.append(copy.deepcopy(winner))
        selected.append(winner)
    return selected

# 轮盘赌选择
def roulette_selection(population):
    """轮盘赌选择"""
    fitnesses = [ind.fitness for ind in population]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected = random.choices(population, probabilities, k=len(population))
    # selected =[]
    # for i in range(len(population)):
    #     r = random.random()
    #     sum_ = 0
    #     for j in range(len(probabilities)):
    #         sum_ += probabilities[j]
    #         if r <= sum_:
    #             selected.append(population[j])
    #             break
    return selected

def plot_fitness(best_fitness_history,average_fitness_history):
    import matplotlib.pyplot as plt
    plt.plot(range(len(best_fitness_history)),best_fitness_history,label='best_fitness')
    plt.plot(range(len(average_fitness_history)),average_fitness_history,label='average_fitness')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()