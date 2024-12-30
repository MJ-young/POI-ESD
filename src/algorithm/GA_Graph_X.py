import networkx as nx
import random
import copy
import time
from multiprocessing import Pool
from .heuristic import heuristic

class Individual:
    def __init__(self, active_nodes=None):
        if active_nodes is None:
            active_nodes = set()
        self.active_nodes = active_nodes
        self.fitness = 0
        self.cost = 0
        self.covered_pois = set()

    def calculate_fitness(self, environment, service):
        match_revenue = sum(environment.edge_servers[n].match_revenue for n in self.active_nodes)
        # 如果covered_pois已经计算过，直接使用
        # if len(self.covered_pois) > 0:
        #     self.fitness = match_revenue + len(self.covered_pois)
        #     return
        covered_pois = set()
        for sid in self.active_nodes:
            covered_pois.update(environment.edge_servers[sid].rho_pois)
        self.covered_pois = covered_pois
        weighted_revenue = match_revenue*environment.alpha + len(covered_pois)*environment.beta

        # self.fitness = match_revenue + len(covered_pois)
        self.fitness = weighted_revenue
        # self.fitness = match_revenue / environment.match_revenue_upper + len(covered_pois) / environment.cover_revenue_upper

    def mutate(self, graph):
        node = random.choice(list(graph.nodes()))
        switch_nodes = [node] + list(graph.neighbors(node))
        for n in switch_nodes:
            if random.random() < 0.5:
                continue
            if n in self.active_nodes:
                self.active_nodes.remove(n)
            else:
                self.active_nodes.add(n)

    def repair(self, environment, service):
        self.cost = sum(environment.edge_servers[n].price_per_unit * service.size for n in self.active_nodes)
        self.covered_pois = set()
        for sid in self.active_nodes:
            self.covered_pois.update(environment.edge_servers[sid].rho_pois)
        # candidate_nodes = set(environment.edge_servers.keys()) - self.active_nodes
        # while self.cost < service.budget:
        #     if not candidate_nodes:
        #         break
        #     # best_node = max(candidate_nodes, key=lambda n: environment.edge_servers[n].match_revenue+len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)))
        #     best_node = max(candidate_nodes, key=lambda n: environment.edge_servers[n].match_revenue * environment.alpha + len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)) * environment.beta)
        #     self.active_nodes.add(best_node)
        #     self.cost += environment.edge_servers[best_node].price_per_unit * service.size
        #     self.covered_pois.update(environment.edge_servers[best_node].rho_pois)
        #     candidate_nodes.remove(best_node)

        while self.cost > service.budget:
            if not self.active_nodes:
                break
            # worst_node = min(self.active_nodes, key=lambda n: environment.edge_servers[n].match_revenue+len(environment.edge_servers[n].rho_pois))
            worst_node = min(self.active_nodes, key=lambda n: environment.edge_servers[n].match_revenue * environment.alpha + len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)) * environment.beta)
            self.active_nodes.remove(worst_node)
            self.cost -= environment.edge_servers[worst_node].price_per_unit * service.size
            self.covered_pois.difference_update(environment.edge_servers[worst_node].rho_pois)
        
        # 寻找预算范围内所有的服务器
        rest_budget = service.budget - self.cost
        candidate_nodes = set(environment.edge_servers.keys()) - self.active_nodes
        rest_nodes = [n for n in candidate_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]
        while rest_nodes:
            # best_node = max(candidate_nodes, key=lambda n: environment.edge_servers[n].match_revenue+len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)))
            best_node = max(rest_nodes, key=lambda n: environment.edge_servers[n].match_revenue * environment.alpha + len(environment.edge_servers[n].rho_pois.difference(self.covered_pois)) * environment.beta)
            add_cost = environment.edge_servers[best_node].price_per_unit * service.size
            if self.cost + add_cost > service.budget:
                rest_nodes.remove(best_node)
                continue
            self.active_nodes.add(best_node)
            self.cost += add_cost
            self.covered_pois.update(environment.edge_servers[best_node].rho_pois)
            rest_budget -= add_cost
            rest_nodes = [n for n in rest_nodes if environment.edge_servers[n].price_per_unit * service.size <= rest_budget]



    @staticmethod
    def crossover(parent1, parent2, graph):
        """执行交叉操作，仅在交叉点的激活状态不同时交换一个随机选择的节点的激活状态。"""
        crossover_node = random.choice(list(graph.nodes()))

        # 创建子代，初始状态复制自父代
        child1_active_nodes = set(parent1.active_nodes)
        child2_active_nodes = set(parent2.active_nodes)

        # 仅在交叉点交换激活状态，如果父代在该节点的状态不同
        if (crossover_node in parent1.active_nodes) != (crossover_node in parent2.active_nodes):
            # 交换激活状态
            if crossover_node in child1_active_nodes:
                child1_active_nodes.remove(crossover_node)
                child2_active_nodes.add(crossover_node)
            else:
                child1_active_nodes.add(crossover_node)
                child2_active_nodes.remove(crossover_node)

        # 创建子代个体
        child1 = Individual(active_nodes=child1_active_nodes)
        child2 = Individual(active_nodes=child2_active_nodes)

        return child1, child2
    @staticmethod
    def crossover_node_and_neighbors(parent1, parent2, graph):
        """执行交叉操作，基于节点度选择交叉点，并在交叉点及其邻居的激活状态不同时交换激活状态。"""
        degrees = nx.degree(graph)
        crossover_node = random.choices([node for node, degree in degrees], weights=[degree for node, degree in degrees], k=1)[0]

        # # 随机选择交叉点
        # crossover_node = random.choice(list(graph.nodes()))

        # 创建子代，初始状态复制自父代
        child1_active_nodes = set(parent1.active_nodes)
        child2_active_nodes = set(parent2.active_nodes)

        # 交换交叉点及其邻居的激活状态，仅当父代状态不同时
        nodes_to_swap = [crossover_node] + list(graph.neighbors(crossover_node))
        for node in nodes_to_swap:
            if (node in parent1.active_nodes) != (node in parent2.active_nodes):
                # 交换激活状态
                if node in child1_active_nodes:
                    child1_active_nodes.remove(node)
                    child2_active_nodes.add(node)
                else:
                    child1_active_nodes.add(node)
                    child2_active_nodes.remove(node)

        # 创建子代个体
        child1 = Individual(active_nodes=child1_active_nodes)
        child2 = Individual(active_nodes=child2_active_nodes)

        return child1, child2

def fitness_calculation(individual, environment, service):
    individual.calculate_fitness(environment, service)
    return individual

def GA_Graph_X(environment, service):

    best_fitness_history = []
    average_fitness_history = []

    same_times = 0
    last_best_fitness = 0

    start_time = time.perf_counter()
    base_graph = environment.network_topology.copy()
    population_size = 20
    generations = 150
    cross_rate = 1
    mutation_rate = 1

    # Generate initial population
    solutions = heuristic(environment, service,population_size)
    population = [Individual(active_nodes=set(solution)) for solution in solutions]
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
            if random.random() < cross_rate:
                child1, child2 = Individual.crossover_node_and_neighbors(parent1, parent2, base_graph)
                # child1, child2 = Individual.crossover(parent1, parent2, base_graph)
            else:
                # child1, child2 = parent1, parent2
                child1,child2 = copy.deepcopy(parent1),copy.deepcopy(parent2)
            if random.random() < mutation_rate:
                child1.mutate(base_graph)
            if random.random() < mutation_rate:
                child2.mutate(base_graph)
            child1.repair(environment, service)
            child2.repair(environment, service)
            child1.calculate_fitness(environment, service)
            child2.calculate_fitness(environment, service)
            next_generation.extend([child1, child2])

        # Calculate fitness in parallel
        # with Pool(processes=4) as pool:
        #     next_generation = pool.starmap(fitness_calculation, [(ind, environment, service) for ind in next_generation])
        # # 精英保留
        # save_num = int(population_size * 0.6)
        # next_generation = sorted(next_generation, key=lambda ind: ind.fitness, reverse=True)
        # population = next_generation[:save_num]
        # next_generation = next_generation[save_num:]
        # new_num = int(population_size * 0)
        # # new_solutions = heuristic(environment, service, new_num)
        # # new_pop = [Individual(active_nodes=set(solution)) for solution in new_solutions]
        # # for ind in new_pop:
        # #     ind.calculate_fitness(environment, service)
        # # population.extend(new_pop)
        # select_num = population_size - save_num - new_num
        # # 按照fitness作为被选中的概率
        # population.extend(random.choices(next_generation, [ind.fitness for ind in next_generation], k=select_num))

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