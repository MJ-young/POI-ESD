# 实现基于匹配收益和覆盖收益的多目标优化问题的NSGA2算法
import numpy as np
import time
import random
from .gurobi_single import gurobi_single
from .randomF import random_deployment
from .greedy import greedy_single
from .heuristic import heuristic
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, num_genes, solution=[]):
        self.num_genes = num_genes
        # self.solution = np.random.randint(2, size=num_genes)  # 二进制解
        # 调用random_deployment函数生成初始解,这里可能需要转换因为random_deployment返回的是字典
        self.solution = solution
        self.fitness = None
        self.rank = None
        self.crowding_distance = 0
        self.domination_count = 0
        self.dominated_solutions = []
        # 罚项
        # self.penalty = 0
        self.cost = 0

def NSGA2(environment, service):
    # 记录每一代的前沿
    fronts_history = []
    max_fitness_sum_history = []
    average_fitness_sum_history = []
    same_times = 0
    last_best_fitness = 0
    # 为函数计时
    start = time.perf_counter()

    # 初始化参数
    num_genes = len(environment.edge_servers)
    pop_size = 50
    max_generations = 100
    cros_try_times = 10

    # 交叉变异概率
    crossover_rate = 1
    mutation_rate = 1

    # 初始化最优解
    best_solution = None

    # 初始化种群
    population = heuristic_population(environment, service, num_genes, pop_size)
    
    # population = [Individual(num_genes, random_solution(environment,service)) for _ in range(pop_size)]

    # solution, _ = greedy_single(environment,service)
    # key_list = [key for key,value in solution.items() if value == 1]
    # population.append(Individual(num_genes, key_list))

    
    # for individual in population:
    #     calculate_fitness(individual, environment, service)
        # sort_solution(individual,environment)
    
    for generation in range(max_generations):
        # print(f"Generation {generation + 1}/{max_generations}")
        # print(f"Population size: {len(population)}")
        # 快速非支配排序
        fronts = fast_non_dominated_sort(population)
        
        # 计算拥挤距离
        for front in fronts:
            calculate_crowding_distance(front)
        
        # 选择、交叉和变异生成新种群
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = parent1, parent2
            for _ in range(cros_try_times):
                child1, child2 = crossover(parent1, parent2, environment, crossover_rate)
                if child1.fitness is None:
                    calculate_fitness(child1, environment, service)
                if child2.fitness is None:
                    calculate_fitness(child2, environment, service)
                # check_res = check_budget(environment, service, child1.solution) and check_budget(environment, service, child2.solution) and not check_duplicate_boolean(child1.solution) and not check_duplicate_boolean(child2.solution)
                # check_res = check_budget(environment, service, child1.solution) and check_budget(environment, service, child2.solution)
        
                c1 = mutate(child1,environment,service, mutation_rate)
                c2 = mutate(child2,environment,service, mutation_rate)
                if check_budget(environment, service, c1.solution) and check_budget(environment, service, c2.solution):
                    child1 = c1
                    child2 = c2
                    break

            new_population.extend([child1, child2])
        
        # # 评估新种群的适应度
        # for individual in new_population:
        #     calculate_fitness(individual, environment, service)
        #     sort_solution(individual,environment)
        
       # 合并、排序和选择进入下一代
        population.extend(new_population)
        # 随机生成一些新个体，以增加种群的多样性，防止早熟
        # 生成数量为种群的20%
        # for _ in range(int(pop_size * 0.2)):
        #     solution = random_solution(environment,service)
        #     individual = Individual(num_genes, solution)
        #     calculate_fitness(individual, environment, service)
        #     # sort_solution(individual,environment)
        #     population.append(individual)

        # population.extend(heuristic_population(environment, service, num_genes, int(pop_size * 0.2)))

        fronts = fast_non_dominated_sort(population)

        next_population = []
        for front in fronts:
            calculate_crowding_distance(front)  # 为整个前沿计算拥挤距离
            if len(next_population) + len(front) <= pop_size:
                # 如果当前前沿可以完全放入下一代种群，则全部选择
                next_population.extend(front)
            else:
                # 如果当前前沿无法完全放入下一代种群
                # 则基于拥挤距离对前沿内的个体进行排序
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                # 选择拥挤距离最大的个体填充剩余空间
                remaining_size = pop_size - len(next_population)
                next_population.extend(front[:remaining_size])
                break  # 已经填满下一代种群，结束循环
        

        population = next_population  # 更新种群为下一代
        # 保存当前代的Pareto前沿
        # 获取0和1代的前沿
        current_front = [(ind.fitness[0], ind.fitness[1]) for ind in fronts[0]]
        fronts_history.append(current_front)
        best_solution = min(fronts[0], key=lambda ind: sum(ind.fitness))
        # 计算当前代所有个体适应度之和的最大值和平均值
        # fitness_sums = [-sum(ind.fitness) for ind in population]
        # 加权适应度
        fitness_sums = [-ind.fitness[0]*environment.alpha - ind.fitness[1]*environment.beta for ind in population]
        max_fitness_sum = max(fitness_sums)
        average_fitness_sum = sum(fitness_sums) / len(fitness_sums)

        # 将这些值添加到历史记录中
        max_fitness_sum_history.append(max_fitness_sum)
        average_fitness_sum_history.append(average_fitness_sum)

        # 考虑早停
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {max_fitness_sum}")
        if generation > 0:
            # 判断是否收敛以及找到过更大的值即使收敛也继续
            last_best_fitness = max(last_best_fitness,max_fitness_sum)
            if max_fitness_sum == last_best_fitness and max_fitness_sum == max_fitness_sum_history[-2]:
                same_times += 1
            else:
                same_times = 0
        if same_times > 10:
            break

    print(f"Best solution: {best_solution.solution}")
    print(f"Best fitness: {best_solution.fitness}, Sum: {-sum(best_solution.fitness)}")
    deployment_plan = {server_id: 0 for server_id in environment.edge_servers.keys()}
    for server_id in best_solution.solution:
        deployment_plan[server_id] = 1
    # 输出结果
    end = time.perf_counter()
    run_time = (end - start)*1000
    # 绘制得到的前沿
    plot_fronts(fronts)
    plot_fronts_over_time(fronts_history)
    # 绘制适应度之和的最大值和平均值
    plot_fitness(max_fitness_sum_history, average_fitness_sum_history)


    
    # 返回最终种群
    return deployment_plan, run_time

# 根据heuristic生成初始种群解
def heuristic_population(environment, service,num_genes,solution_num):
    population = [Individual(num_genes, solution) for solution in heuristic(environment, service, solution_num)]
    for individual in population:
        calculate_fitness(individual, environment, service)
    return population

# 随机生成一个解
def random_solution(environment,service):
    solution, _ = random_deployment(environment,service)
    # 将类型为字典的solution转换为列表，只取值为1的键
    key_list = [key for key,value in solution.items() if value == 1]
    # 将key_list按照服务器unit_price升序排序
    key_list.sort(key=lambda x:environment.edge_servers[x].price_per_unit)
    return key_list
# 对所有个体进行排序，按照适应度降序排序
def sort_population(population):
    population.sort(key=lambda x:x.fitness,reverse=True)

# 对个体的solution进行排序，按照服务器unit_price升序排序
def sort_solution(individual,environment):
    individual.solution.sort(key=lambda x:environment.edge_servers[x].price_per_unit)

# 检查预算约束
def check_budget(environment, service, solution):
    total_cost = 0
    for server_id in solution:
        total_cost += environment.edge_servers[server_id].price_per_unit
    return total_cost <= service.budget

# 检查重复服务器，如果解中有重复服务器，则进行修复
def check_duplicate(environment, solution):
    # 检查是否有重复服务器,如果有则将重复服务器替换为未出现的服务器
    server_set = set()
    solution_set = set(solution)
    candidate = [sid for sid in environment.edge_servers.keys() if sid not in solution_set]

    for idx, server_id in enumerate(solution):
        if server_id in server_set:
            # 随机选择一个未出现的服务器替换
            select_id = random.choice(candidate)
            solution[idx] = select_id
            candidate.remove(select_id)
        else:
            server_set.add(server_id)
    return solution

# 检查是否有重复服务器返回bool值
def check_duplicate_boolean(solution):
    server_set = set()
    for server_id in solution:
        if server_id in server_set:
            return True
        else:
            server_set.add(server_id)
    return False


# 计算匹配收益和覆盖收益
def calculate_revenue(environment, service, solution):
    # match_revenue = 0
    # cover_revenue = 0
    # total_cost = 0
    # all_covered_pois = set()
    # for server_id in solution:
    #     server = environment.edge_servers[server_id]
    #     total_cost += server.price_per_unit * service.size
    #     match_revenue += server.match_revenue
    #     all_covered_pois.update(poi_id for poi_id in environment.pois.keys() if environment.rho.get((server_id, poi_id), 0))
    # cover_revenue = len(all_covered_pois)

    # 提高计算效率
    match_revenue = sum(environment.edge_servers[sid].match_revenue for sid in solution)
    total_coverage = set(poi_id for sid in solution for poi_id in environment.pois.keys() if environment.rho.get((sid, poi_id), 0))
    cover_revenue = len(total_coverage)
    total_cost = sum(environment.edge_servers[sid].price_per_unit for sid in solution) * service.size

    # # 计算比率
    # match_rate = match_revenue / environment.match_max
    # cover_rate = cover_revenue / environment.cover_max

    return match_revenue, cover_revenue, total_cost
    # return match_revenue/environment.match_revenue_upper, cover_revenue/environment.cover_revenue_upper, total_cost
    # return match_rate, cover_rate, total_cost


def calculate_fitness(individual, environment, service):
    # 计算匹配收益和覆盖收益，需要根据实际情况实现
    matching_revenue, coverage_revenue, cost = calculate_revenue(environment, service, individual.solution)
    individual.fitness = (-matching_revenue, -coverage_revenue)  # NSGA-II默认进行最小化
    # print(f"Fitness: {individual.fitness}, Solution: {individual.solution}")
    individual.cost = cost # 记录罚项

def fast_non_dominated_sort(population):
    # 初始化前沿列表
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual != other_individual:
                if dominates(individual, other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif dominates(other_individual, individual):
                    individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for individual in fronts[i]:
            for dominated_individual in individual.dominated_solutions:
                dominated_individual.domination_count -= 1
                if dominated_individual.domination_count == 0:
                    dominated_individual.rank = i + 1
                    next_front.append(dominated_individual)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # 移除最后一个空的前沿

def dominates(individual_a, individual_b):
    # 检查个体A是否支配个体B
    better_in_all = True  # A在所有目标上都不比B差
    better_in_one = False  # A至少在一个目标上优于B
    
    for i in range(len(individual_a.fitness)):
        if individual_a.fitness[i] > individual_b.fitness[i]:
            better_in_all = False  # 发现A在至少一个目标上比B差
        elif individual_a.fitness[i] < individual_b.fitness[i]:
            better_in_one = True  # 发现A在至少一个目标上优于B
    
    return better_in_all and better_in_one  # A支配B当且仅当A在所有目标上都不比B差且至少在一个目标上优于B
    # return better_in_one  

def calculate_crowding_distance(front):
    # 初始化拥挤距离
    for individual in front:
        individual.crowding_distance = 0

    # 计算每个目标的拥挤距离
    num_objectives = len(front[0].fitness)
    for m in range(num_objectives):
        front.sort(key=lambda x: x.fitness[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        # 获取目标m的最大值和最小值用于归一化
        min_fitness = front[0].fitness[m]
        max_fitness = front[-1].fitness[m]
        range_fitness = max_fitness - min_fitness if max_fitness - min_fitness > 0 else 1
        for i in range(1, len(front) - 1):
            # 归一化距离
            distance = (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / range_fitness
            # distance = (front[i + 1].fitness[m] - front[i - 1].fitness[m])
            front[i].crowding_distance += distance


# 根据交叉概率交叉，且进行多点（两点）交叉
def crossover(parent1, parent2,environment, crossover_rate=0.3):
    """多点交叉"""
    if np.random.rand() > crossover_rate:
        return parent1, parent2
    
    n = len(parent1.solution)
    if n < 3:
        # 如果解的长度小于3，采用单点交叉
        return single_crossover(parent1, parent2,environment, crossover_rate)
    
    # 随机选择交叉点
    crossover_points = np.sort(np.random.choice(n, 2, replace=False))

    # 先对父代的solution进行排序
    sort_solution(parent1,environment)
    sort_solution(parent2,environment)

    # 生成子代解
    child1_solution = parent1.solution[:crossover_points[0]] + parent2.solution[crossover_points[0]:crossover_points[1]] + parent1.solution[crossover_points[1]:]
    child2_solution = parent2.solution[:crossover_points[0]] + parent1.solution[crossover_points[0]:crossover_points[1]] + parent2.solution[crossover_points[1]:]

    # 创建新的个体作为子代
    child1 = Individual(parent1.num_genes, child1_solution)
    child2 = Individual(parent2.num_genes, child2_solution)

    # # 检查重复服务器
    child1.solution = check_duplicate(environment, child1.solution)
    child2.solution = check_duplicate(environment, child2.solution)

    return child1, child2

# 单点交叉
def single_crossover(parent1, parent2,environment, crossover_rate=0.3):
    """单点交叉"""
    if np.random.rand() > crossover_rate:
        return parent1, parent2
    
    # 随机选择交叉点
    crossover_point = np.random.randint(0, len(parent1.solution))

    # 先对父代的solution进行排序
    sort_solution(parent1,environment)
    sort_solution(parent2,environment)

    # 生成子代解
    child1_solution = parent1.solution[:crossover_point] + parent2.solution[crossover_point:]
    child2_solution = parent2.solution[:crossover_point] + parent1.solution[crossover_point:]

    # 创建新的个体作为子代
    child1 = Individual(parent1.num_genes, child1_solution)
    child2 = Individual(parent2.num_genes, child2_solution)

    # # 检查重复服务器
    child1.solution = check_duplicate(environment, child1.solution)
    child2.solution = check_duplicate(environment, child2.solution)

    return child1, child2

# 设置变异算子,随机选择位点变异,变异时增大match或delta_covered_pois较大的服务器被选中的概率
def mutate(individual,environment,service,mutation_rate=0.2):
    """位点变异"""
    if np.random.rand() > mutation_rate:
        return individual
    # 随机选择变异位点
    # mutation_point = np.random.randint(0, len(individual.solution))
    n = len(individual.solution)
    mutation_point = np.random.randint(0, n-1)

    # 在选择变异点时，提高成本高但适应度低的服务器被变异的概率，可以设计简单的函数例如：适应度/成本
    # mutation_weights = [environment.edge_servers[sid].price_per_unit/environment.edge_servers[sid].match_revenue for sid in individual.solution]
    # cover_weights = [environment.edge_servers[sid].price_per_unit/len(environment.edge_servers[sid].covered_pois) for sid in individual.solution]
    # match_weights = [environment.edge_servers[sid].price_per_unit/environment.edge_servers[sid].match_revenue for sid in individual.solution]
    # mutation_weights = [cover_weights[i] + match_weights[i] for i in range(n)]
    # mutation_point = random.choices(range(n),weights=mutation_weights,k=1)[0]

    # # 变异
    # mutation_cost = environment.edge_servers[individual.solution[mutation_point]].price_per_unit
    # # 筛选候选者
    # candiate = [sid for sid in environment.edge_servers.keys() if sid not in individual.solution and environment.edge_servers[sid].price_per_unit <= mutation_cost]
    # candiate.append(individual.solution[mutation_point])

    # all_poi_num = len(environment.pois)
    # match_max = sum(server.match_revenue for server in environment.edge_servers.values())
    # # 计算match_revenue和为individual增加的poi个数最多的服务器
    # individual_covered_pois = set()
    # for sid in individual.solution:
    #     if sid != individual.solution[mutation_point]:
    #         individual_covered_pois.update(poi_id for poi_id in environment.pois.keys() if environment.rho.get((sid, poi_id), 0))

    # candidate_and_neighbors_covered_pois = {sid:set(environment.rho.get((sid, poi_id), 0) for poi_id in environment.pois.keys()) for sid in candiate}
    # delta_covered_pois = {sid:len(candidate_and_neighbors_covered_pois[sid] - individual_covered_pois) for sid in candiate}
    # # score = {sid: environment.edge_servers[sid].match_revenue/match_max + delta_covered_pois[sid]/all_poi_num for sid in candiate}
    # # score = [environment.edge_servers[sid].match_revenue/match_max + delta_covered_pois[sid]/all_poi_num for sid in candiate]
    # # 根据score作为随机选择的权重
    # # mutation_sid = random.choices(candiate,weights=score,k=1)[0]
    # match_score = [environment.edge_servers[sid].match_revenue for sid in candiate]
    # cover_score = [delta_covered_pois[sid] for sid in candiate]

    # max_match_sid = candiate[np.argmax(match_score)]
    # max_cover_sid = candiate[np.argmax(cover_score)]
    # # 设置1/2的概率选择match_score,1/2的概率选择cover_score
    # if np.random.rand() > 0.5 and sum(cover_score) > 0:
    #     mutation_sid = random.choices(candiate,weights=cover_score,k=1)[0]
    #     # mutation_sid = max_cover_sid
    # else:
    #     mutation_sid = random.choices(candiate,weights=match_score,k=1)[0]
    #     # mutation_sid = max_match_sid

    # mutation_sid = random.choice(candiate)

    # 随机生成一个新的服务器直到不在原始解中
    last_cost = individual.cost - environment.edge_servers[individual.solution[mutation_point]].price_per_unit
    candiate = [sid for sid in environment.edge_servers.keys() if sid not in individual.solution and environment.edge_servers[sid].price_per_unit <= service.budget - last_cost]
    if len(candiate) == 0:
        return individual
    # 将candidate中的服务器按照价格赋予不同权重，价格越低权重越高
    weights = [1/environment.edge_servers[sid].price_per_unit for sid in candiate]
    new_solution = individual.solution.copy()
    # mutation_sid = random.choice(candiate)
    mutation_sid = random.choices(candiate,weights=weights,k=1)[0]
    new_solution[mutation_point] = mutation_sid
    # 创建新的个体作为子代
    new_individual = Individual(individual.num_genes, new_solution)
    new_individual.solution = check_duplicate(environment, new_individual.solution)

    calculate_fitness(new_individual, environment, service)
    # new_individual.cost = last_cost + environment.edge_servers[mutation_sid].price_per_unit
    # # delta_cover 假设新服务器覆盖B集合，原服务器覆盖A集合，delta_cover AB交集之外B比A多的POI个数（也可能为负数）
    # delta_cover = len(set(environment.rho.get((mutation_sid, poi_id), 0) for poi_id in environment.pois.keys())) - len(set(environment.rho.get((individual.solution[mutation_point], poi_id), 0) for poi_id in environment.pois.keys()))
    # new_individual.fitness = (individual.fitness[0] + environment.edge_servers[individual.solution[mutation_point]].match_revenue/environment.match_max - environment.edge_servers[mutation_sid].match_revenue/environment.match_max, \
    #                           individual.fitness[1] - delta_cover/environment.cover_max)

    return new_individual


# def select_parents(population):
#     # 实现基于锦标赛选择的父代选择过程
#     # 选择两个个体
#     tournament_size = 3
#     selected_parents = []
#     # candidates = [ind for ind in population if ind.penalty == 0]
#     # print('candidates num / population num:',len(candidates),len(population))
#     for _ in range(2):
#         contenders = np.random.choice(population, size=tournament_size, replace=False)
#         winner = min(contenders, key=lambda ind: (ind.rank, -ind.crowding_distance))
#         # 选择惩罚小、等级低、拥挤距离大的个体
#         # winner = min(contenders, key=lambda ind: (ind.penalty, ind.rank, -ind.crowding_distance))
#         selected_parents.append(winner)
#     return selected_parents[0], selected_parents[1]

# 选择父代，基于轮盘赌选择
def select_parents(population):
    # 实现基于轮盘赌选择的父代选择过程
    # 选择两个个体
    # 计算适应度之和
    fitness_sum = sum(-sum(ind.fitness) for ind in population)
    # 计算每个个体的选择概率
    selection_probabilities = [-sum(ind.fitness) / fitness_sum for ind in population]
    # 选择两个个体
    parent1 = np.random.choice(population, p=selection_probabilities)
    parent2 = np.random.choice(population, p=selection_probabilities)
    return parent1, parent2

def plot_fronts(fronts):
    fig, ax = plt.subplots()
    for i, front in enumerate(fronts):
        x = [ind.fitness[0] for ind in front]
        y = [ind.fitness[1] for ind in front]
        ax.scatter(x, y, label=f"Front {i}")
    ax.set_xlabel("Matching Revenue")
    ax.set_ylabel("Coverage Revenue")
    ax.legend()
    plt.show()


def plot_fronts_over_time(fronts_history):
    plt.figure(figsize=(10, 6))
    num_generations = len(fronts_history)
    colors = ['blue', 'green', 'orange', 'red', 'purple']  # 定义颜色列表
    labels = ['First 40%', 'Next 30%', 'Next 20%', 'Last 10%', 'Last Generation']
    
    # 计算每个阶段的代数范围
    stages = [int(num_generations * 0.4), int(num_generations * 0.7), int(num_generations * 0.9), num_generations - 1]
    
    for i, front in enumerate(fronts_history):
        # 确定当前代数所属的阶段，并选择相应的颜色和标签
        if i < stages[0]:
            color = colors[0]
            label = labels[0] if i == 0 else None  # 只在第一次出现时添加标签
        elif i < stages[1]:
            color = colors[1]
            label = labels[1] if i == stages[0] else None
        elif i < stages[2]:
            color = colors[2]
            label = labels[2] if i == stages[1] else None
        elif i < stages[3]:
            color = colors[3]
            label = labels[3] if i == stages[2] else None
        else:
            color = colors[4]
            label = labels[4] if i == stages[3] else None
        
        x = [point[0] for point in front]
        y = [point[1] for point in front]
        plt.scatter(x, y, color=color, label=label)
    
    plt.title('Pareto Fronts over Generations')
    plt.xlabel('Matching Revenue')
    plt.ylabel('Coverage Revenue')
    plt.legend()
    plt.show()

# 绘制适应度之和的最大值和平均值
def plot_fitness(max_fitness_sum_history, average_fitness_sum_history):
    plt.figure(figsize=(10, 6))
    plt.plot(max_fitness_sum_history, label='Max Fitness Sum')
    plt.plot(average_fitness_sum_history, label='Average Fitness Sum')
    plt.title('Fitness Sum over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Sum')
    plt.legend()
    plt.show()