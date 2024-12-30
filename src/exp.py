from environment.generator import *
from algorithm.run import *
from environment.service import Service

def main():
    place = 'huaihai'
    # place = 'lujiazui'
    # place = 'wujiaochang'
    
    place_switcher = {
        'huaihai': {
            'gamma': 0.5,
            'budget': 8,
            'edge_k': 1,
            'exp_dic': f'data/shanghaiTelcom/{place}'
        },
        'lujiazui': {
            'gamma': 0.5,
            'budget': 18,
            'edge_k': 2,
            'exp_dic': f'data/shanghaiTelcom/{place}/260'
        },
        'wujiaochang': {
            'gamma': 0.5,
            'budget': 5,
            'edge_k': 2,
            'exp_dic': f'data/shanghaiTelcom/{place}/220'
        }
    }

    gamma = place_switcher[place]['gamma']
    budget = place_switcher[place]['budget']
    edge_k = place_switcher[place]['edge_k']
    exp_dic = place_switcher[place]['exp_dic']

    dT = 2
    alpha = 0.2 # 0.2 large
    beta = 1-alpha
    top_percentile = 25
    k = 12

    server_file = f"{exp_dic}/server.csv"
    poi_file = f"{exp_dic}/poi.csv"
    rel_file = f"{exp_dic}/poi.csv"
    edge_file = f"{exp_dic}/edge_k{edge_k}.csv"
    dic_name = place.capitalize()

    file_name = f"match_dT{dT}_budget{budget}_g{gamma}_k{edge_k}_s{top_percentile}_{k}"

    # service = Service(row['id'], row['size'], row['budget'])
    service = Service(1,'rel', 1, budget)
    # 构建环境
    params = {
        "service": service,
        "service_file": 'service_file',
        "server_file": server_file,
        "poi_file": poi_file,
        "rel_file": rel_file,
        "edge_file": edge_file,
        "DT": dT,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "top_percentile": top_percentile,
        "k": k
    }
    environment = generate_environment(**params)
    # 这里可以进一步操作或检查返回的environment对象
    print("Environment generated successfully.")
    # 可视化环境
    # visualize_environment(environment,save_path=f"data/example/poi")

    algorithm = [random_deployment, BEAD, RBEAD, CRBEAD, GA_Match, GA_CoverM, GA_CoverMM, GA_CoverMM0, GA_Graph_N0]
    # 调用函数
    run(environment,service,algorithm,file_name,dic_name)
 
def param_exp():
    place = 'huaihai'
    place_switcher = {
        'huaihai': {
            'gamma': 0.5,
            'budget': 8,
            'edge_k': 1,
            'exp_dic': f'data/shanghaiTelcom/{place}'
        },
        'lujiazui': {
            'gamma': 0.4,
            'budget': 22,
            'edge_k': 2,
            'exp_dic': f'data/shanghaiTelcom/{place}/260'
        },
        'wujiaochang': {
            'gamma': 0.5,
            'budget': 5,
            'edge_k': 2,
            'exp_dic': f'data/shanghaiTelcom/{place}'
        }
    }

    gamma = place_switcher[place]['gamma']
    budget = place_switcher[place]['budget']
    edge_k = place_switcher[place]['edge_k']
    exp_dic = place_switcher[place]['exp_dic']

    parmas_switcher = {
        'gamma_range': [0.2,0.4,0.6,0.8,1],
        'budget_range': [4,8,12,16], 
        'dT_range': [1,2,3,4],
    }
    # paramter = 'budget'
    # paramter = 'dT'
    paramter = 'gamma'
    param_range = parmas_switcher[paramter+ '_range']

    alpha = 0.2 # 0.2 large
    beta = 1-alpha
    top_percentile = 25
    k = 12
    dT = 2

    server_file = f"{exp_dic}/server.csv"
    poi_file = f"{exp_dic}/poi.csv"
    rel_file = f"{exp_dic}/poi.csv"
    edge_file = f"{exp_dic}/edge_k{edge_k}.csv"
    dic_name = f'{place.capitalize()}/{paramter}'

    for param in param_range:
        print(param)
        if paramter == 'gamma':
            gamma = param
        elif paramter == 'budget':
            budget = param
        elif paramter == 'dT':
            dT = param
        file_name = f"dT{dT}_budget{budget}_g{gamma}_k{edge_k}_s{top_percentile}_{k}"
        # service = Service(row['id'], row['size'], row['budget'])
        service = Service(1,'rel', 1, budget)
        # 构建环境
        params = {
            "service": service,
            "service_file": 'service_file',
            "server_file": server_file,
            "poi_file": poi_file,
            "rel_file": rel_file,
            "edge_file": edge_file,
            "DT": dT,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "top_percentile": top_percentile,
            "k": k
        }
        environment = generate_environment(**params)
        # 这里可以进一步操作或检查返回的environment对象
        print("Environment generated successfully.")
        # 可视化环境
        # visualize_environment(environment,save_path=f"data/example/poi")

        algorithm = [random_deployment, BEAD, RBEAD, CRBEAD, GA_Match, GA_CoverMM, GA_CoverMM0, GA_Graph_N0]
        # 调用函数
        run(environment,service,algorithm,file_name,dic_name)


if __name__ == "__main__":
    
    main()
    # param_exp()