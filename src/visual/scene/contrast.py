import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

def visualize_rho_pois_only(poi_csv, algorithm_csv, save_path=None, server_csv=None, show_server=False):
    # 设置字体大小
    plt.rcParams.update({'font.size': 26})  # 调整全局字体大小
    
    # 读取 POI 和算法结果的 CSV 文件
    poi_data = pd.read_csv(poi_csv)
    algorithm_data = pd.read_csv(algorithm_csv)
    # 从 POI 数据中提取坐标
    # poi_y = poi_data['x']
    # poi_x = poi_data['y']

    poi_x = poi_data['x']
    poi_y = poi_data['y']

    # 获取所有 POI 的坐标范围
    x_min, x_max = poi_x.min(), poi_x.max()
    y_min, y_max = poi_y.min(), poi_y.max()

    # 如果提供了服务器数据，则绘制服务器位置
    server_data = None
    if server_csv is not None and show_server:
        server_data = pd.read_csv(server_csv)

    # 循环遍历算法结果的每一行，提取 rho_pois 并进行可视化
    for _, row in algorithm_data.iterrows():
        # 提取算法名称
        algorithm_name = row['algorithm']
        print("visual algorithm_name:", algorithm_name)

        # 提取 rho_pois 并转换为集合
        rho_pois = ast.literal_eval(row['rho_pois'])

        # 过滤出被覆盖的 POI
        covered_pois = poi_data[poi_data['id'].isin(rho_pois)]
        # covered_poi_y = covered_pois['x']
        # covered_poi_x = covered_pois['y']
        covered_poi_x = covered_pois['x']
        covered_poi_y = covered_pois['y']

        # 创建单独的图表
        fig, ax = plt.subplots(figsize=(12, 9))  # 固定图表大小

        # 绘制 POI 点的散点图
        # sc = ax.scatter(covered_poi_x, covered_poi_y, c=covered_pois['rel'], cmap='Reds', s=65, alpha=0.8, label='Covered POI')
        sc = ax.scatter(
            covered_poi_x, 
            covered_poi_y, 
            c=covered_pois['rel'], 
            cmap='Reds', 
            s=65, 
            alpha=0.85, 
            # edgecolors='#1F1F1F',  # 设置边线颜色 #CCCCCC
            # linewidths=0.5,      # 设置边线宽度
            label='Covered POI'
        )

        # 只为特定条件的图像添加 colorbar，例如算法名称包含特定关键字
        if 'None' in algorithm_name:  # 修改条件为你的需求
            # 单独将图片宽度调整为 12
            fig.set_size_inches(12, 10)
            cbar = fig.colorbar(sc, ax=ax, label='POI Relevance', location='right', shrink=0.8, pad=0.02)
            cbar.mappable.set_clim(0, 1)
            sc.set_clim(0, 1)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # 如果提供了服务器数据，则提取servers列并转换为集合
        if server_data is not None:
            servers = ast.literal_eval(row['servers'])
            server_x = server_data[server_data['id'].isin(servers)]['x']
            server_y = server_data[server_data['id'].isin(servers)]['y']
            # 绘制服务器位置
            server_color = '#047CD6'
            # ax.scatter(server_x, server_y, color=server_color, s=85, marker='^', label='Deployed Server')
            ax.scatter(server_x, server_y, color=server_color, s=100, marker='^', label='Deployed Server', edgecolors='black', linewidths=1.5)
            # 绘制服务器覆盖范围，圆形虚线展示
            for server_id in servers:
                server = server_data[server_data['id'] == server_id].iloc[0]
                circle = plt.Circle((server['x'], server['y']), server['radius'], color=server_color, fill=False, linestyle='--', linewidth=1.5)
                ax.add_artist(circle)
                # 用灰色三角标记服务器的邻居服务器

        # 设置图例和轴标签
        ax.set_xlabel('x {m}', fontsize=26, labelpad=10)
        ax.set_ylabel('y {m}', fontsize=26, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.legend(fontsize=24, frameon=True, loc='upper left')
        ax.grid(False)
        # 边框变细
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)

        # 设置坐标轴范围和边距
        ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
        ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

        # 减少图像边距
        plt.tight_layout()

        # 保存每个图表
        if save_path is not None:
            # 判断文件夹是否存在，不存在则创建
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            # plt.savefig(f"{save_path}/scene_{algorithm_name}.png", dpi=300)
            plt.savefig(f"{save_path}/scene_{algorithm_name}.pdf")
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    place = 'huaihai'
    dic_name = f'data/shanghaiTelcom/{place}'
    poi_file = f"{dic_name}/poi.csv"
    server_file = f"{dic_name}/server.csv"
    exp_dic = f"results/{place.capitalize()}"
    res_name = 'contrast_b'
    exp_file = f"{exp_dic}/fix/{res_name}.csv"
    save_path = f"src/visual/scene/pdf/{place}_{res_name}"
    
    visualize_rho_pois_only(poi_file, exp_file, save_path, server_file, show_server=True)
