import matplotlib.pyplot as plt
import pandas as pd

def visualize_poi_and_servers(poi_csv, server_csv,dic_name):
    # 读取 POI 和服务器位置的 CSV 文件
    poi_data = pd.read_csv(poi_csv)
    server_data = pd.read_csv(server_csv)

    # 从 POI 数据中提取坐标和相关性
    poi_x = poi_data['x']
    poi_y = poi_data['y']
    poi_relevance = poi_data['rel']

    # 从服务器数据中提取坐标
    server_x = server_data['x']
    server_y = server_data['y']

    # 绘制 POI 和服务器位置的散点图
    plt.figure(figsize=(12, 10))

    # 绘制 POI 点
    sc = plt.scatter(poi_x, poi_y, c=poi_relevance, cmap='Reds', s=30, alpha=0.8, label='POI')

    # 绘制服务器位置
    plt.scatter(server_x, server_y, color='black', s=30, marker='^', label='Server')

    # 设置图例和轴标签
    plt.xlabel('x {m}')
    plt.ylabel('y {m}')
    plt.legend()
    # plt.colorbar(sc, label='POI Relevance')
    # colorbar 的坐标轴从 0 到 1
    # plt.colorbar(sc, label='POI Relevance', ticks=[0, 0.5, 1])
    cbar = plt.colorbar(sc, label='POI Relevance')
    cbar.mappable.set_clim(0, 1)
    sc.set_clim(0, 1)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    # 保存图片
    plt.savefig(f'{dic_name}/scene.png', dpi=300)
    plt.show()

def visualize_poi_and_servers_by_type(poi_csv, server_csv,dic_name):
    # 读取 POI 和服务器位置的 CSV 文件
    poi_data = pd.read_csv(poi_csv)
    server_data = pd.read_csv(server_csv)

    # 从 POI 数据中提取坐标和类型
    poi_x = poi_data['x']
    poi_y = poi_data['y']
    poi_type = poi_data['type']

    # 从服务器数据中提取坐标
    server_x = server_data['x']
    server_y = server_data['y']

    # 定义合并后的 POI 类型和颜色映射（使用亮色系/马克龙色系）
    type_colors = {
        'Residential Area': '#FF7F7F',   # Light Red
        'Business Area': '#FFB347',      # Light Orange
        'Shopping Mall': '#87CEFA',      # Light Blue
        'Scenic Spot': '#98FB98',        # Pale Green
        'Hotel': '#FFD700',              # Bright Yellow
        'Others': '#DA70D6'              # Orchid
    }

    # 类型合并映射
    type_mapping = {
        '住宅区': 'Residential Area',
        '写字楼': 'Business Area',
        '办公楼': 'Business Area',
        '商务大厦': 'Business Area',
        '商场': 'Shopping Mall',
        '大厦': 'Business Area',
        '学校': 'Others',
        '政府机构': 'Others',
        '旅游景点': 'Scenic Spot',
        '码头渡口': 'Others',
        '酒店': 'Hotel',
        '金融': 'Business Area'
    }

    # 应用类型合并
    poi_data['merged_type'] = poi_data['type'].map(type_mapping)

    # 绘制 POI 和服务器位置的散点图
    plt.figure(figsize=(10, 10))

    # 绘制不同类型的 POI 点
    for poi_type_value, color in type_colors.items():
        type_mask = poi_data['merged_type'] == poi_type_value
        plt.scatter(poi_x[type_mask], poi_y[type_mask], color=color, s=30, alpha=0.8, label=poi_type_value)

    # 绘制服务器位置
    plt.scatter(server_x, server_y, color='black', s=30, marker='^', label='Server')

    # 设置图例和轴标签
    plt.xlabel('x {m}')
    plt.ylabel('y {m}')
    plt.legend()
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    # 保存图片
    plt.savefig(f'{dic_name}/scene_type.png', dpi=300)
    plt.show()


# 示例使用
# visualize_poi_and_servers('poi_data.csv', 'server_data.csv')

if __name__ == '__main__':
    place ='huaihai'
    dic_name = f'data/shanghaiTelcom/{place}'
    poi_file = f"{dic_name}/poi.csv"
    server_file = f"{dic_name}/server220_1_0.2.csv"
    visualize_poi_and_servers(poi_file, server_file, dic_name)
    # visualize_poi_and_servers_by_type(poi_file, server_file, dic_name)