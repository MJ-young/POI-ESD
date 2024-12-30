import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_visualization(param, df, columns, plot_types, algorithms_style, output_dir=None):
    """
    Parameters:
    df (DataFrame): 数据表格
    columns (list): 需要可视化的数据列
    plot_types (list): 指定的可视化类型, 例如 'line' 或 'bar'
    algorithms_style (dict): 算法的颜色和标记样式
    output_dir (str): 保存图形的目录, 若为 None 则只展示不保存
    """
    # 定义指标对应的名称
    # metrics = {
    #     'match_revenue': 'Matchting Benefit',
    #     'cover_revenue': 'Coverage Benefit',
    #     'cover_poi_num': 'Covered POI Number',
    #     'run_time': 'CPU Time',
    #     'all_cover': 'All Cover',
    # }
    metrics = {
        'match_revenue': 'Matchting Benefit',
        'cover_revenue': 'Topology-aware Decay Benefit',
        'cover_poi_num': 'Covered POI Number',
        'run_time': 'CPU Time',
        'all_cover': 'Global Coverage Benefit'
    }

    # 定义横坐标参数列表
    x_labels = {
        'dT': 'Hop Threshold',
        'budget': 'Budget Constraint ($)',
        'gamma': 'Decay Factor',
    }
    # 如果保存文件夹不存在则创建
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f'{output_dir}/new/pdf'):
            os.makedirs(f'{output_dir}/new/pdf')
    # budgets = sorted(df['budget'].unique())
    params = sorted(df[param].unique())
    x = np.arange(len(params))  # the label locations
    width = 0.11

    algorithms = list(algorithms_style.keys())

    for idx, (column, plot_type) in enumerate(zip(columns, plot_types)):
        plt.figure(figsize=(12, 10))
        plt.xticks(params, fontsize=30)  # 恢复 xticks 为原始参数列表，确保折线图也有刻度
        plt.yticks(fontsize=30)  # 设置 yticks 字体大小

        if plot_type == 'bar':
            for i, alg in enumerate(algorithms):
                data = df[(df['algorithm'] == alg) & (df[column].notna())].sort_values(param)
                if len(data) > 0:
                    plt.bar(x + i * width, data[column], width, 
                            label=alg, color=algorithms_style[alg]['color'],alpha=0.88) # ,alpha=0.9
            plt.xticks(x + width * (len(algorithms) - 1) / 2, params, fontsize=30)  # 更新 xticks 以确保对齐
        elif plot_type == 'line':
            for alg in algorithms:
                data = df[(df['algorithm'] == alg) & (df[column].notna())].sort_values(param)
                if len(data) > 0:
                    style = algorithms_style[alg]
                    plt.plot(data[param], data[column],
                             marker=style['marker'], linewidth=1.5,
                             color=style['color'], label=alg,
                             markerfacecolor='white', markersize=15,
                             markeredgewidth=1.5, alpha=0.9)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        # plt.xlabel(param.capitalize(), fontsize=30, labelpad=10)
        # plt.ylabel(column.replace('_', ' ').title(), fontsize=30, labelpad=10)
        plt.xlabel(x_labels[param], fontsize=30, labelpad=10)
        plt.ylabel(metrics[column], fontsize=30, labelpad=10)
        plt.grid(True, alpha=0.3, linewidth=1.5)  # 增加网格线宽度
        plt.gca().spines['top'].set_linewidth(1.5)  # 加粗上边框
        plt.gca().spines['right'].set_linewidth(1.5)  # 加粗右边框
        plt.gca().spines['bottom'].set_linewidth(1.5)  # 加粗下边框
        plt.gca().spines['left'].set_linewidth(1.5)  # 加粗左边框


        # 针对 run_time 列，将纵轴刻度设置为 10 的次方形式
        if column == 'run_time':
            plt.yscale('log')
            # plt.ylabel(f'{column.replace("_", " ").title()} (log scale)', fontsize=30)
            plt.ylabel(f'{metrics[column]} (log scale)', fontsize=30)

        # 针对 match_revenue 列添加辅助线
        # if column == 'match_revenue':
        #     plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

        # if plot_type == 'bar':
        #     # 右下角添加图例
        #     plt.legend(fontsize=24, frameon=True, loc='lower right')
        # else:
        #     # 自动调整 legend 位置
        #     plt.legend(fontsize=24, frameon=True, loc='best')
        
        # plt.legend(fontsize=24, frameon=True, loc='best')
        # plt.legend(fontsize=24, frameon=True, loc='right')
        # plt.legend(fontsize=24, frameon=True, loc='lower right',framealpha=0.6)
        plt.legend(fontsize=24, frameon=True, loc='upper left',framealpha=0.6)

        # 保存每张图单独绘制
        plt.tight_layout()
        if output_dir:
            if not os.path.exists(f'{output_dir}/pdf'):
                os.makedirs(f'{output_dir}/pdf')
            # plt.savefig(f'{output_dir}/new/pdf/{column}.png', bbox_inches='tight', dpi=300)
            plt.savefig(f'{output_dir}/pdf/{column}.pdf', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# 示例用法
if __name__ == "__main__":
    # 读取数据
    # param = 'dT'
    param = 'budget'
    # param = 'gamma'
    dict_name = f'src/visual/parameter/{param}'
    df = pd.read_csv(f'{dict_name}/{param}_comp.csv')


    # 定义算法颜色映射、标记和标签 - 使用更鲜艳的配色
    algorithms_style = {
        'Random': {'color': '#33BBEE', 'marker': 'o'},  # 蓝
        'BEAD': {'color': '#9370FB', 'marker': 's'},              # 紫
        'RBEAD': {'color': '#FF69A4', 'marker': '^'},             # 粉
        'CRBEAD': {'color': '#FFD02A', 'marker': 'v'},            # 黄
        'MTGA-M': {'color': '#E95342', 'marker': 'D'},          # 红
        'MTGA-T': {'color': '#FF9500', 'marker': 'p'},         # 橙
        'MTGA': {'color': '#00B980', 'marker': 'h'}         # 绿
    }

    # 指定列名和对应的图形类型
    columns = ['match_revenue', 'cover_revenue', 'cover_poi_num', 'run_time', "all_cover"]
    plot_types = ['bar', 'bar', 'bar', 'bar', 'bar']
    # plot_types = ['line', 'line', 'line', 'line', 'line']

    # 调用函数进行绘图
    plot_visualization(param, df, columns, plot_types, algorithms_style, output_dir=dict_name)
