import numpy as np

def transform_relevance_adaptive(rel_series, top_percentile=30, k=12):
    # 1. 处理负值
    min_val = rel_series.min()
    if min_val < 0:
        rel_series = rel_series - min_val
    
    # 2. 归一化到[0,1]
    # rel_normalized = (rel_series - rel_series.min()) / (rel_series.max() - rel_series.min())
    
    # 3. 计算中心点 - 可以选择以下几种方式:
    # center = rel_normalized.mean()  # 使用均值
    # center = rel_normalized.median()  # 使用中位数
    # center = np.percentile(rel_normalized, 90)  # 使用75分位数

    center = np.percentile(rel_series, top_percentile)  # 使用top_percentile分位数
    
    # 4. 自适应sigmoid变换
    # rel_transformed = 1 / (1 + np.exp(-k * (rel_normalized - center)))
    rel_transformed = 1 / (1 + np.exp(-k * (rel_series - center)))
    
    # 5. 再次归一化到[0,1]
    rel_final = (rel_transformed - rel_transformed.min()) / (rel_transformed.max() - rel_transformed.min())
    
    return rel_final