# 对齐经纬度,由于上海电信数据集的经纬度是错误的，需要对其进行修正
# 已知 31.246672,121.527778 -> 31.243762,121.50292

import pandas as pd
import numpy as np

def uniform(file):
    df = pd.read_csv(file)
    df['latitude'] -= 0.00291
    df['longitude'] += 0.024858
    new_file = file.split('.')[0] + '_uniform.csv'
    df.to_csv(new_file, index=False)

if __name__ == '__main__':
    file = 'data/shanghaiTelcom/base_shanghai.csv'
    uniform(file)
    print("Uniform done!")