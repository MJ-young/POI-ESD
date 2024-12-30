import math
import numpy as np
import pandas as pd
import pandas as pd
import math
import matplotlib.pyplot as plt

x_pi = 3.14159265358979324 * 3000.0 / 180.0

# BD-09 to GCJ-02
def bd09_to_gcj02(bd_lon, bd_lat):
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lon = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return gg_lon, gg_lat

# GCJ-02 to Mercator
def gcj02_to_mercator(lon, lat):
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

# BD-09 to Mercator
def bd09_to_mercator(bd_lon, bd_lat):
    gcj_lon, gcj_lat = bd09_to_gcj02(bd_lon, bd_lat)
    mercator_x, mercator_y = gcj02_to_mercator(gcj_lon, gcj_lat)
    return mercator_x, mercator_y

# 绘制基站分布图
def plot_base_station(df):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y'], c='blue', alpha=0.7)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Base Station Distribution')
    plt.show()


if __name__ == '__main__':
    # 读取CSV文件
    file_name = 'base_shanghai_uniform'
    file_path = f'data/shanghaiTelcom/{file_name}.csv'
    # file_name = 'all_rel'
    # file_path = f'data/sbert/{file_name}.csv'

    df = pd.read_csv(file_path)
    # 如果有navi_location_lng和navi_location_lat列，将其重命名为lng和lat
    # if 'navi_location_lng' in df.columns and 'navi_location_lat' in df.columns:
    #     df.rename(columns={'navi_location_lng': 'lng', 'navi_location_lat': 'lat'}, inplace=True)
    # 将latitude,longitude列重命名为lat,lng
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lng'}, inplace=True)
    # 获取原点坐标,取最小的经纬度坐标作为原点
    origin_lon = df['lng'].min()
    origin_lat = df['lat'].min()

    # 将原点坐标转换为墨卡托投影坐标
    origin_mercator_x, origin_mercator_y = bd09_to_mercator(origin_lon, origin_lat)

    # 批量转换经纬度为相对原点的平面坐标
    def convert_and_offset(row):
        mercator_x, mercator_y = bd09_to_mercator(row['lng'], row['lat'])
        relative_x = mercator_x - origin_mercator_x
        relative_y = mercator_y - origin_mercator_y
        return pd.Series([relative_x, relative_y])

    df[['x', 'y']] = df.apply(convert_and_offset, axis=1)
    # 删除原始经纬度列
    df.drop(['lng', 'lat'], axis=1, inplace=True)
    # 如果没有id列，在第一列插入id列
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))

    # 保存结果到新的CSV文件
    new_file_path = file_path.split('.')[0] + '_mercator.csv'
    df.to_csv(new_file_path, index=False, header=True)

    # 绘制基站分布图
    plot_base_station(df)