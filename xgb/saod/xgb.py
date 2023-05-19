import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm

from util import batch_generator

if __name__ == '__main__':
    file_path = "../../LD2011_2014.txt"
    shift_unit = 24 * 4 * 30
    total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1  # 减去1是为了排除标题行

    data = pd.read_csv(file_path, sep=';', header=None, skiprows=1, low_memory=False, nrows=7000)  # 1个月有2880行
    column_names = ['Date'] + ['MT_' + str(i).zfill(3) for i in range(1, 371)]
    data.columns = column_names

    print(data.head())

    # 假设时间戳列为第一列（索引为0）
    timestamp_column = 0

    # 将时间戳列转换为 pandas 的 datetime 类型
    data[data.columns[timestamp_column]] = pd.to_datetime(data.iloc[:, timestamp_column])

    # 去除数据中的逗号并转换为浮点数
    data[data.columns[1:]] = data[data.columns[1:]].replace(',', '', regex=True).astype(float)

    # 移除原始时间戳列
    data = data.drop(columns=[timestamp_column])

    # Extract the target variable (electricity consumption)
    data_rolled = data.iloc[:, 4:7].rolling(window=shift_unit, min_periods=1).sum().shift(-shift_unit)
    # To convert values in kWh values must be divided by 4.
    target = data_rolled.iloc[:-shift_unit].sum(axis=1).values / 4
    data = data.iloc[:-shift_unit].values

    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    window_size = 3000  # 滑动窗口的大小

    # 使用 sliding_window_view 函数创建滑动窗口的视图
    windowed_data = np.lib.stride_tricks.sliding_window_view(data_normalized, (window_size, data.shape[1])).squeeze(1)
    windowed_target = target[window_size-1:]

    indices = np.arange(len(windowed_data))

    # 划分训练集和测试集的索引
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    batch_size = 64
    train_num_batches = len(train_indices) // batch_size

    train_generator = batch_generator(windowed_data, windowed_target, train_indices, batch_size)
    test_generator = batch_generator(windowed_data, windowed_target, test_indices, batch_size)

    params = {
        'tree_method': 'gpu_hist',  # 使用GPU加速的方法
        'gpu_id': [0, 1]  # 指定要使用的GPU设备索引
    }

    # 创建 XGBoost 模型
    model = xgb.XGBRegressor(**params)

    # 估计训练批次的总数
    num_samples = len(indices)
    num_batches = num_samples // batch_size

    # Training loop with progress bar
    progress_bar = tqdm(total=num_batches, desc="Training")

    for batch_data, batch_target in train_generator:
        # 将批量数据转换为XGBoost的DMatrix格式
        dmatrix = xgb.DMatrix(data=np.reshape(batch_data, (64, -1)), label=batch_target)
        model.fit(np.reshape(batch_data, (64, -1)), batch_target)

        # 更新进度条
        progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()

    # 存储预测结果的列表
    predictions = []

    # 逐个批次进行预测
    for batch_data, _ in test_generator:
        # 将批量数据转换为 XGBoost 的 DMatrix 格式
        dmatrix = xgb.DMatrix(data=np.reshape(batch_data, (64, -1)))

        # 在测试集上进行预测
        batch_predictions = model.predict(dmatrix)

        # 将批次预测结果添加到总体预测列表
        predictions.append(batch_predictions)

    # 将预测结果拼接为一个 numpy 数组
    test_predictions = np.concatenate(predictions)

    # 打印预测结果
    print(test_predictions)

    y_true = []  # 存储真实标签值
    y_pred = []  # 存储预测值

    # 禁用梯度计算
    with torch.no_grad():
        for batch_data, batch_target in test_generator:
            outputs = model(torch.from_numpy(batch_data).float())

            # 将预测值和真实值添加到列表中
            y_true.extend(batch_target.tolist())
            y_pred.extend(outputs.squeeze().tolist())

    # 转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'均方根误差: {rmse}')

    mae = mean_absolute_error(y_true, y_pred)
    print(f'平均绝对误差: {mae}')

    r2 = r2_score(y_true, y_pred)
    print(f'决定系数: {r2}')
