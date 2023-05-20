import numpy as np
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
    file_path = "../LD2011_2014.txt"
    shift_unit = 30
    total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1

    data = pd.read_csv(file_path, sep=';', header=None, skiprows=1, low_memory=False, nrows=200000)  # 假设1个月有30行，200000可超过文件的行数
    column_names = ['Date'] + ['MT_' + str(i).zfill(3) for i in range(1, 371)]
    data.columns = column_names

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.fillna(0, inplace=True)

    data = data.replace(',', '', regex=True).astype(float)  # 去除数据中的逗号并转换为浮点数
    daily_aggregated_data = data.resample('D').sum(numeric_only=True)

    print(daily_aggregated_data.head(3))
    print(daily_aggregated_data.tail(3))

    # Extract the target variable (electricity consumption)
    data_rolled = daily_aggregated_data[['MT_004', 'MT_005', 'MT_006']].rolling(window=shift_unit, min_periods=1).sum().shift(-shift_unit)
    target = data_rolled.iloc[:-shift_unit].sum(axis=1).values / 4  # To convert values in kWh values must be divided by 4.
    time_info = daily_aggregated_data.index.tolist()
    daily_aggregated_data = daily_aggregated_data.iloc[:-shift_unit].values

    scaler = MinMaxScaler()  # Normalize the daily_aggregated_data
    data_normalized = scaler.fit_transform(daily_aggregated_data)

    window_size = 180  # 滑动窗口的大小

    # 使用 sliding_window_view 函数创建滑动窗口的视图
    windowed_data = np.lib.stride_tricks.sliding_window_view(data_normalized, (window_size, daily_aggregated_data.shape[1])).squeeze(1)
    windowed_target = target[window_size-1:]

    indices = np.arange(len(windowed_data))

    test_size = 0.2

    # 划分训练集和测试集的索引
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    batch_size = 16
    train_num_batches = len(train_indices) // batch_size

    train_generator = batch_generator(windowed_data, windowed_target, train_indices, batch_size, time_info)
    test_generator = batch_generator(windowed_data, windowed_target, test_indices, batch_size, time_info)

    params = {
        'tree_method': 'gpu_hist',  # 使用GPU加速的方法
        'gpu_id': 0  # 指定要使用的GPU设备索引
    }

    # 创建 XGBoost 模型
    # model = xgb.XGBRegressor(**params)

    model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=200,  # 设置树的数量
        learning_rate=0.01,  # 设置学习率
        max_depth=3,  # 设置树的最大深度
    )

    # 估计训练批次的总数
    num_batches = int(len(indices) // batch_size * (1 - test_size))

    # Training loop with progress bar
    progress_bar = tqdm(total=num_batches, desc="Training")

    for batch_data, batch_target, _ in train_generator:
        model.fit(np.reshape(batch_data, (batch_data.shape[0], -1)), batch_target)
        progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()

    y_true = []  # 存储真实标签值
    y_pred = []  # 存储预测结果的列表

    predictions_dict = {}

    # 逐个批次进行预测
    for batch_data, batch_target, batch_time_info in test_generator:

        # 在测试集上进行预测
        batch_predictions = model.predict(np.reshape(batch_data, (batch_data.shape[0], -1)))

        # 将批次预测结果添加到总体预测列表
        y_pred.append(batch_predictions)
        y_true.extend(batch_target.tolist())

        # 输出预测结果和对应的时间信息
        for pred, time_info in zip(batch_predictions, batch_time_info):
            predictions_dict[time_info] = pred

    # 按照时间信息升序显示预测结果
    for time_info, pred in sorted(predictions_dict.items()):
        print(f"Time: {time_info}, Prediction: {pred}")

    y_pred = np.concatenate(y_pred)  # 将预测结果拼接为一个 numpy 数组
    y_true = np.array(y_true)  # 转换为NumPy数组

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'均方根误差: {rmse}')

    mae = mean_absolute_error(y_true, y_pred)
    print(f'平均绝对误差: {mae}')

    r2 = r2_score(y_true, y_pred)
    print(f'决定系数: {r2}')
