import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from util import LSTMModel, batch_generator


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    file_path = "../LD2011_2014.txt"
    shift_unit = 30
    window_size = 100  # 滑动窗口的大小
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

    time_info = daily_aggregated_data.index.tolist()

    # Extract the target variable (electricity consumption)
    data_rolled = daily_aggregated_data.iloc[:, 3:6].rolling(window=shift_unit, min_periods=1).sum().shift(-shift_unit)
    daily_aggregated_data['target'] = data_rolled.sum(axis=1) / 4
    daily_aggregated_data = daily_aggregated_data.iloc[:-shift_unit]

    scaler = MinMaxScaler()  # Normalize the daily_aggregated_data
    data_normalized = scaler.fit_transform(daily_aggregated_data)

    # 使用 sliding_window_view 函数创建滑动窗口的视图
    data_and_label = np.lib.stride_tricks.sliding_window_view(data_normalized, (window_size, daily_aggregated_data.shape[1])).squeeze(1)
    windowed_data = data_and_label[:, :, :-1]
    windowed_target = data_and_label[:, -1, -1]

    indices = np.arange(len(windowed_data))

    test_size = 0.2

    # 划分训练集和测试集的索引
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=seed)
    batch_size = 16
    train_num_batches = len(train_indices) // batch_size

    train_generator = batch_generator(windowed_data, windowed_target, train_indices, batch_size, time_info)
    test_generator = batch_generator(windowed_data, windowed_target, test_indices, batch_size, time_info)

    # Set the hyperparameters
    input_dim = data.shape[1]  # Number of input features (excluding the timestamp column)
    hidden_dim = 72  # Number of hidden units
    output_dim = 1  # Number of output predictions
    num_epochs = 1
    learning_rate = 0.001

    # Create the model
    model = LSTMModel(batch_size, input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 估计训练批次的总数
    num_batches = int(len(indices) // batch_size * (1 - test_size))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(total=train_num_batches, desc='Training Progress', leave=False)
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        # Loop over the batches
        for batch_data, batch_target, _ in train_generator:

            # Perform the forward pass and update the model
            train_outputs = model(torch.from_numpy(batch_data).float())
            loss = criterion(train_outputs, torch.from_numpy(batch_target).unsqueeze(1).float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the progress bar
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    y_true = []  # 存储真实标签值
    y_pred = []  # 存储预测结果的列表

    predictions_dict = {}

    model.eval()

    # 逐个批次进行预测
    for batch_data, batch_target, batch_time_info in test_generator:

        batch_predictions = model(torch.from_numpy(batch_data).float())

        # 将批次预测结果添加到总体预测列表
        y_pred.append(batch_predictions)
        y_true.extend(batch_target.tolist())

        # 输出预测结果和对应的时间信息
        for pred, true, time_info in zip(batch_predictions, batch_target, batch_time_info):
            predictions_dict[time_info] = {'Prediction': pred.item(), 'True': true.item()}

    # 按照时间信息升序显示预测结果和真实值
    for time_info, values in sorted(predictions_dict.items()):
        pred = values['Prediction']
        true = values['True']
        pred_original = scaler.inverse_transform(np.array([[pred]*scaler.n_features_in_]))[0][-1]
        true_original = scaler.inverse_transform(np.array([[true]*scaler.n_features_in_]))[0][-1]

        print(f"Time: {time_info}, Prediction: {pred_original}, True: {true_original}")

    y_pred_np = [t.detach().numpy() for t in y_pred]
    y_pred = np.concatenate(y_pred_np)
    y_true = np.array(y_true)  # 转换为NumPy数组

    print()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'均方根误差: {rmse}')

    mae = mean_absolute_error(y_true, y_pred)
    print(f'平均绝对误差: {mae}')

    r2 = r2_score(y_true, y_pred)
    print(f'决定系数: {r2}')
