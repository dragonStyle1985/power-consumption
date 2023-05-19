import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class LSTMModel(nn.Module):
    def __init__(self, batch_size_, win_len_, input_dim_, hidden_dim_, output_dim_):
        super(LSTMModel, self).__init__()
        self.hidden_dim_ = hidden_dim_
        self.batch_size = batch_size_
        self.lstm = nn.LSTM(input_dim_, hidden_dim_, batch_first=True)
        self.fc = nn.Linear(hidden_dim_, output_dim_)

    def forward(self, x_):
        lstm_out, _ = self.lstm(x_)
        return self.fc(lstm_out[:, -1, :])


if __name__ == '__main__':
    file_path = "../../LD2011_2014.txt"
    shift_unit = 24 * 4 * 30
    total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1  # 减去1是为了排除标题行

    data = pd.read_csv(file_path, sep=';', header=None, skiprows=1, low_memory=False, nrows=7000)  # 1个月有2880行

    print(data.head())

    # 假设时间戳列为第一列（索引为0）
    timestamp_column = 0

    # 将时间戳列转换为 pandas 的 datetime 类型
    data[data.columns[timestamp_column]] = pd.to_datetime(data.iloc[:, timestamp_column])

    # 去除数据中的逗号并转换为浮点数
    data[data.columns[1:]] = data[data.columns[1:]].replace(',', '', regex=True).astype(float)

    # 计算相对时间值（以分钟为单位）
    # reference_time = data.iloc[0, timestamp_column]  # 参考时间点，这里选择第一个时间戳作为参考
    # data['relative_time'] = (data.iloc[:, timestamp_column] - reference_time).dt.total_seconds() / 60 / 15

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

    batch_size = 64

    X_train, X_test, y_train, y_test = train_test_split(windowed_data, windowed_target, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set the hyperparameters
    input_dim = data.shape[1]  # Number of input features (excluding the timestamp column)
    hidden_dim = 64  # Number of hidden units
    output_dim = 1  # Number of output predictions
    num_epochs = 1
    learning_rate = 0.001

    # Create the model
    model = LSTMModel(batch_size, window_size, input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, desc='Training Progress', leave=False)
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        for batch_data in pbar:
            # Forward pass
            x, y = batch_data
            train_outputs = model(x)
            loss = criterion(train_outputs, y.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    # Evaluation
    model.eval()

    y_true = []  # 存储真实标签值
    y_pred = []  # 存储预测值

    # 禁用梯度计算
    with torch.no_grad():
        for batch_data in test_dataloader:
            x, y = batch_data
            outputs = model(x)

            # 将预测值和真实值添加到列表中
            y_true.extend(y.tolist())
            y_pred.extend(outputs.squeeze().tolist())

    # 转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pass
