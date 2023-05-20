import torch
import torch.nn as nn


def batch_generator(data, target, indices, batch_size, time_info):
    num_batches = len(indices) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_indices = indices[start_idx:end_idx]
        batch_data = data[batch_indices]
        batch_target = target[batch_indices]
        batch_time_info = [time_info[i] for i in batch_indices]

        yield batch_data, batch_target, batch_time_info

    # 处理最后一个不完整的批次
    if len(indices) % batch_size != 0:
        start_idx = num_batches * batch_size
        end_idx = len(indices)

        batch_indices = indices[start_idx:end_idx]
        batch_data = data[batch_indices]
        batch_target = target[batch_indices]
        batch_time_info = [time_info[i] for i in batch_indices]

        yield batch_data, batch_target, batch_time_info


class LSTMModel(nn.Module):
    def __init__(self, batch_size_, input_dim_, hidden_dim_, output_dim_):
        super(LSTMModel, self).__init__()
        self.hidden_dim_ = hidden_dim_
        self.batch_size = batch_size_
        self.lstm = nn.LSTM(input_dim_, hidden_dim_, batch_first=True)
        self.fc = nn.Linear(hidden_dim_, output_dim_, dtype=torch.float32)

    def forward(self, x_):
        lstm_out, _ = self.lstm(x_)
        return self.fc(lstm_out[:, -1, :])
