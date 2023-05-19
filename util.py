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
