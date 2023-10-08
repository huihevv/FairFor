import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler


def split_data(data, train_length=7, valid_length=2, test_length=1):
    train_ratio = train_length / (train_length + valid_length + test_length)
    valid_ratio = valid_length / (train_length + valid_length + test_length)
    test_ratio = 1 - train_ratio - valid_ratio
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    return train_data, valid_data, test_data

def data_loader(dataset, batch_size, shuffle=False, drop_last=False):
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)
    return data_loader

def load_dataset(data_dir, dataset):
    if dataset == 'electricity' or dataset == 'traffic' or dataset == 'PeMSD7':
        # input data, sometimes transpose
        data = np.load(data_dir).transpose()
    elif dataset == 'metr-la':
        # metr-la
        df = pd.read_hdf(data_dir)
        data = np.array(df)
        data = data[:, 1:207]
    elif dataset == 'PeMSD4' or dataset == 'PeMSD8':
        # PeMSD4, PeMSD8
        data = np.load(data_dir)['data'][:, :, 0]
    elif dataset == 'covid-19' or dataset == 'ECG5000' or dataset == 'WTH':
        # covid-19, ECG5000, WTH
        data = np.loadtxt(data_dir, delimiter=',')
    else:
        # solar_energy, exchange_rate
        data = np.loadtxt(open(data_dir), delimiter=',')
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def normalize_dataset(data, norm_method, column_wise=False):
    if norm_method == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif norm_method == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif norm_method == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif norm_method == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif norm_method == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def add_window_horizon(data, seq_len, horizon):
    data_X = []
    data_Y = []
    # considering output_weindow
    for i in range(0, len(data[:, 0]) - seq_len - horizon + 1):
        # Previous seq_len data as features
        temp_X = data[i:i + seq_len, :]
        temp_Y = data[i + seq_len:i + seq_len + horizon, :]
        # Values at next time point as labels
        data_X = data_X + [temp_X]
        data_Y = data_Y + [temp_Y]
    data_X = np.asarray(data_X).astype(np.float32)
    data_Y = np.asarray(data_Y).astype(np.float32)
    data_X_t = torch.from_numpy(data_X).type(torch.Tensor)
    data_Y_t = torch.from_numpy(data_Y).type(torch.Tensor)
    # data batch_size
    dataset = TensorDataset(data_X_t, data_Y_t)
    return dataset

def get_dataloader(data, args):
    # normalization
    data, scaler = normalize_dataset(data, args.norm_method, args.column_wise)
    # split dataset
    train_data, valid_data, test_data = split_data(data)
    # add time window
    train_dataset = add_window_horizon(train_data, args.seq_len, args.horizon)
    valid_dataset = add_window_horizon(valid_data, args.seq_len, args.horizon)
    test_dataset = add_window_horizon(test_data, args.seq_len, args.horizon)
    train_loader = data_loader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
    valid_loader = data_loader(valid_dataset, args.batch_size, shuffle=False, drop_last=True)
    test_loader = data_loader(test_dataset, args.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader, scaler


