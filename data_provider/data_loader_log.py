"""
Log-transformed Dataset for Epidemic Forecasting
=================================================
This module extends Dataset_Custom with log1p transformation to handle
data with high variance and zero values (e.g., epidemic incidence data).

Transformation chain:
1. Raw data -> log1p -> StandardScaler -> model input
2. Model output -> inverse StandardScaler -> expm1 -> original scale
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom_Log(Dataset):
    """
    Custom dataset with log1p transformation for epidemic forecasting.

    Handles high variance data by applying log1p before StandardScaler.
    Provides inverse_transform method to convert predictions back to original scale.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='data.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, percent=100):

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Reorder columns: ['date', ...(other features), target feature]
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # Train/Val/Test split: 70/20/10
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # Select features based on mode
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Apply log1p transformation BEFORE scaling
        data_raw = df_data.values
        data_log = np.log1p(data_raw)  # log1p handles zeros gracefully: log1p(0) = 0

        if self.scale:
            # Fit scaler on TRAIN data only (in log space)
            train_data = data_log[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            # Transform all data
            data = self.scaler.transform(data_log)
        else:
            data = data_log

        # Time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1].astype(np.float32)
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1].astype(np.float32)
        seq_x_mark = self.data_stamp[s_begin:s_end].astype(np.float32)
        seq_y_mark = self.data_stamp[r_begin:r_end].astype(np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        """
        Full inverse transformation: scaled -> log space -> original scale

        Args:
            data: np.ndarray in scaled space (model output)

        Returns:
            np.ndarray in original scale
        """
        # Step 1: Inverse StandardScaler (scaled -> log space)
        data_log = self.scaler.inverse_transform(data)

        # Step 2: Inverse log1p (log space -> original scale)
        data_original = np.expm1(data_log)

        # Step 3: Clip to non-negative (epidemic constraint)
        data_original = np.maximum(data_original, 0)

        return data_original
