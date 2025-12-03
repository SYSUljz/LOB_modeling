import pandas as pd
import numpy as np
import torch
import bisect
from torch.utils import data
from torch.utils.data import DataLoader, random_split

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    
    # 确保数据长度足够进行至少一次切片
    if N < T:
        return None, None

    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    
    # 注意：这里使用循环效率较低，如果数据量巨大建议改用 stride_tricks
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, df, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
        
        # 容器，用于存放每个 session 处理后的数据
        self.all_x_list = []
        self.all_y_list = []
        self.cumulative_lengths = []
        total_len = 0
        
        # --- 核心修改开始 ---
        # 1. 按照 session_id 分组
        # 假设 session_id 是 df 的最后一列，或者直接用列名 'session_id'
        grouped = df.groupby('session_id')
        
        print(f"开始处理 {len(grouped)} 个 Session 的数据拼接...")

        for session_id, group_data in grouped:
            # 2. 对每个 Session 独立提取特征和标签
            # 确保只取特征列 (前40列)
            x_session = group_data.iloc[:, :40].to_numpy()
            
            # 提取标签列 (排除最后一列 session_id)
            # 根据 df 结构，labels 是倒数第5列到倒数第1列
            y_session = group_data.iloc[:, -5:-1].to_numpy() 

            # 3. 在组内进行滑动窗口切片 (T=100)
            x_processed, y_processed = data_classification(x_session, y_session, self.T)
            
            # 4. 如果该 Session 长度小于 T，会返回 None，需要跳过
            if x_processed is not None and len(x_processed) > 0:
                self.all_x_list.append(x_processed)
                
                # 处理 Label: 取第 k 个 label 并转为 0-based
                # y_processed shape: (N_sess, 4)
                # y_target shape: (N_sess,)
                #y_target = y_processed[:, self.k] - 1
                y_target = y_processed[:, self.k] + 1
                self.all_y_list.append(y_target)
                
                total_len += len(x_processed)
                self.cumulative_lengths.append(total_len)
        
        if total_len == 0:
            raise ValueError("数据不足：所有 Session 的长度都小于 T，无法生成数据集。")
            
        print(f"数据处理完成。总样本数: {total_len}")
        # --- 核心修改结束 ---
        
        self.length = total_len

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        # 找到 index 对应的 session
        # cumulative_lengths 是递增序列，使用 bisect_right 查找插入点
        session_idx = bisect.bisect_right(self.cumulative_lengths, index)
        
        if session_idx == 0:
            local_index = index
        else:
            local_index = index - self.cumulative_lengths[session_idx - 1]
        
        # 取数据
        # x_data shape: (T, D)
        x_data = self.all_x_list[session_idx][local_index]
        # y_data shape: scalar
        y_data = self.all_y_list[session_idx][local_index]
        
        # 转为 Tensor
        # x_tensor: (1, T, D)
        x_tensor = torch.from_numpy(x_data).float()
        x_tensor = torch.unsqueeze(x_tensor, 0) 
        
        # y_tensor: scalar
        y_tensor = torch.tensor(y_data).long()
        
        return x_tensor, y_tensor

def prepare_dataframe(file_path, window=450000):
    print(f"Loading data from {file_path}...")
    df_out = pd.read_parquet(file_path)
    
    # 1. Initialize the final lists of COLUMN NAMES
    all_price_cols = []
    all_size_cols = []
    k_values = [5, 10, 30, 50]
    label_cols = [f'label_{k}' for k in k_values]

    # 2. Use a loop/list comprehension to generate all 20 column names (10 levels * 2 sides)
    for i in range(10):
        all_price_cols.append(f"ask_px_{i}")
        all_price_cols.append(f"bid_px_{i}")

        all_size_cols.append(f"ask_sz_{i}")
        all_size_cols.append(f"bid_sz_{i}")

    # 3. Select the data in one step for pooling
    price_data = df_out[all_price_cols] # This is a single DataFrame with 20 price columns
    size_data = df_out[all_size_cols]   # This is a single DataFrame with 20 size columns

    # 4. Calculate Pooled Stats (as discussed previously)
    pooled_mean_price = np.mean(price_data.values)
    pooled_devition_price = np.std(price_data.values)
    print(f"Pooled Mean Price: {pooled_mean_price}")
    print(f"Pooled Price Deviation: {pooled_devition_price}")
    
    all_cols = all_price_cols + all_size_cols
    df = df_out[all_cols]

    # Calculate rolling mean / std (excluding current row)
    rolling_mean = df.rolling(window=window, min_periods=window).mean().shift(1)
    rolling_std  = df.rolling(window=window, min_periods=window).std().shift(1)

    # Z-score
    df_z = (df - rolling_mean) / rolling_std
    
    df_processed = pd.concat([
            df_z.reset_index(drop=True),
            df_out[label_cols].reset_index(drop=True),
            df_out[['session_id']].reset_index(drop=True)
        ], axis=1)
    
    df_processed = df_processed.dropna(how='any')
    return df_processed

def get_dataloaders(df, k_index, num_classes, T, batch_size, train_split=0.8):
    dataset = Dataset(df, k=k_index, num_classes=num_classes, T=T)
    
    # Dataset size
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    # Random split
    dataset_train, dataset_test = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
