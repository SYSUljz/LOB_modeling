import pandas as pd
import numpy as np
import torch
import bisect
from torch.utils import data
from torch.utils.data import DataLoader, random_split

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch with Lazy Loading"""
    def __init__(self, df, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
        
        # Containers for RAW session data (not windowed)
        self.raw_sessions_x = []
        self.raw_sessions_y = []
        
        # Metadata to map global index to specific session
        self.cumulative_lengths = []
        total_valid_samples = 0
        
        # 1. Group by session_id
        grouped = df.groupby('session_id')
        print(f"Processing {len(grouped)} Sessions...")

        for session_id, group_data in grouped:
            # 2. Extract raw features and labels
            # Ensure we use float32 to save 50% memory compared to float64
            x_session = group_data.iloc[:, :40].to_numpy(dtype=np.float32)
            y_session = group_data.iloc[:, -5:-1].to_numpy(dtype=np.float32)

            N = len(x_session)
            
            # 3. Calculate how many valid windows this session CAN provide
            # If length is 1000 and T is 100, we can make 901 windows.
            num_windows = N - self.T + 1
            
            # Only keep sessions long enough to form at least one window
            if num_windows > 0:
                self.raw_sessions_x.append(x_session)
                self.raw_sessions_y.append(y_session)
                
                total_valid_samples += num_windows
                self.cumulative_lengths.append(total_valid_samples)
        
        if total_valid_samples == 0:
            raise ValueError("Insufficient data: All sessions are shorter than T.")
            
        print(f"Data setup complete. Total Virtual Samples: {total_valid_samples}")
        # Note: We are now storing ~100x less data in RAM
        self.length = total_valid_samples

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 1. Find which session this index belongs to
        session_idx = bisect.bisect_right(self.cumulative_lengths, index)
        
        if session_idx == 0:
            local_index = index
        else:
            local_index = index - self.cumulative_lengths[session_idx - 1]
        
        # 2. Retrieve the Raw Session Data
        x_raw = self.raw_sessions_x[session_idx]
        y_raw = self.raw_sessions_y[session_idx]
        
        # 3. SLICE ON THE FLY (The Magic Step)
        # We need a window of length T starting at local_index
        # x_window shape: (T, D)
        x_window = x_raw[local_index : local_index + self.T]
        
        # 4. Get the corresponding label
        # In your original logic, label corresponds to the END of the window
        # The label index matches the last row of the x_window
        label_index = local_index + self.T - 1
        y_vals = y_raw[label_index] 
        
        # Process Label (Specific to your logic)
        y_target = y_vals[self.k] + 1
        
        # 5. Convert to Tensor
        # x_tensor: (1, T, D) -> Unsqueeze to match your format
        x_tensor = torch.from_numpy(x_window).float()
        x_tensor = torch.unsqueeze(x_tensor, 0) 
        
        y_tensor = torch.tensor(y_target).long()
        
        return x_tensor, y_tensor

# The rest of your helper functions (prepare_dataframe, get_dataloaders) remain mostly the same,
# but you no longer need the standalone 'data_classification' function.

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
