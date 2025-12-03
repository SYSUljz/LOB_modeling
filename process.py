import torch
import torch.nn as nn
import sys
import os

# Ensure we can import from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data import prepare_dataframe, get_dataloaders
from model import deeplob
from train import batch_gd

def main():
    # Configuration
    FILE_PATH = r'/home/jack_li/python/LOB_research/fetch_data/data/BTC/processed_merged.parquet'
    WINDOW = 450000
    K_INDEX = 3  # Index of the label to use from the label columns
    NUM_CLASSES = 3
    T = 100
    BATCH_SIZE = 64
    EPOCHS = 50
    MAX_BATCHES_PER_EPOCH_DEBUG = 5 # Set to None for full epoch, e.g., 5 for a quick test
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    # This step reads the parquet, calculates stats, and computes Z-scores
    df = prepare_dataframe(FILE_PATH, window=WINDOW)
    
    # 2. Create DataLoaders
    # This splits the data into train/test and creates PyTorch DataLoaders
    train_loader, test_loader = get_dataloaders(df, k_index=K_INDEX, num_classes=NUM_CLASSES, T=T, batch_size=BATCH_SIZE)
    
    # 3. Model Setup
    model = deeplob(y_len=NUM_CLASSES)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 4. Train
    print("Starting training...")
    train_losses, val_losses = batch_gd(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=EPOCHS, 
        device=device,
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH_DEBUG
    )
    print("Training complete.")

if __name__ == "__main__":
    main()
