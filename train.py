import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, device, max_batches_per_epoch=None):
    
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        
        model.train()
        t0 = datetime.now()
        train_loss = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss) 
    
        model.eval()
        test_loss = []
        with torch.no_grad(): # Efficient inference
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                    break
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses
