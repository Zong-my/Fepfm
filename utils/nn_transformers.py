# -*- encoding: utf-8 -*-
'''
@File    :   nn_transformers.py
@Time    :   2025/04/18 12:12:13
@Author  :   myz 
'''
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from visualization import *
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from evaluation import *

#
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # 
        y = self.y[idx]  # 
        x = x.reshape(1, 1, 1, -1)  # 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Enhanced Transformer Model
class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, dim_feedforward, dropout, out_dim):
        super(EnhancedTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.out_dim = out_dim

        # Multi-scale feature extraction using 1D convolutions
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # Linear layer to convert input_dim to hidden_dim
        self.input_linear = nn.Linear(hidden_dim, hidden_dim)

        # Transformer Encoder Layer with Layer Normalization and Residual Connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, src):
        # Reshape input from (batch_size, 1, 1, sequence_length) to (sequence_length, batch_size, input_dim)
        src = src.squeeze(1).squeeze(1)  # (batch_size, sequence_length, input_dim)
        src = src.permute(0, 2, 1)  # (batch_size, input_dim, sequence_length)
        src = self.conv1d(src)  # Apply 1D convolution for multi-scale feature extraction
        src = src.permute(2, 0, 1)  # (sequence_length, batch_size, hidden_dim)
        src = self.input_linear(src)  # (sequence_length, batch_size, hidden_dim)
        src = self.pos_encoder(src)  # Add positional encoding
        output = self.transformer_encoder(src)  # Pass through transformer encoder
        output = output[-1, :, :]  # Take the last time step
        output = self.fc(output)  # Final prediction
        return output

def transformer_train_test(train_loader,
                           val_loader,
                           test_loader,
                           scaler_y,
                           input_dim=None, 
                           out_dim=None, 
                           hidden_dim=64,  
                           nhead=8,  
                           num_layers=2,  
                           dim_feedforward=256,  
                           dropout=0.1,
                           lr=0.001,
                           T_max=50, 
                           eta_min=1e-6,
                           patience=10,
                           early_stopping_counter=0,
                           num_epochs=50,
                           v='3',
                           ms=None, 
                           rt=None,
                           result_path=''):  
    print(""" Enhanced Transformer Training """)
    result_csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(result_csv_path): 
        result = pd.read_csv(result_csv_path, index_col=0)
    else:
        result = pd.DataFrame()  

    # Initialize Enhanced Transformer Model
    model = EnhancedTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        out_dim=out_dim
    )
    mp = os.path.join(result_path, 'transformer')
    os.makedirs(mp, exist_ok=True)
    if ms:
        model_path = os.path.join(mp, f'tranformer_model_v{v}_rt_{rt}_ms{ms}.pth')
        pic_path= os.path.join(mp, f'transformer_v{v}_rt_{rt}_ms{ms}.svg')
    else:
        model_path =  os.path.join(mp, f'tranformer_model_v{v}_rt_{rt}.pth')
        pic_path= os.path.join(mp, f'transformer_v{v}_rt_{rt}.svg')
     
    model = model.cuda()  # 
   
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # AdamW with weight decay

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    # 
    warmup_steps = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # 
    best_val_loss = float('inf')

    # 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()  #
            targets = targets.cuda()  # 

            optimizer.zero_grad()

            # Mixed Precision Training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.cuda()  #
                targets = targets.cuda()  # 

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')

     
        scheduler.step()

    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # 
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')

    # 
    model.eval()  # 

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda()  # 
            targets = targets.cuda()  # 

            outputs = model(inputs)
            outputs = outputs.cpu().numpy()  # 
            targets = targets.cpu().numpy()  # 

            all_predictions.append(outputs)
            all_true_values.append(targets)

    all_predictions = np.concatenate(all_predictions)
    all_true_values = np.concatenate(all_true_values)

    # 
    all_predictions = scaler_y.inverse_transform(all_predictions)
    all_true_values = scaler_y.inverse_transform(all_true_values)

    # 
    y1_test, y2_test = all_true_values[:, 0], all_true_values[:, 1]
    y1_pred, y2_pred = all_predictions[:, 0], all_predictions[:, 1]
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)

    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")
    plot_prediction(y1_test, y1_pred, y2_test, y2_pred, save_path=pic_path, dpi=600)
    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    svr_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    svr_index =  [f'tranformer_v{v}_rt_{rt}_ms{ms}'] if ms else [f'tranformer_v{v}_rt_{rt}']
    df_tmp = pd.DataFrame([svr_values], columns=cols, index=[svr_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(result_csv_path)
    print('Enhanced Transformer complete!')
    return y1_pred, y2_pred