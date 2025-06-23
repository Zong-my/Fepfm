# -*- encoding: utf-8 -*-
'''
@File    :   nn_convlstms.py
@Time    :   2025/04/18 12:12:04
@Author  :   myz 
'''
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from evaluation import rmse_loss, mae_loss, mape_loss, smape_loss


# datasetclass defined
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # inut
        y = self.y[idx]  # output
        x = x.reshape(1, 1, 1, -1)  # reshape (1, 1, 1, input_len)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Enhanced ConvLSTMCell with BatchNorm and Dropout
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dropout=0.2):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.bias = bias
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.bn(self.conv(combined))
        combined_conv = self.dropout(combined_conv)

        if combined_conv.size(-1) != combined.size(-1):
            pad_last_dim = combined.size(-1) - combined_conv.size(-1)
            combined_conv = torch.nn.functional.pad(combined_conv, (0, pad_last_dim))

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# Enhanced ConvLSTM with Residual Connections and Attention
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, out_dim=None):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim[-1], num_heads=4)
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        last_output = layer_output_list[-1][:, -1, :, :, :]
        attention_output, _ = self.attention(last_output.view(b, -1, self.hidden_dim[-1]), 
                                             last_output.view(b, -1, self.hidden_dim[-1]), 
                                             last_output.view(b, -1, self.hidden_dim[-1]))
        prediction = self.fc(attention_output.mean(dim=1))

        return prediction, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# weight init
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


# Optimized Training Function
def convlstm_train_test(train_loader, val_loader, test_loader, scaler_y,
                        input_dim=1, hidden_dim=[64, 64], kernel_size=[(1, None), (1, None)],
                        num_layers=2, batch_first=True, bias=True, return_all_layers=False,
                        lr=0.001, T_max=50, eta_min=1e-6, patience=10, early_stopping_counter=0,
                        num_epochs=50, out_dim=None, v='3', ms=None, rt=None, result_path=''):
    print("Optimized ConvLSTM Training")
    result_csv_path = os.path.join(result_path, 'result.csv')
    result = pd.read_csv(result_csv_path, index_col=0) if os.path.exists(result_csv_path) else pd.DataFrame()
    mp = os.path.join(result_path, 'convlstm')
    os.makedirs(mp, exist_ok=True)

    model = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                     num_layers=num_layers, batch_first=batch_first, bias=bias,
                     return_all_layers=return_all_layers, out_dim=out_dim)
    model_path = os.path.join(mp, f'convlstm_model_v{v}_rt_{rt}_ms{ms}.pth') if ms else os.path.join(mp, f'convlstm_model_v{v}_rt_{rt}.pth')
    pic_path = os.path.join(mp, f'convlstm_v{v}_rt_{rt}_ms{ms}.svg') if ms else os.path.join(mp, f'convlstm_v{v}_rt_{rt}.svg')

    model.apply(init_weights)
    model = model.cuda()

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')
        scheduler.step()
        reduce_lr_on_plateau.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    model.load_state_dict(torch.load(model_path))
    model.to('cuda')

    model.eval()
    all_predictions, all_true_values = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_true_values.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_true_values = np.concatenate(all_true_values)
    all_predictions = scaler_y.inverse_transform(all_predictions)
    all_true_values = scaler_y.inverse_transform(all_true_values)

    y1_test, y2_test = all_true_values[:, 0], all_true_values[:, 1]
    y1_pred, y2_pred = all_predictions[:, 0], all_predictions[:, 1]
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)

    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}, rmse: {rmse_y1}, mape: {mape_y1}, smape: {smape_y1}")
    print('t_delta:')
    print(f"y2 mae: {mae_y2}, rmse: {rmse_y2}, mape: {mape_y2}, smape: {smape_y2}")

    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape',
            'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    svr_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    svr_index = [f'convlstm_v{v}_rt_{rt}_ms{ms}'] if ms else [f'convlstm_v{v}_rt_{rt}']
    df_tmp = pd.DataFrame([svr_values], columns=cols, index=svr_index)
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(result_csv_path)
    print('ConvLSTM complete!')
    return y1_pred, y2_pred