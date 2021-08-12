import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from livelossplot import PlotLosses

df = pd.read_csv('Traffic_Prediction/data/train.csv',  index_col='5 Minutes')
df = df.astype('float')

def create_data(df):
    indexs = list(df.index)
    indexs = [[i.split(' ')[0],i.split(' ')[1]] for i in indexs]
    # 生成了 xx xx xx xx
    indexs = [[dt[0].split('/')[0],dt[0].split('/')[1],dt[1].split(':')[0], dt[1].split(':')[1]] for dt in indexs]
    date_times = np.array(indexs).astype('float')
    flows = np.array(df['Lane 1 Flow (Veh/5 Minutes)'])
    # 维度弥补
    flows = flows[:,np.newaxis].astype('float')
    data = np.concatenate((date_times, flows),axis=1)
#     mean = np.mean(data, axis=1)
#     std = np.std(data, axis=1)
#     data = (data-mean)/std
    return data

data = create_data(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_mean = np.mean(data,axis=0)
data_std = np.std(data,axis=0)
data = (data-data_mean)/data_std

train_x = data[:-1]
train_y = data[1:,4]
train_x = torch.as_tensor(torch.from_numpy(train_x), dtype=torch.float32)
train_y = torch.as_tensor(torch.from_numpy(train_y), dtype=torch.float32)

input_size = train_x.shape[1]
output_size = 1
hidden_size = 10
num_layers = 1
seq_len = 128
batch_size = 32
train_size = train_x.shape[0]

batch_var_x = []
batch_var_y = []

for i in range(train_size):
    j = train_size - i
    k = j + seq_len
    batch_var_x.append(train_x[j:k])
    batch_var_y.append(train_y[j:k])
    
batch_var_x = pad_sequence(batch_var_x, batch_first = True)
batch_var_y = pad_sequence(batch_var_y, batch_first = True)

class RegLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RegLSTM, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        y = self.rnn(x)[0]
        batch_size, seq_len, hidden_size = y.shape
        y = y.contiguous().view(-1, hidden_size)
        
        y = self.reg(y)
        y = y.view(batch_size, seq_len, -1)
        
        return y

with torch.no_grad():
    weights = np.tanh(np.arange(seq_len) * (np.e / seq_len))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)


epoch = 230
net = RegLSTM(input_size, hidden_size, output_size, num_layers).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

net.train()

p_loss = PlotLosses()
logs = {}

loss_value_list = []
e_tqdm = tqdm(range(epoch))
for e in e_tqdm:
    pred_y = net(batch_var_x.cuda())
    if e%100 == 0:
        print(1)

    loss = ((pred_y - batch_var_y[:,:,np.newaxis].cuda()).view(-1, seq_len) **2 * weights)
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value_list.append(loss.item())
    logs['loss'] = loss.item()
    e_tqdm.set_description('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))

    p_loss.update(logs)
    p_loss.send()
    