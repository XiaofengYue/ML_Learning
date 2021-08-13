import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from livelossplot import PlotLosses
from apex import amp


def load_csv_data(path, index_col):
    df = pd.read_csv(path,  index_col=index_col)
    df = df.astype('float')
    return df

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
    return data

def generate_data_x_y(data):
    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)
    data = (data-data_mean)/data_std

    train_x = data[:-1]
    train_y = data[1:,4]
    train_x = torch.as_tensor(torch.from_numpy(train_x), dtype=torch.float32)
    train_y = torch.as_tensor(torch.from_numpy(train_y), dtype=torch.float32)

    return data_mean, data_std, train_x, train_y

def loss_func(pred_y, data_y, loss_name):
    if loss_name == "custom":
        with torch.no_grad():
            weights = np.tanh(np.arange(seq_len) * (np.e / seq_len))
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        loss = ((pred_y - data_y[:,:,np.newaxis].cuda()).view(-1, seq_len) **2 * weights)
        loss = loss.mean()
    
    elif loss_name == "MSE":
        loss = nn.MSELoss(pred_y, data_y)
    
    return loss

def save_model(model, path):
    torch.save(model.state_dict(), path)



def generate_train_data(batch_size, train_size, train_x, train_y):
        batch_var_x = []
        batch_var_y = []
        for i in range(batch_size):
                j = train_size - i
                batch_var_x.append(train_x[j:])
                batch_var_y.append(train_y[j:])

        batch_var_x = pad_sequence(batch_var_x, batch_first = True)
        batch_var_y = pad_sequence(batch_var_y, batch_first = True)

        return batch_var_x, batch_var_y


'''Model'''
class RegGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RegGRU, self).__init__()
        
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_csv_data('data/train.csv',  index_col='5 Minutes')
data = create_data(df)
data_mean, data_std, train_x, train_y = generate_data_x_y(data)

input_size = train_x.shape[1]
output_size = 1
hidden_size = 10
num_layers = 1
seq_len = 128
batch_size = 32
train_size = train_x.shape[0]

batch_size = 12 *24  *7 * 2
seq_len = batch_size - 1

batch_var_x, batch_var_y = generate_train_data(batch_size, train_size, train_x, train_y)

'''Training'''

def train_model(epoch, data_x, data_y, loss_name):
    net = RegGRU(input_size, hidden_size, output_size, num_layers).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    net.train()
    p_loss = PlotLosses()
    logs = {}
    loss_value_list = []
    e_tqdm = tqdm(range(epoch))

    for e in e_tqdm:
        pred_y = net(data_x.cuda())

        loss = loss_func(pred_y, data_y, loss_name = 'custom')
        
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # plot
        loss_value_list.append(loss.item())
        logs['loss'] = loss.item()
        e_tqdm.set_description('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
        p_loss.update(logs)
        p_loss.send()
    return net

'''Plot'''
def plot_train(net):
    net.eval()
    pred_x_1 = net(batch_var_x[-1].view(-1,seq_len,5).cuda()).view(-1).cpu().detach().numpy()
    plt.plot(pred_x_1, 'r', label='pred')
    plt.plot(batch_var_y[-1], 'b', label='real', alpha=0.3)
    plt.plot([batch_size, batch_size], [-1, 2], color='k', label='train | pred')



# loss = 'custom'
net = train_model(epoch=200, data_x=batch_var_x, data_y=batch_var_y, loss_name='custom')
save_model(net, './GRU_custom_loss.pkl')


# loss = 'MSE'
net = train_model(epoch=200, data_x=batch_var_x, data_y=batch_var_y, loss_name='MSE')
save_model(net, './GRU_MSE_loss.pkl')