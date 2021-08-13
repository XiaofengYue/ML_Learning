# 机器学习-学习之旅


## Some Tips(一些技巧)

- apex用来减少显存的使用(大概是一半)，以便塞下更大的Batch
```
from apex import amp
net = RegGRU(input_size, hidden_size, 
net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

# 损失函数这边采用
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```
- 实时绘制损失值（能够初步预判是否在收敛）
```
from livelossplot import PlotLosses

p_loss = PlotLosses()
logs = {}
for e in e_tqdm:
    logs['loss'] = loss.item()
    p_loss.update(logs)
    p_loss.send()
```
- 训练进度条（能够清晰的去判断训练进展如何）
```
from tqdm import tqdm
e_tqdm = tqdm(range(epoch))
for e in e_tqdm:
    e_tqdm.set_description('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
```

## Traffic_Prediction 
> LSTM模型、时序预测算法

> GRU模型、较于LSTM有更少的参数，性能却不见得会掉多少