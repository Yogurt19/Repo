import torch
import numpy as np
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# numpy与pytorch对比

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)

tensor2array = torch_data.numpy()

print(
    '\nnumpy: ', np_data,
    '\ntorch: ', torch_data,
    '\ntensor2array: ', tensor2array,
)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 32-bit floating point

print(
    '\nabs: ', data,
    '\nnumpy: ', np.abs(data),
    '\ntorch: ', torch.abs(tensor),
)

print(
    '\nmean: ', data,
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor),
)

data = [[1, 2], [3, 4]]
data = np.array(data)
tensor = torch.FloatTensor(data)

print(
    '\nnumpy: ', np.matmul(data, data),
    '\ntorch: ', torch.mm(tensor, tensor),
    '\nnumpy: ', data.dot(data),
    # '\ntorch: ', tensor.dot(tensor),
)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 变量

from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor) # x^2
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*var = var/2
print(variable)
print(variable.grad)
print(variable.data)
print(variable.data.numpy())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 激励函数

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200) # x data (tensor), shape=(100, 1)
x = Variable(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 回归

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x), Variable(y)
'''
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
'''

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
net = Net(1, 10, 1)
print(net)

plt.ion() # something about plotting
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    
    loss = loss_func(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        # plt.pause(0.1)
      
plt.ioff()
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 分类

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)           # class0 y data (tensor), shape=(100, 1) 
x1 = torch.normal(-2*n_data, 1) # class1 x data (tensor), shape=(100, 2) 
y1 = torch.ones(100)            # class1 y data (tensor), shape=(100, 1) 
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1),).type(torch.LongTensor)     # FloatTensor = 64-bit integer

x, y = Variable(x), Variable(y)
'''
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy())
plt.show()
'''
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
net = Net(2, 10, 2)
# [0, 1] - 1
# [1, 0] - 0
print(net)

plt.ion() # something about plotting
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

# [0, 0, 1]
# [0.1, 0.2, 0.7] = 1

for t in range(100):
    out = net(x) # [-2, -0.12, 20] -> [0.1, 0.2, 0.7] F.softmax(out)
    
    loss = loss_func(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1] # [1]: max prob index, [0]: max prob
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(0.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        # plt.pause(0.1)

plt.ioff()
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 快速搭建法

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print (net)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 保存和提取

torch.manual_seed(1) # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(net1, 'net.pkl') # entire net
    torch.save(net1.state_dict(), 'net_params.pkl') # parameters
    
    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    
    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

save()
restore_net()
restore_params()
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 批数据训练

import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1, 10, 10) # this is x data (torch tensor)
y = torch.linspace(10, 1, 10) # this is x data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # training...
        print('Epoch: ', epoch, '| Step：', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 优化器加速神经网络训练

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20) # hidden layer
        self.predict = torch.nn.Linear(20, 1) # output layer
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# different nets
net_SGD =       Net()
net_Momentum =  Net()
net_RMSprop =   Net()
net_Adam =      Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD =       torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum =  torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop =   torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam =   torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_hist = [[], [], [], []] # record loss

for epoch in range(EPOCH):
    print (epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        
        for net, opt, l_hist in zip(nets, optimizers, losses_hist):
            output = net(b_x) # get output for every net
            loss = loss_func(output, b_y) # compute loss for every net
            opt.zero_grad() # clear gradients for next train
            loss.backward() # backpropagation, compute gradients
            opt.step() # apply gradients
            l_hist.append(loss.data[0]) # loss recorder

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_hist in enumerate(losses_hist):
    plt.plot(l_hist, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 卷积神经网络

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# hyper parameters
EPOCH = 1 # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001 # learning rate
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), # (0-255) -> (0-1)
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.train_data.size()) # (60000, 28, 28)
# print(train_data.train_labels.size()) # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

test_data = torchvision.datasets.MNIST(
    root='mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(), # (0-255) -> (0-1)
    download=DOWNLOAD_MNIST,
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/ 255.
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, # (1, 28, 28)
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2, # if stride = 1, padding = (kernel_size-1)/2 = (5-1)/2 = 2
                      ), # -> (16, 28, 28)
            nn.ReLU(), # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2), # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential( # -> (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # -> (32, 14, 14)
            nn.ReLU(), # -> (32, 14, 14)
            nn.MaxPool2d(2), # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()
# print(cnn) # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x) # batch x
        b_y = Variable(y) # batch y
        
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    # print 10 predictions from test data
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
