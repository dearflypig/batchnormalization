import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

#train data
x = np.linspace(-5, 5, 1000)[:, np.newaxis]
y = np.square(x) + np.random.normal(0, 1, x.shape)
train_x = torch.from_numpy(x).float()
train_y = torch.from_numpy(y).float()

# test data
test_x = np.linspace(-5, 5, 100)[:, np.newaxis]
test_y = np.square(test_x)  + np.random.normal(0, 1, test_x.shape)
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2,)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.bn_input=nn.BatchNorm1d(1,momentum=0.5)
        self.fc0=nn.Linear(1,10)
        self.bn0=nn.BatchNorm1d(10,momentum=0.5)
        self.fc1=nn.Linear(10,10)
        self.bn1=nn.BatchNorm1d(10,momentum=0.5)
        self.fc2=nn.Linear(10,10)
        self.bn2=nn.BatchNorm1d(10,momentum=0.5)
        self.fc3=nn.Linear(10,10)
        self.bn3=nn.BatchNorm1d(10,momentum=0.5)
        self.predict=nn.Linear(10,1)
    
    def forward(self,x):
        x = self.fc0(x)
        x = self.bn0(x)   # batch normalization
        x = torch.tanh(x)

        x = self.fc1(x)
        x = self.bn1(x)   # batch normalization
        x = torch.tanh(x)

        x = self.fc2(x)
        x = self.bn2(x)   # batch normalization
        x = torch.tanh(x)

        x = self.fc3(x)
        x = self.bn3(x)   # batch normalization
        x = torch.tanh(x)

        out=self.predict(x)
        return out

net=Net() 
opt=torch.optim.Adam(net.parameters(),lr=0.02)
#print(opt.param_groups) 

loss_func=nn.MSELoss()
if __name__ == "__main__":
    l=[] 
    for epoch in range(10):
        print('Epoch: ', epoch)
        net.eval()              # set eval mode to fix moving_mean and moving_var
        pred = net(test_x)
        l.append(loss_func(pred, test_y).data.item())
        net.train()             # free moving_mean and moving_var
        for step, (b_x, b_y) in enumerate(train_loader):
            pred = net(b_x)
            loss = loss_func(pred, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()    # it will also learns the parameters in Batch Normalization
    print(l)
    plt.figure(1)
    
    plt.plot(l, c='#74BCFF', lw=2, label='Batch Normalization')
    plt.xlabel('step');plt.ylabel('test loss');plt.ylim((0, 200));plt.legend(loc='best')

    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    net.eval() 
    preds = net(test_x)
    plt.figure(2)

    plt.plot(test_x.data.numpy(), preds.data.numpy(), c='#74BCFF', lw=3, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()
