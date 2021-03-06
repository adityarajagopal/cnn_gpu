from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sf', type=float, default=128.0, metavar='N',
                    help='scaling factor initialisation')
parser.add_argument('--fp16', type=bool, default=False, metavar='N',
                    help='run mixed precision training or not')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5,5)), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=2), 
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=2), 
            nn.Conv2d(16, 120, kernel_size=(5,5)),
            nn.ReLU() 
        )
        
        self.fullyConn = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.convNet(x)
        x = x.view(-1,120)
        x = self.fullyConn(x)
        return x

model = Net()
if args.cuda:
    if args.fp16 : 
        model.cuda().half()
    else : 
        model.cuda()

masterCopy = None
if args.fp16 : 
    masterCopy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            if args.fp16 : 
                data, target = data.cuda().half(), target.cuda()
            else : 
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        params = list(model.parameters())
        if args.fp16 : 
            for i in range(len(params)) : 
                params[i].data = masterCopy[i].data.half()
        
        output = model.forward(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        if args.fp16 :
            loss = loss * args.sf
        loss.backward()

        if args.fp16 :   
            # applyGrad = True
            for param in params : 
                param.grad.data /= args.sf
            # if applyGrad == True : 
            for i in range(len(masterCopy)) :
                masterCopy[i].data -= (args.lr * params[i].grad.data.type(torch.cuda.FloatTensor))

        optimizer.step()
        
        # if args.fp16 : 
        #     for i in range(len(params)) : 
        #         params[i].data = masterCopy[i].data.half()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.data[0]/args.sf))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            if args.fp16 : 
                data, target = data.cuda().half(), target.cuda()
            else : 
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model.forward(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
