from __future__ import print_function
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from nets import LeNet

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

# setup models
model32 = LeNet()
model16 = LeNet()
if args.cuda : 
    model32.cuda()
    model16.cuda().half()
else : 
    model16.half()

masterCopy = list(model32.parameters())

# setup separate optimisers
optimiser16 = optim.SGD(model16.parameters(), lr=args.lr, momentum=args.momentum)
optimiser32 = optim.SGD(model32.parameters(), lr=args.lr, momentum=args.momentum)

iteration = []
trainLoss = []

def train16 (epoch, batch_idx, data, target, model, optimiser) : 
    model.train()
    
    if args.cuda:
        data, target = data.cuda().half(), target.cuda()
    else : 
        data, target = data.half(), target
    data, target = Variable(data), Variable(target)
    
    localParams = list (model.parameters()) 
    for i in range(len(localParams)) : 
        localParams[i].data = masterCopy[i].data.half()
    
    output = model.forward(data) 
    optimiser.zero_grad()
    loss = F.nll_loss(output, target) 
    loss = loss * args.sf 
    loss.backward()
    
    for param in localParams : 
        param.grad.data /= args.sf
    
    for i in range(len(masterCopy)) : 
        masterCopy[i].data -= (args.lr * localParams[i].grad.data.type(torch.cuda.FloatTensor))
    
    optimiser.step()
    
    if batch_idx % args.log_interval == 0:
        print('Train Epoch (half) : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.data[0]/args.sf))
        iteration.append((epoch-1) * len(train_loader.dataset) + batch_idx * len(data))
        trainLoss.append(loss.data[0]/args.sf)

def train32(epoch, batch_idx, data, target, model, optimizer):
    model.train()
    
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    
    output = model.forward(data)
    optimizer.zero_grad()
    loss = F.nll_loss(output, target)
    loss.backward()

    optimizer.step()
    
    if batch_idx % args.log_interval == 0:
        print('Train Epoch (full) : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
             100. * batch_idx / len(train_loader), loss.data[0]))
        iteration.append((epoch-1) * len(train_loader.dataset) + batch_idx * len(data))
        trainLoss.append(loss.data[0])

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
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

def main() : 
    halfTrainCount = 20 
    fullTrainCount = 10 
    halfTrain = True
    for epoch in range(1, args.epochs + 1) :
        for batch_idx, (data, target) in enumerate(train_loader) :
            if halfTrain == True : 
                halfTrainCount -= 1
                
                train16(epoch, batch_idx, data, target, model16, optimiser16)
                
                if halfTrainCount == 0 : 
                    halfTrain = False
                    halfTrainCount = 20
            else : 
                fullTrainCount -= 1
                
                train32(epoch, batch_idx, data, target, model32, optimiser32)
                
                if fullTrainCount == 0 : 
                    halfTrain = True 
                    fullTrainCount = 10
        
        test(model32)

    plt.plot(iteration, trainLoss)
    plt.show()

if __name__ == "__main__" : 
    main()



















