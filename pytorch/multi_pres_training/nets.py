import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
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
