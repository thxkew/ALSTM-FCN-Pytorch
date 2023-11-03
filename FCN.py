import torch.nn as nn
import torch

class FCN(nn.Module):
    def __init__(self, input_size, num_classes, momentum=0.99, eps=0.001):
        super(FCN, self).__init__()
        
        self.feature = nn.Sequential(  

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
        
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Input : x has dimensions [B, C, L]
        Output : x has dimentions [B, N]

        Where B: batch size, C: number of channel, L:sequence length, N: number of classes
        """
        
        x = self.feature(x)
        x = torch.mean(x, 2)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x