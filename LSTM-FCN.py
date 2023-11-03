import torch.nn as nn
import torch

class LSTM_FCN(nn.Module):
    def __init__(self, input_size, momentum=0.99, eps=0.001):
        super(LSTM_FCN, self).__init__()
        
        self.lstm = nn.LSTM(240, 8)
        self.lstm_dropout = nn.Dropout(p=0.8)
        
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
        
        
        self.fc = nn.Linear(136, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Input : x has dimensions [B, C, L]
        Output : x has dimentions [B, 1]

        Where B: batch size, C: number of channel, L:sequence length
        """

        #FCN branch
        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        #Dimention Shuffle Layer
        x_lstm = x.clone().permute(1, 0, 2)

        #LSTM branch
        _, (h, c) = self.lstm(x_lstm)
        x_lstm = self.lstm_dropout(h)
        x_lstm = torch.squeeze(x_lstm, dim=0)

        x = torch.cat((x_lstm, x_fcn),dim=1)
        x = self.fc(x)

        return x