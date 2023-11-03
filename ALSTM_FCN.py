import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_key = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x_t, h_prev):
        """
        Input:  x_t has dimensions [B, C]
                h_prev has dimentions [B, H]

        Output: context has dimensions [B, 1]

        Where B: batch size, C: number of channel, H: hidden size
        """

        query = self.W_query(h_prev)
        keys = self.W_key(x_t)

        # Batch matrix multiplication
        energy = self.V(torch.tanh(query + keys))
        attention_weights = F.softmax(energy, dim=1)

        context = torch.sum(x_t * attention_weights, dim=1).unsqueeze(-1)
        
        return context

class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTMCell, self).__init__()

        self.lstm_cell = torch.nn.LSTMCell(input_size+1, hidden_size)
        self.attention = Attention(input_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        Input:  x_t has dimensions [B, C]
                h_prev has dimensions [B, H]
                c_prev has dimensions [B, H]

        Output: h_t has dimensions [B, H]
                c_t has dimensions [B, H]

        Where B: batch size, C: number of channel, H: hidden size
        """

        context = self.attention(x_t, h_prev)

        lstm_input = torch.cat((x_t, context), dim=1)
        h_t, c_t = self.lstm_cell(lstm_input, (h_prev, c_prev))

        return h_t, c_t


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = AttentionLSTMCell(self.input_size, self.hidden_size)

    def forward(self, x, init_states=None):
        """
        Input : x has dimensions [L, B, C]
        
        Output: h_t has dimensions [B, H]
                c_t has dimensions [B, H]
        
        (L: sequence lenght, B: batch size, C: number of channel) 
        """

        L, B, _ = x.size()

        h_t, c_t = (torch.zeros(B, self.hidden_size).to(x.device),
                    torch.zeros(B, self.hidden_size).to(x.device)) if init_states is None else init_states

        for i in range(L):

            h_t, c_t = self.cell(x[i], h_t, c_t)

        return h_t, c_t

class ALSTM_FCN(nn.Module):
    def __init__(self, input_size, num_classes, momentum=0.99, eps=0.001):
        super(ALSTM_FCN, self).__init__()
        
        self.avg = nn.AdaptiveAvgPool1d(240)
        
        self.alstm = AttentionLSTM(input_size=240, hidden_size=8)
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
        
        
        self.fc = nn.Linear(136, num_classes)
    
    def forward(self, x):
        """
        Input : x has dimensions [B, C, L]
        Output : x has dimentions [B, N]

        Where B: batch size, C: number of channel, L:sequence length, N: number of classes
        """

        #FCN branch
        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        #Dimension Shuffle
        x_lstm = x.clone().permute(1, 0, 2)

        #LSTM branch
        h, c = self.alstm(x_lstm)
        x_lstm = self.lstm_dropout(h)
        x_lstm = torch.squeeze(x_lstm, dim=0)

        x = torch.cat((x_lstm, x_fcn),dim=1)
        x = self.fc(x)

        return x