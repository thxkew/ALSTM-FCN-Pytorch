import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys):
        # query: [seq_len, batch_size, hidden_size]
        # keys: [seq_len, batch_size, hidden_size]

        query = query.unsqueeze(0)  # [1, batch_size, hidden_size]

        energy = torch.tanh(self.W_q(query) + self.W_k(keys))  # [seq_len, batch_size, hidden_size]

        attention_scores = self.V(energy).squeeze(-1)  # [seq_len, batch_size]
        attention_weights = torch.softmax(attention_scores, dim=0)  # [seq_len, batch_size]

        # Expand dimensions of attention_weights for broadcasting
        attention_weights_expanded = attention_weights.unsqueeze(2)  # shape: [seq_length, batch_size, 1]

        # Multiply attention weights with keys to get weighted sum
        context_vector = torch.sum(keys * attention_weights_expanded, dim=0)  # shape: [batch_size, 256]

        return context_vector, attention_weights

class ALSTM_FCN(nn.Module):
    def __init__(self, input_size, lstm_input_size, num_classes, momentum=0.99, eps=0.001):
        super(ALSTM_FCN, self).__init__()

        self.lstm = nn.LSTM(lstm_input_size, 8)
        self.lstm_dropout = nn.Dropout(p=0.8)

        self.attention = BahdanauAttention(128)
        
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
        outputs, (h, c) = self.lstm(x_lstm)
        context, attention_weights = self.attention(h[-1], outputs)
        x_lstm = self.lstm_dropout(context)

        x = torch.cat((x_lstm, x_fcn),dim=1)
        x = self.fc(x)

        return x
