import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, rnn_type, dropout, bidirectional):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_grulstm_layers = num_grulstm_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,
                               dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, input, hidden):  # input [batch_size, length T, dimensionality d]
        if self.rnn_type == 'GRU':
            output, hidden = self.rnn(input, hidden)
            return output, hidden
        elif self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(input, hidden)
            return output, (hidden, cell)

    def init_hidden(self, x,  device):
        # [num_layers*num_directions,batch,hidden_size]
        if self.bidirectional:
            return torch.zeros(2*self.num_grulstm_layers, x.shape[0], self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_grulstm_layers, x.shape[0], self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, fc_units, output_size, rnn_type,
                 dropout, bidirectional):
        super(DecoderRNN, self).__init__()
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,
                               dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = F.relu(self.fc(output))
        output = self.out(output)
        return output, hidden

class VanillaRNN(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, enc_layers, dec_hid_dim, dec_layers, fc_layers, output_dim,
                 target_length, rnn_type, dropout, bidirectional, device):
        super(VanillaRNN, self).__init__()


        self.encoder = EncoderRNN(input_dim, enc_hid_dim, enc_layers, rnn_type, dropout, bidirectional)
        self.decoder = DecoderRNN(input_dim, dec_hid_dim, dec_layers, fc_layers, output_dim, rnn_type, dropout, bidirectional)

        self.target_length = target_length
        self.device = device

    def forward(self, x):
        input_length = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(x, self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei + 1, :], encoder_hidden)

        decoder_input = x[:, -1, :].unsqueeze(1)  # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di + 1, :] = decoder_output
        return outputs
