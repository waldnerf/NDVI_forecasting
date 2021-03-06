import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import random


# Model
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout, bidirectional=bidirectional)
        hid_dim_fc = (hid_dim * 2 if bidirectional else hid_dim)
        self.fc_out = nn.Linear(hid_dim_fc, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size, n_dims]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        input = input.unsqueeze(0)
        # input : [1, ,batch_size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, output_dim, output_len, bidirectional, device):
        super().__init__()

        self.encoder = EncoderLSTM(input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional)
        self.decoder = DecoderLSTM(output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional)
        self.device = device
        self.max_len = output_len

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src = L,N,H
        # trg = L,N,H
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = src.shape[1]
        max_len = self.max_len

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # only use y initial y
        output = src[-1, :, :]
        for t in range(0, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(output, hidden, cell)
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t, :, :] if teacher_force else output)
        return outputs

# EOF