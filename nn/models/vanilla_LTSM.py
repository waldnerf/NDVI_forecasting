import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import random



# Model
class Encoder_LSTM(nn.Module):
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
        return outputs, hidden, cell

class Global_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Global_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim + self.dec_hid_dim, self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # only pick up last layer hidden
        hidden = torch.unbind(hidden, dim=0)[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)

class AttDecoder_LSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_dim, emb_dim)

        self.attention_layer = Global_Attention(hid_dim, hid_dim)
        self.rnn = nn.LSTM(input_size=hid_dim+emb_dim, hidden_size=hid_dim, num_layers=self.n_layers, dropout=dropout, bidirectional=bidirectional)
        hid_dim_fc = (hid_dim * 2 if bidirectional else hid_dim)
        self.fc_out = nn.Linear(hid_dim_fc, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch_size, n_dims]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        input = input.unsqueeze(0)
        # input : [1, batch_size, n_dims]

        embedded = self.dropout(torch.tanh(self.embedding(input)))
        # embedded = [1, batch_size, emb_dim]

        # attention mechanism
        a = self.attention_layer(hidden, encoder_outputs)
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, weighted), dim=2)


        # RNN layer
        output, (hidden, cell) = self.rnn(lstm_input, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq_aLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, output_dim, output_len, bidirectional, device):
        super().__init__()

        self.encoder = Encoder_LSTM(input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional)
        self.decoder = AttDecoder_LSTM(output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional)
        self.device = device
        self.max_len = output_len

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src = L,N,H
        # trg = L,N,H
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = src.shape[1]
        max_len = self.max_len

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size,  self.decoder.output_dim).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden, cell = self.encoder(src)

        # only use y initial y
        output = src[-1, :, :]
        for t in range(0, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(output, hidden, cell, encoder_outputs)
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t, :, :] if teacher_force else output)
        return outputs

INPUT_DIM = 1
OUTPUT_DIM = 1
EMB_DIM = 128
HID_DIM = 128
N_LAYERS = 2
DROPOUT = 0.3
OUTPUT_LEN = 33
BIDIRECTIONAL = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq_aLSTM(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, OUTPUT_DIM, OUTPUT_LEN, BIDIRECTIONAL, device).to(device)

model.apply(init_weights)

class CustomRunner_LSTM(dl.Runner):
    def handle_batch(self, batch):
        x, y = batch
        x = torch.unsqueeze(x, 2)
        y = torch.unsqueeze(y, 2)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        if self.is_train_loader:
            outputs = self.model(x, y, 0.5)
        else:
            outputs = self.model(x, y, 0)

        loss = F.mse_loss(outputs, y)
        # output_numpy = output.cpu().data.numpy()
        # y_numpy = y.cpu().data.numpy()

        # self.batch_metrics = {
        #     "loss": loss
        # }

        self.batch = {
            "outputs": outputs,
            "preds": y,
        }


