import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim #hidden_vector 크기.
        self.n_layers = n_layers #rnn layer 수.
        #input_dum의 크기를 받아 embedding_dim크기만큼으로 embedding.
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        #embedding_dim크기만큼 받아 hidden_dim크기만큼 은닉층을 생성. 이 때, rnn 층은 n_layer만큼.
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        #src(= source sentence)
        #src를 embedding 시키고, dropout.
        embedded = self.dropout(self.embedding(src))
        #rnn에 embedded를 넣고 다음 세 값을 리턴. outputm (hidden, cell).
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Encoder의 output_dim만큼의 크기를 받아, embedding_dim만큼의 크기로 Embedding.
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        #embedding_dim 크기를 받아, hidden_dim 크기의 은닉층 생성, rnn 층은 n_layers만큼.
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        #hidden_dim을 입력받아 output_dim을 출력
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        #rnn의 인자로(embedded, (hidden, cell))을 입력으로 넣음.
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #결과 예측
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        #encoder의 hidden_dim과 decoder의 hidden_dim은 같아야함.(n_layers도)
        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #src = (src len, batch_size)
        #trg = (trg len, batch_size)
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.ouput_dim

        #shape(trg_len, batch_size, trg_vocab_size)로 빈값 생성
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        hidden, cell = self.encoder(src)

        #decoder의 첫 input은 <SOS>이다.
        input = trg[0, :]

        #1부터 시작하는 이유는 0은 <SOS>이기 때문.
        #target length만큼 반복
        for t in range(1, trg_len):
            #encoder의 output인 hidden과 cell을 decoder의 input으로 넣음.
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            #1/2 확률로, 다음 시점의 input값으로 현재 시점의 output 값을 넣을 지
            #실제 다음 시점의 input을 넣을 지 랜덤하게 결정.
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs