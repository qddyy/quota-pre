import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""


class Args:
    def __init__(
        self,
        input_size,
        lstm_hidden_dim,
        lstm_num_layers,
        batch_first,
        batch_size,
        class_num,
        kernel_num,
        kernel_sizes,
        cuda=False,
    ) -> None:
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.input_size = input_size
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.cuda = cuda
        self.batch_first = batch_first
        self.batch_size = batch_size


class CNN_LSTM(nn.Module):

    def __init__(self, args: Args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        C = args.class_num
        Ci = 1  # 输入通道数=batch_size
        Co = args.kernel_num  # 卷积核数量
        Ks = args.kernel_sizes  # 卷积核大小

        # CNN
        self.convs1 = [
            nn.Conv2d(Ci, Co, (K, 5), padding=((K - 1) // 2, 0)) for K in Ks
        ]  # (kernel_num, high, width)

        # LSTM
        self.lstm = nn.LSTM(
            args.input_size,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # linear
        L = self.hidden_dim + (args.input_size - 4) * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

    def forward(self, x: torch.Tensor):
        # CNN
        cnn_x = x.unsqueeze(1)
        # cnn_x = torch.transpose(cnn_x, 0, 1)
        # cnn_x = x  # (batch_size, ci, seq_len, input_dim)
        cnn_x = [F.relu(conv(cnn_x)) for conv in self.convs1]
        # [(batch_size,kernel_num, high,width), ...]*len(Ks)
        cnn_x = [torch.transpose(i, 1, 3) for i in cnn_x]
        cnn_x = [F.max_pool2d(i, (1, i.size(3))).squeeze(3) for i in cnn_x]
        cnn_x = torch.cat(cnn_x, 1)  # (batch_size,high,seq_len)
        cnn_x = self.dropout(cnn_x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size,seq_len, hidden_size)

        # CNN and LSTM cat
        cnn_x = torch.transpose(cnn_x, 1, 2)
        cnn_lstm_out = torch.cat(
            (cnn_x, lstm_out), 2
        )  # (batch_size,seq_len, hidden_size+high)
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))

        # output
        logit = cnn_lstm_out
        return logit
