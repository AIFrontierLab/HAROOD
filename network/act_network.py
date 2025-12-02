# coding=utf-8
import torch.nn as nn
import torch
from network.tst.transformer import TransformerFea, TransformerFeaLAG


def transformfea(args):
    d_model = 64
    q = 8
    v = 8
    h = 8
    N = 2
    attention_size = 12
    dropout = 0.2
    pe = None
    chunk_mode = None
    d_input = args.input_shape[-1]  # *args.input_shape[0]
    net = TransformerFea(
        d_input,
        d_model,
        args.input_shape[0],
        q,
        v,
        h,
        N,
        attention_size=attention_size,
        dropout=dropout,
        chunk_mode=chunk_mode,
        pe=pe,
    )
    return net


def transformfea_LAG(args):
    d_model = 64
    q = 8
    v = 8
    h = 8
    N = 2
    attention_size = 12
    dropout = 0.2
    pe = None
    chunk_mode = None
    net = TransformerFeaLAG(
        args, d_model, q, v, h, N, attention_size, dropout, chunk_mode, pe
    )
    return net


var_size = {
    "dsads": {
        "in_size": 45,
        "ker_size": 9,
        "fc_size": 32 * 25,
    },
    "ddsads": {
        "in_size": 15,
        "ker_size": 5,
        "fc_size": 32 * 28,
    },
    "usc": {"in_size": 6, "ker_size": 6, "fc_size": 32 * 46},
    "opp": {
        "in_size": 45,
        "ker_size": 9,
        "fc_size": 32 * 44,
    },
    "realworld": {"in_size": 9, "ker_size": 9, "fc_size": 1408},
    "pamap": {"in_size": 27, "ker_size": 9, "fc_size": 32 * 44},
    "emg": {"in_size": 8, "ker_size": 9, "fc_size": 32 * 44},
    "shemg": {"in_size": 8, "ker_size": 5, "fc_size": 32 * 22},
    "loemg": {"in_size": 8, "ker_size": 9, "fc_size": 32 * 119},
    "wesad": {"in_size": 8, "ker_size": 9, "fc_size": 32 * 44},
    "har": {"in_size": 6, "ker_size": 9, "fc_size": 32 * 26},
    "pdsads": {
        "in_size": 9,
        "ker_size": 9,
        "fc_size": 32 * 25,
    },
    "pusc": {"in_size": 6, "ker_size": 6, "fc_size": 32 * 46},
    "ppamap": {"in_size": 9, "ker_size": 9, "fc_size": 32 * 44},
    "phar": {"in_size": 6, "ker_size": 9, "fc_size": 32 * 26},
    "cross_dataset": {"in_size": 6, "ker_size": 6, "fc_size": 32 * 8},
}


class SActNetwork(nn.Module):
    def __init__(self, taskname):
        super(SActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]["in_size"],
                out_channels=16,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.in_features = 16 * 96
        self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.in_features)
        x = self.activation(x)
        return x


class LActNetwork(nn.Module):
    def __init__(self, taskname):
        super(LActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]["in_size"],
                out_channels=16,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.in_features = 32 * 5
        self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.view(-1, self.in_features)
        x = self.activation(x)
        return x


class SimpleTwoLayerRNN(nn.Module):
    def __init__(
        self,
        dataset,
        input_channels=8,  # channels dimension of input
        hidden_size=128,
        num_layers=2,
        rnn_type="GRU",  # 'GRU' or 'LSTM' or 'RNN'
        bidirectional=False,
        dropout=0.2,
    ):
        super().__init__()
        self.dataset = dataset
        self.input_size = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                nonlinearity="tanh",
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # default GRU
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.in_features = hidden_size * self.num_directions

    def forward(self, x):
        """
        x: (batch, channels=8, 1, length=200)
        returns logits: (batch, num_classes)
        """
        # remove middle singleton dim if exists -> (batch, channels, length)
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)  # (B, C, L)
        elif x.dim() == 4:
            # keep as is but merge dims appropriately: assume shape (B, C, 1, L)
            x = x.view(x.size(0), x.size(1), -1)  # (B, C, L)

        # RNN expects (batch, seq_len, input_size)
        x = x.permute(0, 2, 1).contiguous()  # (B, L, C) where C == input_size

        # Apply RNN
        # output: (B, L, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, B, hidden_size)  (or tuple for LSTM)
        output, h_n = self.rnn(x)

        # get last layer hidden state:
        if self.rnn_type == "LSTM":
            # h_n is (h_n, c_n) tuple for LSTM
            h_n = h_n[0]

        # h_n shape: (num_layers * num_directions, B, hidden_size)
        # we want last layer's forward/backward states:
        if self.num_directions == 1:
            last_h = h_n[-1]  # (B, hidden_size)
        else:
            # concatenate forward and backward last layer states
            # forward is -2, backward is -1 when num_layers>=1
            last_h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_size*2)
        return last_h


class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]["in_size"],
                out_channels=16,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.in_features = var_size[taskname]["fc_size"]
        self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        x = self.activation(x)
        return x

    def getfea(self, x):
        x = self.conv2(self.conv1(x))
        return x


class ActNetwork_LAG_CNN(nn.Module):
    def __init__(self, taskname, patch_nums=5):
        super(ActNetwork_LAG_CNN, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]["in_size"],
                out_channels=16,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, var_size[taskname]["ker_size"]),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.in_features = var_size[taskname]["fc_size"]
        self.patch_nums = patch_nums
        self.conv11 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]["in_size"],
                out_channels=16,
                kernel_size=(self.patch_nums, int(var_size[taskname]["ker_size"] / 2)),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, int(var_size[taskname]["ker_size"] / 2)),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.net1 = nn.Sequential(self.conv11, self.conv12)

    def init(self, x):
        x = x.view(-1, x.shape[1], self.patch_nums, int(x.shape[3] / self.patch_nums))
        x = self.conv12(self.conv11(x))
        self.in_features1 = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x1 = x.view(-1, x.shape[1], self.patch_nums, int(x.shape[3] / self.patch_nums))
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)

        x1 = self.net1(x1)
        x1 = x1.view(-1, self.in_features1)
        return x, x1
