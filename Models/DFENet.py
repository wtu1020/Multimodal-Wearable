import torch
import torch.nn as nn
from scipy.fft import fft
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from scipy.fftpack import dct

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
class FDFEM(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to value
        out = torch.matmul(attn_weights, v)
        return out

class DFENet(nn.Module):
    def __init__(self, num_classes, num_features,
                 num_transformer_layers, num_heads,
                 conv1_nf, conv2_nf, conv3_nf, fc_drop_p):
        super(DFENet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features

        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.fc_drop_p = fc_drop_p

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_heads),
            num_layers=self.num_transformer_layers
        )

        self.conv1 = DepthwiseSeparableConv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = DepthwiseSeparableConv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = DepthwiseSeparableConv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        self.relu = nn.ReLU()
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf + self.num_features, self.num_classes)
        # self.fc = nn.Linear(self.num_features, self.num_classes)
        self.FDFEM = FDFEM(self.num_features)

    def DCT(self, x):
        """对输入进行离散余弦变换 (DCT)"""
        n, seq_len, input_size = x.shape
        x_dct = torch.empty((n, seq_len, input_size), dtype=x.dtype, device=x.device)

        for i in range(n):
            for j in range(input_size):
                x_dct[i, :, j] = torch.tensor(dct(x[i, :, j].cpu().numpy(), type=2, norm='ortho')).to(x.device)

        return x_dct

    def forward(self, x):
        x1 = self.DCT(x)
        x1 = self.transformer_encoder(x1)
        x1 = self.FDFEM(x1)
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        out_log = self.fc(x_all)
        output = F.softmax(out_log, dim=1)

        return out_log, output



