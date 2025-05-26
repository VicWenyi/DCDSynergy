import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgllife.model.gnn import GCN
from torch.nn import MultiheadAttention
from torch.nn.utils import weight_norm

class DrugGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(DrugGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size//2, 2 , -1, self.output_feats)
        return node_feats

class CellResidualEncoder(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(CellResidualEncoder, self).__init__()
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.res_block1 = ResidualBlock(in_ch[0], in_ch[1], kernels[0])
        self.res_block2 = ResidualBlock(in_ch[1], in_ch[2], kernels[1])

    def forward(self, v):
        v = v.transpose(2, 1)
        v = self.res_block1(v)
        v = self.res_block2(v)
        v = v.transpose(2, 1)
        return v

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.05):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.match_dim = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if x.shape[-1] != out.shape[-1]:
            residual = F.pad(residual, (0, out.shape[-1] - x.shape[-1]))

        residual = self.match_dim(residual)
        out += residual
        out = self.relu(out)
        return out

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x