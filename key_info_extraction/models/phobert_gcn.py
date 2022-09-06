import torch
from torch_geometric.nn import GCNConv
from key_info_extraction.utils import ID2LABEL
import torch.nn.functional as F
from torch import nn


class BERTxGCN(torch.nn.Module):

    def __init__(self, n_classes=len(ID2LABEL), hidden_size=768, dropout_rate=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

        self.conv1 = GCNConv(self.hidden_size + 2, 128, improved=True)
        self.conv2 = GCNConv(128, self.n_classes, improved=True)

    def forward(self, data):
        edge_index, edge_weight = data.edge_index, data.edge_attr
        pooled_output = self.dense(data.embedding)
        x = self.activation(pooled_output)
        x = torch.cat((x, data.p_num, data.text_len), dim=1)
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                      p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x
