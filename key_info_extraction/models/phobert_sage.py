from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from key_info_extraction.utils import ID2LABEL

from transformers import AutoModel, AutoTokenizer


class SageNet(nn.Module):
    def __init__(self, in_channels, n_classes=len(ID2LABEL.keys()), dropout_rate=0.2, bert_model='vinai/phobert-base'):
        super(SageNet, self).__init__()
        self.embedding = AutoModel.from_pretrained(bert_model)
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=64)
        self.relu1 = nn.ReLU()
        self.conv2 = SAGEConv(in_channels=64, out_channels=32)
        self.relu2 = nn.ReLU()
        self.conv3 = SAGEConv(in_channels=32, out_channels=16)
        self.relu3 = nn.ReLU()
        self.conv4 = SAGEConv(in_channels=16, out_channels=n_classes)

        self.dropout_rate = dropout_rate

    def forward(self, data):

        edge_index, edge_weight = data.edge_index, data.edge_attr
        bert_output = self.embedding(attention_mask=data.attention_mask,
                                     input_ids=data.input_ids)
        x = bert_output['last_hidden_state'][:, 0]

        x = F.dropout(self.relu1(self.conv1(x, edge_index, edge_weight)), p=self.dropout_rate)
        x = F.dropout(self.relu2(self.conv2(x, edge_index, edge_weight)), p=self.dropout_rate)
        x = F.dropout(self.relu3(self.conv3(x, edge_index, edge_weight)), p=self.dropout_rate)
        x = self.conv4(x, edge_index, edge_weight)

        return x
