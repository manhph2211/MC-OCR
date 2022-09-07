import json
import string
import os.path as osp
import os

import networkx as nx
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from .utils import ID2LABEL, LABEL2ID
from transformers import AutoTokenizer, AutoModel


class Receipt(Dataset):
    def __init__(self, json_file, bert_model="vinai/phobert-base"):
        super(Receipt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        print(self.data)
        self.filenames = list(self.data.keys())
        self.embedding = AutoModel.from_pretrained(bert_model)

    def connect(self, bboxes, imgw, imgh):
        G = nx.Graph()
        for src_idx, src_row in enumerate(bboxes):
            if not src_row.get('label', None):
                src_row['label'] = "OTHER"
            src_row['y'] = torch.tensor([LABEL2ID[src_row['label']]], dtype=torch.long)
            src_row["x_max"], src_row["y_max"], src_row["x_min"], src_row["y_min"] = src_row["crop"][0][0], \
                                                                                     src_row["crop"][0][1], \
                                                                                     src_row["crop"][1][0], \
                                                                                     src_row["crop"][1][1]
            src_row['bbox'] = list(map(float, [
                src_row["x_min"] / imgw,
                src_row["y_min"] / imgh,
                src_row["x_max"] / imgw,
                src_row["y_max"] / imgh
            ]))

            if not len(src_row['text']):
                p_num = 0.0

            else:
                p_num = sum([n in string.digits for n in src_row['text']]) / len(src_row['text'])

            G.add_node(
                src_idx,
                text=src_row['text'],
                bbox=src_row['bbox'],
                label=src_row['label'],
                p_num=p_num,
                y=src_row['y']
            )
            src_range_x = (src_row["x_min"], src_row["x_max"])
            src_range_y = (src_row["y_min"], src_row["y_max"])

            src_center_x, src_center_y = np.mean(src_range_x), np.mean(src_range_y)

            neighbor_vert_top = []
            neighbor_vert_bot = []
            neighbor_hozi_left = []
            neighbor_hozi_right = []

            for dest_idx, dest_row in enumerate(bboxes):
                if dest_idx == src_idx:
                    continue
                dest_row["x_min"], dest_row["y_min"], dest_row["x_max"], dest_row["y_max"] = dest_row["crop"][1][0], \
                                                                                             dest_row["crop"][1][1], \
                                                                                             dest_row["crop"][0][0], \
                                                                                             dest_row["crop"][0][1]
                dest_range_x = (dest_row["x_min"], dest_row["x_max"])
                dest_range_y = (dest_row["y_min"], dest_row["y_max"])
                dest_center_x, dest_center_y = np.mean(dest_range_x), np.mean(dest_range_y)

                # Find box in vertical must have common x range.
                if max(src_range_x[0], dest_range_x[0]) <= min(src_range_x[1], dest_range_x[1]):
                    # Find underneath box: neighbor yminx must be smaller than source ymax
                    if dest_range_y[0] >= src_range_y[1]:
                        neighbor_vert_bot.append(dest_idx)
                    elif dest_range_y[1] <= src_range_y[0]:
                        neighbor_vert_top.append(dest_idx)
                # Find box in horizontal must have common y range.
                if max(src_range_y[0], dest_range_y[0]) <= min(src_range_y[1], dest_range_y[1]):
                    # Find right box: neighbor xmin must be smaller than source xmax
                    if dest_range_x[0] >= src_range_x[1]:
                        neighbor_hozi_right.append(dest_idx)
                    elif dest_range_x[1] <= src_range_x[0]:
                        neighbor_hozi_left.append(dest_idx)

            neighbors = []

            if neighbor_hozi_left:
                nei = min(neighbor_hozi_left, key=lambda x: src_center_x - bboxes[x]['x_max'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)
            if neighbor_hozi_right:
                nei = min(neighbor_hozi_right,
                          key=lambda x: bboxes[x]['x_min'] - src_center_x)
                neighbors.append(nei)
                G.add_edge(src_idx, nei)
            if neighbor_vert_bot:
                nei = min(neighbor_vert_bot, key=lambda x: bboxes[x]['y_min'] - src_center_y)
                neighbors.append(nei)
                G.add_edge(src_idx, nei)
            if neighbor_vert_top:
                nei = min(neighbor_vert_top, key=lambda x: src_center_y - bboxes[x]['y_max'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)

        return G

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]
        filepath = '../data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images/' + filename + '.jpg'
        bboxes = self.data[filename]

        G = self.connect(bboxes=bboxes,
                         imgw=1000, imgh=1000)
        # nx.draw(G,node_size= 20)
        # if not os.path.isfile('./vis/image_'+str(idx)+'.png'):
        #     plt.savefig('./vis/image_'+str(idx)+'.png')
        data = from_networkx(G)

        token = self.tokenizer(data.text, add_special_tokens=True, truncation=True,
                               max_length=128, padding='max_length', return_tensors='pt')
        data.input_ids, data.attention_mask = token.input_ids, token.attention_mask
        data.text_len = torch.count_nonzero(data.input_ids, dim=1) / 128.0
        data.text_len = torch.unsqueeze(data.text_len, dim=1)
        data.bbox = torch.Tensor(data.bbox)
        data.p_num = torch.Tensor(data.p_num)
        data.p_num = torch.unsqueeze(data.p_num, dim=1)

        data.path = filepath
        data.imname = self.filenames[idx]
        bert_output = self.embedding(attention_mask=data.attention_mask,
                                     input_ids=data.input_ids)
        data.embedding = bert_output['last_hidden_state'][:, 0]

        return data


if __name__ == '__main__':
    dataset = Receipt(json_file='../data/labeled_predict.json')
    print(dataset[4])
    pass