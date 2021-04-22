"""
created by PyCharm
date: 2021/4/11
time: 18:19
user: hxf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, text_in_features, digital_in_features):
        super(Network, self).__init__()
        self.bn_text = nn.BatchNorm1d(text_in_features)
        self.bn_digital = nn.BatchNorm1d(digital_in_features)
        self.fc_text_1 = nn.Linear(in_features=text_in_features, out_features=128)
        self.fc_digital_1 = nn.Linear(in_features=digital_in_features, out_features=64)
        self.fc_connect_1 = nn.Linear(in_features=192, out_features=256)
        self.fc_connect_2 = nn.Linear(in_features=256, out_features=128)
        self.fc_connect_3 = nn.Linear(in_features=128, out_features=64)
        self.fc_connect_4 = nn.Linear(in_features=64, out_features=2)

    def forward(self, text_input, digital_input):
        text_input_bn = self.bn_text(text_input)
        text_1 = self.fc_text_1(text_input_bn)
        text_1 = F.softsign(text_1)

        digital_input_bn = self.bn_digital(digital_input)
        digital_1 = self.fc_digital_1(digital_input_bn)
        digital_1 = torch.tanh(digital_1)

        t_d_connect = torch.cat([text_1, digital_1], 1)
        t_d_connect = torch.tanh(self.fc_connect_1(t_d_connect))
        t_d_connect = torch.tanh(self.fc_connect_2(t_d_connect))
        t_d_connect = torch.tanh(self.fc_connect_3(t_d_connect))
        t_d_connect = nn.Dropout(p=0.5)(t_d_connect)
        t_d_connect = self.fc_connect_4(t_d_connect)
        return t_d_connect
