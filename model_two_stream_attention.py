import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from wide_resnet101 import resnext50_32x4d

import torch.utils.model_zoo as model_zoo
import os
import sys





##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, sequence_length):
        super(Attention, self).__init__()
        self.attention_module = nn.Sequential(
            nn.Linear(latent_dim*sequence_length, 1)
            )

    def forward(self, x):
        batch_size, others = x.shape
        attention_w = F.sigmoid(self.attention_module(x).squeeze(-1))
        attention_w = attention_w.view(batch_size, 1)


        return attention_w


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1,
                                               width * height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dim=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map

class CAModule(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    code reference:
    https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    """

    def __init__(self, num_channels, reduc_ratio=2):
        super(CAModule, self).__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio,
                             bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels,
                             bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # attention branch--squeeze operation
        gap_out = feat_map.view(feat_map.size()[0], self.num_channels,
                                -1).mean(dim=2)

        # attention branch--excitation operation
        fc1_out = self.relu(self.fc1(gap_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))

        # attention operation
        fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1, 1)
        feat_map = torch.mul(feat_map, fc2_out)

        return feat_map










##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, attention_1, attention_2):
        super(Encoder, self).__init__()
        self.attention = [attention_1, attention_2]
        self.frame_attention = SAModule()
        resnet_x = resnext50_32x4d(pretrained=True)
        resnet_y = resnext50_32x4d(pretrained=True)
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.feature_extractor= nn.Sequential(*list(resnet_x.children())[:-1])
        self.feature_extractor_y= nn.Sequential(*list(resnet_y.children())[:-1])
        self.final= nn.Sequential(
            nn.AlphaDropout(0.4),
            nn.Linear(resnet_x.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01)
        )


        self.final_y= nn.Sequential(
            nn.AlphaDropout(0.4),
            nn.Linear(resnet_y.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

        #self.attention_s = Attention(latent_dim, sequence_length)
        self.attention_s = CAModule(latent_dim*sequence_length)

    def forward(self, x,y, batch_size):
        
        #with torch.no_grad():
        x = self.feature_extractor(x)    
        y = self.feature_extractor_y(y)
        
        if self.attention[0]:
            for i in self.sequence_length:
                x[:,i,:,:,:] = self.frame_attention(x[:,i,:,:,:])
                y[:,i,:,:,:] = self.frame_attention(y[:,i,:,:,:])


        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        

        
        x = self.final(x)
        y = self.final_y(y)

        
        if self.attention[1]:
            x=x.view(batch_size, self.sequence_length*self.latent_dim, 1, 1)
            y=y.view(batch_size, self.sequence_length*self.latent_dim, 1, 1)

        


            # attention_w_x = self.attention_s(x)
            # attention_w_y = self.attention_s(y)



            # x = torch.mul(x, attention_w_x)
            # y = torch.mul(y, attention_w_y)

            x = self.attention_s(x)
            y = self.attention_s(y)

        

        x= x.view(batch_size*self.sequence_length, self.latent_dim)
        y= y.view(batch_size*self.sequence_length, self.latent_dim)

        z=[x,y]


        z = torch.cat(z, 1)

        
        return z


##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(2*latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x



##############################
#         ConvLSTM
##############################


class ConvLSTM(nn.Module):
        self, num_classes, latent_dim
        def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention_1=True, attention_2=True, sequence_length=40
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim, sequence_length, attention_1, attention_2)
        self.lstm_c = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers_c = nn.Sequential(
            nn.Linear(2* hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.AlphaDropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )
        self.attention = attention
        self.attention_layer_c = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x,y):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)

        batch_size, seq_length, c, h, w = y.shape
        y = y.view(batch_size * seq_length, c, h, w)


        x = self.encoder(x,y, batch_size)

        x = x.view(batch_size, seq_length, -1)
        x = self.lstm_c(x)
       
        # if self.attention:
        #     attention_w = F.sigmoid(self.attention_layer_c(x).squeeze(-1))
        #     x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        # else:
        x = x[:, -1]
        return self.output_layers_c(x)