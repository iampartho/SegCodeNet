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



##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim, sequence_length):
        super(Encoder, self).__init__()
        resnet_x = resnext50_32x4d(pretrained=True)
        resnet_y = resnext50_32x4d(pretrained=True)
        trained_kernel = resnet_y.conv1.weight
        new_conv = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*2, dim=1)
        resnet_y.conv1 = new_conv
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.feature_extractor= nn.Sequential(*list(resnet_x.children())[:-1])
        self.feature_extractor_z= nn.Sequential(*list(resnet_y.children())[:-1])
        self.final= nn.Sequential(
            nn.AlphaDropout(0.4),
            nn.Linear(resnet_x.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01)
        )


        self.final_z= nn.Sequential(
            nn.AlphaDropout(0.4),
            nn.Linear(resnet_y.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

        self.attention_x = Attention(latent_dim, sequence_length)
        self.attention_y = Attention(latent_dim, sequence_length)

    def forward(self, x,y, batch_size):
        
        #with torch.no_grad():
        x = self.feature_extractor(x)    
        y = self.feature_extractor_z(y)
        

        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        

        
        x = self.final(x)
        y = self.final_z(y)

        
        
        x=x.view(batch_size, self.sequence_length*self.latent_dim)
        y=y.view(batch_size, self.sequence_length*self.latent_dim)

    


        attention_w_x = self.attention_x(x)
        attention_w_y = self.attention_y(y)



        x = torch.mul(x, attention_w_x)
        y = torch.mul(y, attention_w_y)

        

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
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True, sequence_length=40
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim, sequence_length)
        self.lstm_c = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers_c = nn.Sequential(
            nn.Linear(2* hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.AlphaDropout(0.4),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
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
       
        if self.attention:
            attention_w = F.sigmoid(self.attention_layer_c(x).squeeze(-1))
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers_c(x)