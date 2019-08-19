# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from central_pooling import central_pooling
from torchsummary import summary

class conv_prelu(nn.Module):
    '''(conv => BN => PReLU) '''
    def __init__(self, in_ch, out_ch):
        super(conv_prelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CF_CNN(nn.Module):
    def __init__(self):
        super(CF_CNN, self).__init__()
        self.conv1_2d = conv_prelu(2, 36)
        self.conv2_2d = conv_prelu(36, 36)
        self.conv3_2d = conv_prelu(36, 48)
        self.central_pooling1_2d = central_pooling(35)

        self.conv4_2d = conv_prelu(48, 48)
        self.conv5_2d = conv_prelu(48, 68)
        self.central_pooling2_2d = central_pooling(18)
        self.conv6_2d = conv_prelu(68, 68)

        self.fc1_2d = nn.Linear(68*9*9, 300)
        self.prelu_2d = nn.PReLU(num_parameters=1, init=0.25)
        self.batch_norm1_2d = nn.BatchNorm1d(300)
        self.fc2_2d = nn.Linear(300, 2)

        self.conv1_3d = conv_prelu(3, 36)
        self.conv2_3d = conv_prelu(36, 36)
        self.conv3_3d = conv_prelu(36, 48)
        self.central_pooling1_3d = central_pooling(35)

        self.conv4_3d = conv_prelu(48, 48)
        self.conv5_3d = conv_prelu(48, 68)
        self.central_pooling2_3d = central_pooling(18)
        self.conv6_3d = conv_prelu(68, 68)

        self.fc1_3d = nn.Linear(68*9*9, 300)
        self.prelu_3d = nn.PReLU(num_parameters=1, init=0.25)
        self.batch_norm1_3d = nn.BatchNorm1d(300)
        self.fc2_3d = nn.Linear(300, 2)

        self.initialize()

    def forward(self, x_2d, x_3d):

        x_2d = self.conv1_2d(x_2d)
        x_2d = self.conv2_2d(x_2d)
        x_2d = self.conv3_2d(x_2d)
        x_2d = self.central_pooling1_2d(x_2d)
        x_2d = self.conv4_2d(x_2d)
        x_2d = self.conv5_2d(x_2d)
        x_2d = self.central_pooling2_2d(x_2d)
        x_2d = self.conv6_2d(x_2d)

        x_2d = x_2d.view(-1, 68 * 9 * 9)
        x_2d = self.fc1_2d(x_2d)
        x_2d = self.prelu_2d(x_2d)
        x_2d = self.batch_norm1_2d(x_2d)
        x_2d = self.fc2_2d(x_2d)

        x_3d = self.conv1_3d(x_3d)
        x_3d = self.conv2_3d(x_3d)
        x_3d = self.conv3_3d(x_3d)
        x_3d = self.central_pooling1_3d(x_3d)
        x_3d = self.conv4_3d(x_3d)
        x_3d = self.conv5_2d(x_3d)
        x_3d = self.central_pooling2_3d(x_3d)
        x_3d = self.conv6_3d(x_3d)

        x_3d = x_3d.view(-1, 68 * 9 * 9)
        x_3d = self.fc1_3d(x_3d)
        x_3d = self.prelu_3d(x_3d)
        x_3d = self.batch_norm1_3d(x_3d)
        x_3d = self.fc2_3d(x_3d)
        return x_2d, x_3d

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')



if __name__=='__main__':

    base_model = CF_CNN()
    a_2d = torch.randn(8,2,35,35)
    a_3d = torch.randn(8,3,35,35)
    #summary(base_model, input_size=(2, 35, 35))
    output = base_model(a_2d,a_3d)

    

        
    
    
    
    