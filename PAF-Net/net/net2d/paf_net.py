# -*- coding: utf-8 -*-
"""
An implementation of the U-Net paper:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    MICCAI (3) 2015: 234-241
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
from __future__ import print_function, division
from typing import ForwardRef
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from util.visualize import VisualizeFeatureMapPCA
import cv2
# import tensorflow as tf

from torch.nn.modules.activation import ReLU

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )
        self.se=SELayer(in_channels*2,reduction=int(in_channels/2))

    def forward(self, x):
        x_max=self.maxpool_conv(x)
        x_avg=self.avgpool_conv(x)
        x=torch.cat([x_max,x_avg],dim=1)

        x=self.se(x)
        return x


class bot_aspp(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, edge_dim):
        super(bot_aspp, self).__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(512, edge_dim,  kernel_size=3,padding=1, bias=False), 
            nn.BatchNorm2d(edge_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, aspp):
        x=self.aspp(aspp)
        return x


class RAF(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, ft_chns):
        super(RAF, self).__init__()

        self.upsample1 = BasicConv2d(ft_chns[4],ft_chns[3], 3, padding=1)
        self.upsample2 = BasicConv2d(ft_chns[2], ft_chns[3], 3, padding=1)
        self.upsample3 = BasicConv2d(ft_chns[3], ft_chns[4], 3, padding=1)

    def forward(self, x1,x2,x3,x4):
        size=64
        x3=F.interpolate(x3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_3=self.upsample1(x3)
        x1_1=x_3
        x2_1=x_3*x2
        # x3=F.interpolate(x3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x2=F.interpolate(x2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x1=F.interpolate(x1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x3_1=x_3*x2*self.upsample2(x1)
        x1_1=F.interpolate(x1_1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x2_2=torch.cat([x1_1,x2_1],dim=1)
        x2_2=F.interpolate(x2_2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x3_2=torch.cat([self.upsample3(x3_1),x2_2],dim=1)
        x4=F.interpolate(x4,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        aspp=torch.cat([x3_2,x4],dim=1)
        return aspp


class edge(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channel,edge_dim,end_dim):
        super(edge, self).__init__()
        self.bot_fine = nn.Conv2d(in_channel, end_dim, kernel_size=3,padding=1, bias=False)
        self.edge_fusion = nn.Conv2d(edge_dim + end_dim, edge_dim,1,bias=False)
        self.se=SELayer(edge_dim,reduction=16)
        self.upsample4 = BasicConv2d(64, 512, 3, padding=1)

    def forward(self, seg_edge,x1):
        _,_,w,h=seg_edge.shape
        dec0_fine = self.bot_fine(x1)
        dec0_fine=F.interpolate(dec0_fine,size=torch.Size([w, h]), mode='bilinear', align_corners=True)
        seg_edge = self.edge_fusion(torch.cat([seg_edge, dec0_fine], dim=1))
        seg_edge=self.se(seg_edge)
        seg_edge=self.upsample4(seg_edge)
        return seg_edge


class outfeature(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self):
        super( outfeature,self).__init__()

    def forward(self, aspp,seg_edge):
        seg_final=aspp+seg_edge
        return seg_final


class out(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self,n_class):
        super(out, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(512, n_class, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        )

    def forward(self, seg_final):
        output = F.interpolate(seg_final, size=[256,256], mode="bilinear", align_corners=True)

        output=self.out(output)
        return output





class SELayer(nn.Module):
    def __init__(self,channel,reduction=2):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
        self.fs=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.conv=nn.Conv2d(2,1,kernel_size=7,stride=1,padding=3)
    def forward(self,x):
        b,c,h,w=x.size()
        y_1=self.avg_pool(x).view(b,c)
        y_2=self.max_pool(x).view(b,c)
        y=y_1*y_2
        y=self.fc(y).view(b,c,1,1)
        y=x*y.expand_as(x)

        x_avg=torch.mean(x,dim=1,keepdim=True)
        x_max,_=torch.max(x,dim=1,keepdim=True)
        x_output=torch.cat([x_avg,x_max],dim=1)
        x_output=self.conv(x_output)
        x_output=self.fs(x_output)

        return x_output.expand_as(x)*y


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x




class PAFNet(nn.Module):
    def __init__(self, params):
        super(PAFNet, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        edge_dim=64
        end_dim=4
        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0]*2, self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0]*2, self.ft_chns[1], self.dropout[1])
        self.down2= DownBlock(self.ft_chns[1]*2, self.ft_chns[2], self.dropout[2])
        self.down3= DownBlock(self.ft_chns[2]*2, self.ft_chns[3], self.dropout[3])
        self.down4= DownBlock(self.ft_chns[3]*2, self.ft_chns[4], self.dropout[4])
        self.RAF=RAF(self.ft_chns)
        self.bot_aspp=bot_aspp(edge_dim)
        self.squeeze_body_edge = SqueezeBodyEdge(edge_dim, nn.BatchNorm2d)
        self.edge=edge(32,edge_dim,end_dim)
        self.outfeature=outfeature()
        self.out=out(self.n_class)
        self.number=0


    def forward(self, x):

        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        aspp1=self.RAF(x1,x2,x3,x4)
        aspp = self.bot_aspp(aspp1)
        seg_body, seg_edge = self.squeeze_body_edge(aspp,x1)
        aspp=aspp1
        seg_edge=self.edge(seg_edge,x1)
        output=self.outfeature(aspp,seg_edge)
        output=self.out(output)
        self.number+=1
        return output




class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = DownBlock(inplane, inplane, 0)
        self.out = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, padding=1,bias=False),
            norm_layer(inplane),
            # nn.ReLU(inplace=True),
        )
        self.xflow = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.flow_make = nn.Conv2d(inplane*3 , 2, kernel_size=3, padding=1, bias=False)
        self.se=SELayer(64,reduction=16)


    def forward(self, x,x2):
        size = x.size()[2:]
        seg_down = self.down(x)   #512,53,53
        seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        x2=F.interpolate(x2, size=size, mode="bilinear", align_corners=True)
        
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))

        seg_flow_warp = self.flow_warp(x, flow, size,x2)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge
        
    def flow_warp(self, input, flow, size,x):
        
        out_h, out_w = size
        n, c, h, w = input.size()
        input=self.se(input)
        x2=self.xflow(x)
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        input=input+x2

        output = F.grid_sample(input, grid,mode='bilinear',align_corners=True)
        return output


if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True}
    Net = PAFNet()
    Net = Net.double()

    x  = np.random.rand(2,3,256,256)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)

 