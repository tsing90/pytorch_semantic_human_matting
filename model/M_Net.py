
import torch
import torch.nn as nn


class M_net(nn.Module):
    '''
        encoder + decoder
    '''

    def __init__(self, classes=2):

        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        # stage-1
        self.conv_1_1 = nn.Sequential(nn.Conv2d(6, 64, 3, 1, 1, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-2
        self.conv_2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        self.conv_2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-3
        self.conv_3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.conv_3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.conv_3_3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-4
        self.conv_4_1 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.conv_4_2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.conv_4_3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-5
        self.conv_5_1 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.conv_5_2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.conv_5_3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())

        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # stage-5
        self.deconv_5 = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        
        # stage-4
        self.up_pool_4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_4 = nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        
        # stage-3
        self.up_pool_3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_3 = nn.Sequential(nn.Conv2d(256, 128, 5, 1, 2, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        
        # stage-2
        self.up_pool_2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_2 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        
        # stage-1
        self.up_pool_1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_1 = nn.Sequential(nn.Conv2d(64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        
        # stage-0
        self.conv_0 = nn.Conv2d(64, 1, 5, 1, 2, bias=True)


    def forward(self, input):

        # ----------------
        # encoder
        # --------
        x11 = self.conv_1_1(input)
        x12 = self.conv_1_2(x11)
        x1p, id1 = self.max_pooling_1(x12)

        x21 = self.conv_2_1(x1p)
        x22 = self.conv_2_2(x21)
        x2p, id2 = self.max_pooling_2(x22)

        x31 = self.conv_3_1(x2p)
        x32 = self.conv_3_2(x31)
        x33 = self.conv_3_3(x32)
        x3p, id3 = self.max_pooling_3(x33)

        x41 = self.conv_4_1(x3p)
        x42 = self.conv_4_2(x41)
        x43 = self.conv_4_3(x42)
        x4p, id4 = self.max_pooling_4(x43)
        
        x51 = self.conv_5_1(x4p)
        x52 = self.conv_5_2(x51)
        x53 = self.conv_5_3(x52)
        # ----------------
        # decoder
        # --------
        x5d = self.deconv_5(x53)
        
        x4u = self.up_pool_4(x5d, id4)
        x4d = self.deconv_4(x4u)

        x3u = self.up_pool_3(x4d, id3)
        x3d = self.deconv_3(x3u)

        x2u = self.up_pool_2(x3d, id2)
        x2d = self.deconv_2(x2u)
        
        x1u = self.up_pool_1(x2d, id1)
        x1d = self.deconv_1(x1u)

        # raw alpha pred
        raw_alpha = self.conv_0(x1d)

        return raw_alpha





