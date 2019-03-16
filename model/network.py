
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.M_Net import M_net
from model.T_Net_psp import PSPNet

class net_T(nn.Module):
    # Train T_net
    def __init__(self):

        super(net_T, self).__init__()

        self.t_net = PSPNet()

    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        return trimap

class net_M(nn.Module):
    '''
		train M_net
    '''

    def __init__(self):

        super(net_M, self).__init__()
        self.m_net = M_net()

    def forward(self, input, trimap):

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return alpha_p

class net_F(nn.Module):
    '''
		end to end net 
    '''

    def __init__(self):

        super(net_F, self).__init__()

        self.t_net = PSPNet()
        self.m_net = M_net()



    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        bg, unsure, fg = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
