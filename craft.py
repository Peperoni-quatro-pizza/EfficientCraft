import torch 
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.cuda.amp import autocast 

from torchvision import models

from basenet.model import EfficientNet

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class double_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class CRAFT(nn.Module): 
    def __init__(self):
        super(CRAFT, self).__init__()

        """Base Net""" 
        self.basenet = EfficientNet.from_pretrained('efficientnet-b0')

        """U net"""
        self.upconv1 = double_conv(320+192,256)
        self.upconv2 = double_conv(112+256,128)
        self.upconv3 = double_conv(40+128,64)
        self.upconv4 = double_conv(24+64,32)
        self.upconv5 = double_conv(16+32,16)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.upconv5.modules())
        init_weights(self.conv_cls.modules())

    @autocast()
    def forward(self,x):

        """Base Net""" 
        endpoints = self.basenet.extract_endpoints_2(x)
        
        p1 = endpoints['reduction_1']
        p2 = endpoints['reduction_2']
        p3 = endpoints['reduction_3']
        p4 = endpoints['reduction_5']
        p5 = endpoints['reduction_6']
        p5_1 = endpoints['reduction_7']

        """U Net"""
        
        y = torch.cat([p5,p5_1] , dim=1)
        y = self.upconv1(y)


        y = F.interpolate(y, size=p4.size()[2:] , mode='bilinear' , align_corners=False)
        y = torch.cat([y , p4] , dim = 1)
        y = self.upconv2(y)


        y = F.interpolate(y , size = p3.size()[2:] , mode = 'bilinear' , align_corners=False)
        y = torch.cat([y,p3] , dim = 1)
        y = self.upconv3(y)


        y = F.interpolate(y , size=p2.size()[2:] , mode='bilinear' , align_corners=False)
        y = torch.cat([y , p2] , dim = 1)
        y = self.upconv4(y)


        y = F.interpolate(y , size=p1.size()[2:] , mode='bilinear' , align_corners=False)
        y = torch.cat([y,p1] , dim = 1)
        feature = self.upconv5(y)

        y = self.conv_cls(feature)
        return  y.permute(0,2,3,1) , feature 

if __name__ == '__main__': 
    model  = CRAFT().cuda()
    output, _  = model(torch.rand(1,3,768,768).cuda())





        

