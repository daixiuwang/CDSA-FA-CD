from .functions import *
from .resnet import ResNet
import models

class ResNet(torch.nn.Module):
    def __init__(self, input_nc,
                 resnet_stages_num=4, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True, replace_stride_with_dilation=[False,False,False])
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        #self.resnet_stages_num=4
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.forward_single(x)

        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        if self.resnet_stages_num > 3:
            x_16 = self.resnet.layer3(x_8) # 1/16, in=128, out=256
        if self.resnet_stages_num == 4:
            x_32 = self.resnet.layer4(x_16) # 1/32, in=256, out=512
        return  (x_4,x_8, x_16, x_32)


class Conv_ex(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(Conv_ex,self).__init__()
        self.conv3 = DoubleConv(in_dim, out_dim)
        self.se = SEModule(out_dim)
        self.dlt = Dlt(out_dim)
        self.c = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        self.ex = Exchange(out_dim)

    def forward(self,x1, x2, y=None):
        z = self.ex(x1, x2, y)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.se(x)
        x = self.dlt(x)
        x = self.c(x)

        return x * z




class CAFB_PA(nn.Module):
    def __init__(self, in_ch = 3):
        super(CAFB_PA, self).__init__()
        self.backbone = ResNet(input_nc=in_ch)
        self.inc = (DoubleConv(in_ch,64))
        self.down0 = (Down(64, 128))

        self.down1 = (Down(128,256))
        self.down2 = (Down(256,512))

        self.cafb3 = Conv_ex(1024,512)
        self.cafb2 = Conv_ex(512, 256)
        self.cafb1 = Conv_ex(256, 128)
        self.cafb0 = Conv_ex(128, 64)

        self.pa1 = PA(512, 256)
        self.pa2 = PA(256, 128)
        self.pa3 = PA(128, 64)
        self.outc2 = OutConv(64, 2)

    def forward(self, x1, x2):
        # #sat
        x1_0 = self.inc(x1)     #256   3-64
        x1_1 = self.down0(x1_0)  #128 64-128
        x1_2 = self.down1(x1_1)  #64 128-256
        x1_3 = self.down2(x1_2) #32 256-512

        # #uav
        x2_0, x2_1, x2_2, x2_3 = self.backbone(x2)

        x_3 = self.cafb3(x1_3, x2_3)    #1024-512
        x_2 = self.cafb2(x1_2,x2_2, x_3)  #512-256
        x_1 = self.cafb1(x1_1,x2_1, x_2)  #256-128
        x_0 = self.cafb0(x1_0,x2_0, x_1)  #128-64

        x = self.pa1(x_3, x_2)         #64 512-256
        x = self.pa2(x, x_1)           #128  256-128
        x = self.pa3(x, x_0)           #256 128-64

        logits = self.outc2(x)
        return logits


