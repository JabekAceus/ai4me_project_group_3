import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv_block(in_dim, out_dim, **kwconv):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                         nn.BatchNorm2d(out_dim),
                         nn.PReLU())

class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, *, dropoutRate=0.01, dilation=1, asym: bool = False, use_se=False, use_attention=False):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)
        self.do = nn.Dropout2d(p=dropoutRate)
        
        if use_se:
            self.attention = SqueezeExcite(out_dim)
        elif use_attention:
            self.attention = CBAM(out_dim)
        else:
            self.attention = nn.Identity()

        self.PReLU_out = nn.PReLU()
        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, in_):
        main = self.conv_out(in_)
        other = self.do(self.block2(self.block1(self.block0(in_))))
        return self.PReLU_out(main + self.attention(other))

class BottleNeckDownSampling(BottleNeck):
     def __init__(self, in_dim, out_dim, projectionFactor, **kwargs):
        super().__init__(in_dim, out_dim, projectionFactor, **kwargs)
        mid_dim: int = in_dim // projectionFactor
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, stride=2, padding=0)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
     def forward(self, in_):
        main_downsampled, indices = self.maxpool0(in_)
        other = self.do(self.block2(self.block1(self.block0(in_))))
        
        _, c_other, _, _ = other.shape
        if c_other > main_downsampled.shape[1]:
            padding = torch.zeros(main_downsampled.shape[0], 
                                  c_other - main_downsampled.shape[1], 
                                  main_downsampled.shape[2],
                                  main_downsampled.shape[3], device=in_.device)
            main_downsampled = torch.cat([main_downsampled, padding], dim=1)
        
        output = self.PReLU_out(main_downsampled + self.attention(other))
        return output, indices
        


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, **kwargs):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.bottleneck = BottleNeck(in_dim, out_dim, projectionFactor, **kwargs)
    def forward(self, args) -> Tensor:
        in_, indices, skip = args
        unpooled = self.unpool(in_, indices, output_size=skip.size())
        concatenated = torch.cat((unpooled, skip), dim=1)
        return self.bottleneck(concatenated)

class ENet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_se=False, use_attention=False, improved_upsampling=False, **kwargs):
        super().__init__()
        K: int = kwargs.get("kernels", 16)
        
        init_conv_out_channels = K - in_dim
        self.conv0 = nn.Conv2d(in_dim, init_conv_out_channels, kernel_size=3, stride=2, padding=1) \
                     if init_conv_out_channels > 0 else nn.Identity()
        
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False)
        
        bn_kwargs = {"use_se": use_se, "use_attention": use_attention}
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, 4, **bn_kwargs)
        self.bottleneck1_1 = nn.Sequential(*[BottleNeck(K * 4, K * 4, 4, **bn_kwargs) for _ in range(4)])
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, 4, **bn_kwargs)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck(K * 8, K * 8, 4, dropoutRate=0.1, **bn_kwargs),
            BottleNeck(K * 8, K * 8, 4, dilation=2, **bn_kwargs),
        )
        self.bottleneck3 = BottleNeck(K * 8, K * 4, 4, dropoutRate=0.1, **bn_kwargs)
        self.bottleneck4_0 = BottleNeckUpSampling(K*4 + K*4, K*4, 4, **bn_kwargs)
        self.bottleneck4_1 = BottleNeck(K*4, K, 4, **bn_kwargs)
        self.bottleneck5_0 = BottleNeckUpSampling(K + K, K, 4, **bn_kwargs)
        self.bottleneck5_1 = BottleNeck(K, K, 4, **bn_kwargs)
        
        if improved_upsampling:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(K, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.PReLU()
            )
            print("> Initialized ENet with IMPROVED upsampling path.")
        else:
            self.final = nn.ConvTranspose2d(K, out_dim, kernel_size=2, stride=2)
        print(f"> Initialized ENet ({in_dim=}->{out_dim=}) with {use_se=}, {use_attention=}")



    def forward(self, input):
        maxpool_0 = self.maxpool0(input)
        
        if isinstance(self.conv0, nn.Conv2d):
            conv_0 = self.conv0(input)
            outputInitial = torch.cat((conv_0, maxpool_0), dim=1)
        else: 
            outputInitial = maxpool_0
        
        bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)
        bn3_out = self.bottleneck3(bn2_out)
        bn4_0_out = self.bottleneck4_0((bn3_out, indices_2, bn1_out))
        bn4_out = self.bottleneck4_1(bn4_0_out)
        bn5_0_out = self.bottleneck5_0((bn4_out, indices_1, outputInitial))
        bn5_out = self.bottleneck5_1(bn5_0_out)
        return self.final(bn5_out)
        
    def init_weights(self):
        self.apply(random_weights_init)


class ENet2_5D(ENet):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        K: int = kwargs.get("kernels", 16)
        # Pass use_attention flag to parent
        super().__init__(in_dim=K, out_dim=out_dim, use_se=True, 
                         use_attention=kwargs.get("use_attention", False), **kwargs)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(1, K, kernel_size=(in_dim, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(K),
            nn.PReLU()
        )
        
        self.final = nn.Conv2d(K, out_dim, kernel_size=1)
        
        print(f"> Initialized ENet2.5D ({in_dim=}->{out_dim=})")
    
    def forward(self, input):
        x = input.unsqueeze(1)
        fused_features_3d = self.fusion_conv(x)
        fused_features_2d = fused_features_3d.squeeze(2)
        bn1_0, indices_1 = self.bottleneck1_0(fused_features_2d)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)
        bn3_out = self.bottleneck3(bn2_out)
        bn4_0_out = self.bottleneck4_0((bn3_out, indices_2, bn1_out))
        bn4_out = self.bottleneck4_1(bn4_0_out)
        bn5_0_out = self.bottleneck5_0((bn4_out, indices_1, fused_features_2d))
        bn5_out = self.bottleneck5_1(bn5_0_out)
        return self.final(bn5_out)