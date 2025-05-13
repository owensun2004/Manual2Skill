import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_backbone(nn.Module):
    def __init__(self, num_classes, input_channel=1, input_size=224, width_mult=1.):
        super(MobileNetV2_backbone, self).__init__()
        block = InvertedResidual
        last_channel = 512
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 256, 3, 2],  # 32, 32, 96 -> 16, 16, 160     14
            [6, 512, 1, 1],  # 16, 16, 160 -> 16, 16, 320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 512, 512, 3 -> 256, 256, 32
        self.features = [self.conv_bn(input_channel, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(self.conv_1x1_bn(
            input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False),
        )

    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False),
        )


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, input_channel=1, downsample_factor=8):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = MobileNetV2_backbone(num_classes=num_classes,input_channel=input_channel)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0,
                      dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 *
                      rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 *
                      rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 *
                      rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_gn = nn.GroupNorm(num_groups=4, num_channels=dim_out)
        self.branch5_relu = nn.ReLU(inplace=False)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        if b > 1 or (not self.training):
            global_feature = self.branch5_bn(global_feature)   
        else:
            global_feature = self.branch5_gn(global_feature)  
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(
            global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

def tensor_mean(x):
    return x.mean(dim=(2, 3))

def tensor_max(x):
    x_max1 = torch.max(x, dim=3).values 
    x_max2 = torch.max(x_max1, dim=2).values  
    return x_max2


class DeepLab_V3_plus(nn.Module):
    def __init__(self, num_classes, input_channel=1, downsample_factor=16, out_dim=256):
        super(DeepLab_V3_plus, self).__init__()
        self.backbone = MobileNetV2(num_classes = num_classes, input_channel=input_channel, downsample_factor=downsample_factor)
        in_channels = 512

        self.aspp = ASPP(dim_in=in_channels, dim_out=out_dim,
                         rate=16//downsample_factor)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x1 = self.backbone(x)
        x2 = self.aspp(x1)

        x_mean = tensor_mean(x2)
        x_max = tensor_max(x2)
        result = torch.cat([x_mean, x_max], dim=1)
        return result
    
if __name__ == "__main__":
    t = torch.rand(4, 3, 1920, 1080)
    net = DeepLab_V3_plus(num_classes = 1, input_channel=3)
    _ = net(t)
