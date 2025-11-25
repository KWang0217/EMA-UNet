import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # ----------xy---------- #
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx---------- #
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy---------- #
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw---------- #
        x4 = self.dw(x4)
        # ----------concat---------- #
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw---------- #
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class Spatial_Group_Enhance_Module(nn.Module):
    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class Channel_Attention_Module(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.shared_conv = nn.Conv2d(dim_in, dim_in, kernel_size=1, bias=False)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim_in, eps=1e-6, elementwise_affine=True)

    def forward(self, x):
        b, c, h, w = x.size()
        feature = self.shared_conv(x)

        proj_query = feature.view(b, c, -1)
        proj_key = feature.view(b, c, -1).permute(0, 2, 1)
        proj_value = feature.view(b, c, -1)

        energy = torch.bmm(proj_query, proj_key)
        energy = self.layer_norm(energy)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)

        out = self.gamma * out + feature
        return out


class Spatial_Channel_Enhancement_Module(nn.Module):
    def __init__(self, dim_in, groups=4):
        super().__init__()
        self.sge = Spatial_Group_Enhance_Module(groups)
        self.cam = Channel_Attention_Module(dim_in)

    def forward(self, x):
        x = self.sge(x)
        x = self.cam(x)
        return x


class Dynamic_Feature_Fusion_Module(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_norm = nn.GroupNorm(dim_in, dim_in, eps=1e-6, affine=True)

        self.conv_att = nn.Sequential(
            nn.Conv2d(dim_in * 2 + 1, dim_in * 2 + 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_rec = nn.Conv2d(dim_in * 2 + 1, dim_in, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim_in, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim_in, 1, kernel_size=1, stride=1, bias=True)
 
    def forward(self, en_x, de_x, mask):
        output = torch.cat([en_x, de_x, mask], dim=1)

        att = self.conv_att(self.avg_pool(output))
        output = output * att
        output = self.group_norm(self.conv_rec(output))    

        att = self.conv1(en_x) + self.conv2(de_x) + mask      
        att = self.sigmoid(att / 3.0)   

        output = output * att
        return output


class LEMAUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True,
                 deep_supervision=True):
        super().__init__()

        self.bridge = bridge
        self.deep_supervision = deep_supervision

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[2]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[3]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[4]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if self.bridge:
            self.dff_1 = Dynamic_Feature_Fusion_Module(c_list[0])
            self.dff_2 = Dynamic_Feature_Fusion_Module(c_list[1])
            self.dff_3 = Dynamic_Feature_Fusion_Module(c_list[2])
            self.dff_4 = Dynamic_Feature_Fusion_Module(c_list[3])
            self.dff_5 = Dynamic_Feature_Fusion_Module(c_list[4])
            print('DFF was used')
        if self.deep_supervision:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('Deep Supervision was used')

        self.decoder1 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[5]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[4]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            Spatial_Channel_Enhancement_Module(c_list[3]),
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        "----------------------------------------------------------------------------------------------------stage5"
        gt_pre5 = self.gt_conv5(out5)
        t5 = self.dff_5(t5, out5, gt_pre5)
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        "----------------------------------------------------------------------------------------------------"

        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        "----------------------------------------------------------------------------------------------------stage4"
        gt_pre4 = self.gt_conv4(out4)
        t4 = self.dff_4(t4, out4, gt_pre4)
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        "----------------------------------------------------------------------------------------------------"

        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        "----------------------------------------------------------------------------------------------------stage3"
        gt_pre3 = self.gt_conv3(out3)
        t3 = self.dff_3(t3, out3, gt_pre3)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        "----------------------------------------------------------------------------------------------------"

        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        "----------------------------------------------------------------------------------------------------stage2"
        gt_pre2 = self.gt_conv2(out2)
        t2 = self.dff_2(t2, out2, gt_pre2)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        "----------------------------------------------------------------------------------------------------"

        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        "----------------------------------------------------------------------------------------------------stage1"
        gt_pre1 = self.gt_conv1(out1)
        t1 = self.dff_1(t1, out1, gt_pre1)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        "----------------------------------------------------------------------------------------------------"

        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), out0


from thop import profile

if __name__ == '__main__':
    model = LDFFUNet(num_classes=1,
                     input_channels=3,
                     c_list=[8, 16, 24, 32, 48, 64],
                     bridge=True,
                     deep_supervision=True).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")
    print(f"Parameters: {params / 1e6} M")
