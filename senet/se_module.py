from torch import nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*16, channel*16 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*16 // reduction, channel*16, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(4)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)
        # buff_error = buff_x - x
        # buff_error = buff_x - x
        return x

class MALayer_msfa(nn.Module):
    def __init__(self, msfa_size, channel, reduction=16):
        super(MALayer_msfa, self).__init__()
        self.shuffledown = Shuffle_d(msfa_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*msfa_size**2, channel*msfa_size**2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*msfa_size**2 // reduction, channel*msfa_size**2, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(msfa_size)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)
        return x

# class MALayer_Conv_msfa(nn.Module):
#     #This one is Conv-type
#     def __init__(self, msfa_size, channel, reduction=1):
#         super(MALayer_Conv_msfa, self).__init__()
#         self.shuffledown = Shuffle_d(msfa_size)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.shuffleup = nn.PixelShuffle(msfa_size)
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=channel, out_channels=int(channel // reduction), kernel_size=msfa_size, stride=1, padding=0, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=int(channel // reduction), out_channels=channel*msfa_size**2, kernel_size=1, stride=1, padding=0,
#                       bias=False),
#             nn.Sigmoid()
#         )
#         self.shuffleup1 = nn.PixelShuffle(msfa_size)
#
#     def forward(self, x):
#         ex_x = self.shuffledown(x)
#         b, c, _, _ = ex_x.size()
#         y = self.avg_pool(ex_x)
#         y = self.shuffleup(y)
#         y = self.convs(y)
#         ex_x = ex_x * y.expand_as(ex_x)
#         x = self.shuffleup1(ex_x)
#         return x

class MALayer_Conv_msfa(nn.Module):
    #This one is Conv-type
    def __init__(self, msfa_size, channel, reduction=1):
        super(MALayer_Conv_msfa, self).__init__()
        self.shuffledown = Shuffle_d(msfa_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shuffleup = nn.PixelShuffle(msfa_size)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=int(channel*msfa_size**2 // reduction), kernel_size=msfa_size, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(channel // reduction), out_channels=channel*msfa_size**2, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid()
        )
        self.shuffleup1 = nn.PixelShuffle(msfa_size)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x)
        y = self.shuffleup(y)
        y = self.convs(y)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup1(ex_x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.mean(x,1)
        return torch.mean(x, 1).unsqueeze(1)


### This CA_AA_par_Layer1 use the new AALayer1
class CA_AA_par_Layer1(nn.Module):
    def __init__(self, msfa_size, channel, reduction=16):
        super(CA_AA_par_Layer1, self).__init__()
        self.compress = ChannelPool()
        self.shuffledown = Shuffle_d(msfa_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(msfa_size**2, msfa_size**2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(msfa_size**2 // reduction, msfa_size**2, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(msfa_size)

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        buff_x = x
        N, C, H, W = x.size()
        x = x.view(N * C, 1, H, W)  # N, C, H, W to N*C, 1, H, W
        sq_x = self.shuffledown(x)  # N*C, 1, H, W to N*C, 16, H/4, W/4
        b, c, _, _ = sq_x.size()
        y = self.avg_pool(sq_x).view(b, c)  # N*C, 16, H/4, W/4 to N*C, 16, 1, 1 to N*C, 16
        y = self.fc(y).view(b, c, 1, 1)  # N*C, 16, 1, 1
        y = y.expand_as(sq_x)  # N*C, 16, 1, 1 to N*C, 16, H/4, W/4
        ex_y = self.shuffleup(y)  # N*C, 16, H/4, W/4 to N*C, 1, H, W
        out = x * ex_y
        out = out.view(N, C, H, W)

        b, c, _, _ = buff_x.size()
        y = self.avg_pool1(buff_x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        return out


class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)