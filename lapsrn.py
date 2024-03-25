import torch.nn as nn
import numpy as np
import math
from senet.se_module import SELayer, MALayer_Conv_msfa, CA_AA_par_Layer1, MALayer_msfa
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple
from thop import profile
from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(16)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter0 = np.reshape(BilinearFilter, (7, 7))
    return torch.from_numpy(filter0).float()

def get_WB_filter_msfa(msfa_size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    size = 2*msfa_size-1
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(msfa_size**2)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (msfa_size**2))
    filter0 = np.reshape(BilinearFilter, (size, size))
    return torch.from_numpy(filter0).float()

def RGB_interpolation_tensor(bayer, cfa='RGGB'):
    bayer_numpy = bayer.cpu().detach().numpy()
    N ,C ,H, W = bayer_numpy.shape
    for i in range(N):
        if i ==0:
            rgb_tensor = demosaicing_CFA_Bayer_Menon2007(bayer_numpy[i, 0,:, :], cfa)
            rgb_tensor = np.expand_dims(rgb_tensor, axis=0)
        else:
            rgb = demosaicing_CFA_Bayer_Menon2007(bayer_numpy[i, 0, :, :], cfa)
            rgb = np.expand_dims(rgb, axis=0)
            rgb_tensor = np.concatenate([rgb_tensor, rgb], axis=0)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_tensor)).permute(0, 3, 1, 2).float().to(bayer.device)

    return rgb_tensor

class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output += residual
        output = self.relu(output)
        return output

class _2Conv_Block(nn.Module):
    def __init__(self):
        super(_2Conv_Block, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output += residual
        output = self.relu(output)
        return output

class _Conv_attention_Block(nn.Module):
    def __init__(self):
        super(_Conv_attention_Block, self).__init__()
        self.ma = MALayer(64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class _Conv_attention_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_Conv_attention_Block_msfasize, self).__init__()
        self.ma = MALayer_Conv_msfa(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class _Conv_HSA_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_Conv_HSA_Block_msfasize, self).__init__()
        self.ma = MALayer_msfa(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class _MosaicConv_HSA_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_MosaicConv_HSA_Block_msfasize, self).__init__()
        self.ma = MALayer_msfa(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            ConvMosaic_new2(in_channels=64, out_channels=64, msfa_size=msfa_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ConvMosaic_new2(in_channels=64, out_channels=64, msfa_size=msfa_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ConvMosaic_new2(in_channels=64, out_channels=64, msfa_size=msfa_size, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class _Conv_LSA_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_Conv_LSA_Block_msfasize, self).__init__()
        self.ma = CA_AA_par_Layer1(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class _Conv_SE_Block(nn.Module):
    def __init__(self):
        super(_Conv_SE_Block, self).__init__()
        self.ma = SELayer(64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class branch_block_front(nn.Module):
    def __init__(self):
        super(branch_block_front, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=False)
        # self.se = MALayer(16, 4)
        # self.se = SELayer(16, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # x = self.se(x)
        x = self.relu(x)
        # x = self.front_conv_input(x)
        return x

class branch_block_front_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(branch_block_front_msfasize, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.se = MALayer_Conv_msfa(msfa_size, msfa_size**2, 4)
        # self.se = SELayer(16, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.se(x)
        x = self.relu(x)
        # x = self.front_conv_input(x)
        return x

class branch_block_front_LSA_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(branch_block_front_LSA_msfasize, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.se = CA_AA_par_Layer1(msfa_size, msfa_size**2, 4)
        # self.se = SELayer(16, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.se(x)
        x = self.relu(x)
        # x = self.front_conv_input(x)
        return x

class branch_block_back(nn.Module):
    def __init__(self):
        super(branch_block_back, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.se = SELayer(64, 16)
        self.cov_block = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        # x = self.relu(x)
        output = self.cov_block(x)
        return output

class branch_block_back_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(branch_block_back_msfasize, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.se = SELayer(64, 16)
        self.cov_block = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        # x = self.relu(x)
        output = self.cov_block(x)
        return output

class front_input_block_msfa_size(nn.Module):
    def __init__(self, msfa_size):
        super(front_input_block_msfa_size, self).__init__()
        self.st_conv = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        # self.aa = AALayer1(4)
        # self.ca = SELayer(16, 4)
        # self.fir_att = CA_AA_par_Layer1(4, 2, 1)
        self.par_att = CA_AA_par_Layer1(msfa_size, msfa_size**2, 4)
        self.par_att1 = CA_AA_par_Layer1(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x = self.fir_att(x)
        x = self.st_conv(x)
        x = self.par_att(x)
        x = self.relu(x)
        x = self.front_conv_input(x)
        x = self.par_att1(x)
        x = self.relu1(x)
        return x

class front_input_block_msfa_size_noAtt(nn.Module):
    def __init__(self, msfa_size):
        super(front_input_block_msfa_size_noAtt, self).__init__()
        self.st_conv = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.st_conv(x)
        x = self.relu(x)
        x = self.front_conv_input(x)
        x = self.relu1(x)
        return x


class Pos2Weight(nn.Module):
    def __init__(self, outC=16, kernel_size=5, inC=1):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output

class _ConvMultiW(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvMultiW, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.msfa_size = msfa_size
        self.weight = Parameter(torch.Tensor(msfa_size[0]*msfa_size[1], in_channels*kernel_size[0]*kernel_size[1], out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(msfa_size[0]*msfa_size[1], out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self):
    #     s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
    #          ', stride={stride}')
    #     if self.padding != (0,) * len(self.padding):
    #         s += ', padding={padding}'
    #     if self.dilation != (1,) * len(self.dilation):
    #         s += ', dilation={dilation}'
    #     if self.output_padding != (0,) * len(self.output_padding):
    #         s += ', output_padding={output_padding}'
    #     if self.groups != 1:
    #         s += ', groups={groups}'
    #     if self.bias is None:
    #         s += ', bias=False'
    #     return s.format(**self.__dict__)

class ConvMosaic(_ConvMultiW):

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        msfa_size = _pair(msfa_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConvMosaic, self).__init__(
            in_channels, out_channels, kernel_size, msfa_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        N, inC, H, W = input.size()
        cols = nn.functional.unfold(input, self.kernel_size, padding=self.padding)
        scale_int = 1
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        spe_num, kernel_size_and_inC, outC = self.weight.size()
        self.weight1 = self.weight.view(1, spe_num, kernel_size_and_inC, outC)
        self.weight1 = torch.cat([self.weight1] * ((H*W)//spe_num), 0)
        self.weight2 = self.weight1.view(1, H*W, kernel_size_and_inC, outC)
        buff_weight = self.weight2.cpu().detach().numpy()
        # buff_weight_1 = buff_weight[0, 0, :, :]
        # buff_weight_2 = buff_weight[0, 1, :, :]
        # buff_weight_3 = buff_weight[0, 7, :, :]
        # buff_weight_4 = buff_weight[0, 8, :, :]
        # buff_weight_5 = buff_weight[0, 9, :, :]
        # buff_weight_1 = buff_weight[0, 0, :, :]
        # buff_weight_2 = buff_weight[0, 1, :, :]
        # buff_weight_3 = buff_weight[0, 15, :, :]
        # buff_weight_4 = buff_weight[0, 16, :, :]
        # buff_weight_5 = buff_weight[0, 17, :, :]
        Raw_conv = torch.matmul(cols, self.weight2).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(N, scale_int, scale_int, self.out_channels, H, W).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(N, self.out_channels, scale_int * H, scale_int * W)
        self.bias1 = self.bias.view(1, spe_num, outC)
        self.bias1 = torch.cat([self.bias1] * ((H*W)//spe_num), 0)
        self.bias2 = self.bias1.view(1, H*W, outC)
        return Raw_conv

class ConvMosaic_new(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride=1,
                 padding=0, bias=True):
        super(ConvMosaic_new, self).__init__()
        self.msfa_size = msfa_size
        self.outC = out_channels
        self.mcm_ksize = kernel_size
        self.MosaicConv = nn.Conv2d(in_channels=msfa_size**2, out_channels=self.outC*msfa_size**2, kernel_size=self.mcm_ksize, stride=msfa_size, padding=0, bias=bias, groups=msfa_size**2)
        self.shuffleup = nn.PixelShuffle(int(msfa_size))

    def forward(self, input_raw):
        pad_size = (self.mcm_ksize - 1) // 2
        input_raw = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(input_raw)
        raw_cube = input_raw.repeat(1, self.msfa_size ** 2, 1, 1)

        for i in range(self.msfa_size):
            for j in range(self.msfa_size):
                if i == 0 and j == 0:
                    continue
                else:
                    raw_cube[:, i * self.msfa_size + j, :, :] = torch.roll(input_raw, (-i, -j), dims=(2, 3))

        mosaicfm = self.MosaicConv(raw_cube)
        for c in range(0, self.outC, 1):
            if c == 0:
                order = list(range(c, self.outC * self.msfa_size ** 2 + c, self.outC))
            else:
                order = order + list(range(c, self.outC * self.msfa_size ** 2 + c, self.outC))
        mosaicfm = mosaicfm[:, order, :, :]
        mosaicfm = self.shuffleup(mosaicfm)
        return mosaicfm

class ConvMosaic_new1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride=1,
                 padding=0, bias=True):
        super(ConvMosaic_new1, self).__init__()
        self.msfa_size = msfa_size
        self.inC = in_channels
        self.outC = out_channels
        self.mcm_ksize = kernel_size
        self.MosaicConv = nn.Conv2d(in_channels=self.inC * msfa_size ** 2, out_channels=self.outC * msfa_size ** 2,
                                    kernel_size=self.mcm_ksize, stride=msfa_size, padding=0, bias=bias,
                                    groups=msfa_size ** 2)
        self.shuffleup = nn.PixelShuffle(int(msfa_size))

    def forward(self, input_raw):
        pad_size = (self.mcm_ksize - 1) // 2
        input_raw = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(input_raw)
        raw_cube = input_raw.repeat(1, self.msfa_size ** 2, 1, 1)

        for i in range(self.msfa_size):
            for j in range(self.msfa_size):
                if i == 0 and j == 0:
                    continue
                else:
                    n = i * self.msfa_size + j
                    raw_cube[:, n*self.inC:(n+1)*self.inC, :, :] = torch.roll(input_raw, (-i, -j), dims=(2, 3))

        mosaicfm = self.MosaicConv(raw_cube)
        for c in range(0, self.outC, 1):
            if c == 0:
                order = list(range(c, self.outC * self.msfa_size ** 2 + c, self.outC))
            else:
                order = order + list(range(c, self.outC * self.msfa_size ** 2 + c, self.outC))
        mosaicfm = mosaicfm[:, order, :, :]
        mosaicfm = self.shuffleup(mosaicfm)
        return mosaicfm

class ConvMosaic_new2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, msfa_size, stride=1,
                 padding=0, bias=True):
        super(ConvMosaic_new2, self).__init__()
        self.msfa_size = msfa_size
        self.inC = in_channels
        self.outC = out_channels
        self.mcm_ksize = kernel_size
        self.kernel_period = int(msfa_size / stride)
        self.stride = stride
        self.padding = padding

        self.MosaicConv = nn.Conv2d(in_channels=self.inC * self.kernel_period ** 2, out_channels=self.outC * self.kernel_period ** 2,
                                    kernel_size=self.mcm_ksize, stride=msfa_size, padding=0, bias=bias,
                                    groups=self.kernel_period ** 2)
        self.shuffleup = nn.PixelShuffle(self.kernel_period)

    def forward(self, input_raw):
        input_raw = nn.ZeroPad2d((self.padding, self.padding, self.padding, self.padding))(input_raw)
        raw_cube = input_raw.repeat(1, self.kernel_period ** 2, 1, 1)
        for i in range(self.kernel_period):
            for j in range(self.kernel_period):
                if i == 0 and j == 0:
                    continue
                else:
                    n = i * self.kernel_period + j
                    raw_cube[:, n*self.inC:(n+1)*self.inC, :, :] = torch.roll(input_raw, (-i*self.stride, -j*self.stride), dims=(2, 3))

        mosaicfm = self.MosaicConv(raw_cube)
        for c in range(0, self.outC, 1):
            if c == 0:
                order = list(range(c, self.outC * self.kernel_period ** 2, self.outC))
            else:
                order = order + list(range(c, self.outC * self.kernel_period ** 2, self.outC))
        mosaicfm = mosaicfm[:, order, :, :]
        mosaicfm = self.shuffleup(mosaicfm)
        return mosaicfm

class Net_buff(nn.Module):
    def __init__(self, msfa_size):
        super(Net, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)

        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net(nn.Module):
    def __init__(self, msfa_size):
        super(Net, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)

        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        msfa_size = 5
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:msfa_size, 0:msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(msfa_size, msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / msfa_size), 1)
        local_weight1 = local_weight1.view(h_pattern_n * msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * msfa_size * W, -1, self.outC)
        for i in range(0, int(H / msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * msfa_size * h_pattern_n * W:(i + 1) * msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        convt_br1_front = self.convt_br1_front(Raw_conv_buff)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St_BUFF(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        # self.convt_br1_front = self.make_ma_layer(branch_block_front, msfa_size)
        self.convt_F1 = self.make_layer(_Conv_Block)
        # self.convt_F1 = self.make_ma_layer(_Conv_Block, msfa_size)
        self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        # self.convt_br1_front = self.make_ma_layer(branch_block_front, msfa_size)
        self.convt_F1 = self.make_layer(_Conv_Block)
        # self.convt_F1 = self.make_ma_layer(_Conv_Block, msfa_size)
        self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        msfa_size = 5
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:msfa_size, 0:msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(msfa_size, msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / msfa_size), 1)
        local_weight1 = local_weight1.view(h_pattern_n * msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * msfa_size * W, -1, self.outC)
        for i in range(0, int(H / msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * msfa_size * h_pattern_n * W:(i + 1) * msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        convt_br1_front = self.convt_br1_front(Raw_conv_buff)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_MC_LSA_BUFF(nn.Module):
    def __init__(self, msfa_size):
        super(Net_MC_LSA, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_LSA_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_MC_LSA(nn.Module):
    def __init__(self, msfa_size):
        super(Net_MC_LSA, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_LSA_msfasize, msfa_size)
        self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)

        N, C, H, W = y.size()
        msfa_size = 5
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:msfa_size, 0:msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, msfa_size**2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(msfa_size, msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        # local_weight = local_weight.repeat(int(H / msfa_size), int(W / msfa_size), 1)
        # local_weight = local_weight.view(H * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        # local_weight = local_weight.contiguous().view(1, H*W, -1, self.outC)
        # Raw_conv = torch.matmul(cols, local_weight)
        # Raw_conv = Raw_conv.permute(0, 1, 4, 2, 3)
        # Raw_conv = Raw_conv.contiguous().view(N, 1, 1, self.outC, H, W)
        # Raw_conv = Raw_conv.contiguous().view(N, self.outC, H, W)

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / msfa_size), 1)
        local_weight1 = local_weight1.view(h_pattern_n * msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * msfa_size * W, -1, self.outC)
        # Raw_conv_buff = torch.zeros(1, 1, msfa_size * h_pattern_n * W, 1, self.outC)
        for i in range(0, int(H / msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * msfa_size * h_pattern_n * W:(i + 1) * msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        convt_br1_front = self.convt_br1_front(Raw_conv_buff)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_MC_StRes(nn.Module):
    def __init__(self, msfa_size):
        super(Net_MC_StRes, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        # self.convt_br1_front = self.make_ma_layer(branch_block_front, msfa_size)
        self.convt_F1 = self.make_layer(_Conv_Block)
        # self.convt_F1 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_MC_LSARes1(nn.Module):
    def __init__(self, msfa_size):
        super(Net_MC_LSARes1, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_LSA_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_MC_HSARes1(nn.Module):
    def __init__(self, msfa_size):
        super(Net_MC_HSARes1, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St_StRes1(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St_StRes1, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.firstconv = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.relu = nn.LeakyReLU(0.2)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        # self.convt_br1_front = self.make_ma_layer(branch_block_front, msfa_size)
        self.convt_F1 = self.make_layer(_Conv_Block)
        # self.convt_F1 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        # local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        # up_y = self.repeat_y(y)
        # cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        # scale_int = math.ceil(self.scale)
        # cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
        #                               1).permute(0, 1, 3, 4, 2).contiguous()
        # local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
        #                                               self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        # local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        # Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
        #     0, 3, 4, 1, 5, 2)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        Raw_conv1 = self.firstconv(y)
        Raw_conv = self.relu(Raw_conv1)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St_LSARes1(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St_LSARes1, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.firstconv = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_LSA_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        # local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        # up_y = self.repeat_y(y)
        # cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        # scale_int = math.ceil(self.scale)
        # cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
        #                               1).permute(0, 1, 3, 4, 2).contiguous()
        # local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
        #                                               self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        # local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        # Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
        #     0, 3, 4, 1, 5, 2)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        Raw_conv = self.firstconv(y)
        Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St_HSARes1(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St_HSARes1, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.firstconv = nn.Conv2d(in_channels=1, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        # local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        # up_y = self.repeat_y(y)
        # cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        # scale_int = math.ceil(self.scale)
        # cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
        #                               1).permute(0, 1, 3, 4, 2).contiguous()
        # local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
        #                                               self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        # local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        # Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
        #     0, 3, 4, 1, 5, 2)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        Raw_conv = self.firstconv(y)
        Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_St_StRes2(nn.Module):
    def __init__(self, msfa_size):
        super(Net_St_StRes2, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.firstconv = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.relu = nn.LeakyReLU(0.2)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        # self.convt_br1_front = self.make_ma_layer(branch_block_front, msfa_size)
        self.convt_F1 = self.make_layer(_Conv_Block)
        # self.convt_F1 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_Block)
        # self.convt_F2 = self.make_ma_layer(_Conv_Block, msfa_size)
        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        # out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        # y = torch.sum(x,1)
        # y = y.contiguous().view(y.size(0),1 ,y.size(1), y.size(2))
        # buff_x = x[0, 0, :, :].cpu().numpy()
        # buff_x1 = x[0, 3, :, :].cpu().numpy()
        # buff_y = y[0, 0, :, :].cpu().numpy()
        # buff_z = z[0, 0, :, :].cpu().numpy()
        # local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        # up_y = self.repeat_y(y)
        # cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        # scale_int = math.ceil(self.scale)
        # cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
        #                               1).permute(0, 1, 3, 4, 2).contiguous()
        # local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
        #                                               self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        # local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        # Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
        #     0, 3, 4, 1, 5, 2)
        # Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        Raw_conv1 = self.firstconv(y)
        Raw_conv = self.relu(Raw_conv1)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        # return HR_4x
        return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class L1_Charbonnier_mean_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_mean_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class reconstruction_loss(nn.Module):
    """reconstruction loss of raw_msfa"""

    def __init__(self, msfa_size):
        super(reconstruction_loss, self).__init__()
        self.wt = 1
        self.msfa_size = msfa_size
        # self.mse_loss = nn.MSELoss(reduce=True, size_average=False)
        self.mse_loss = L1_Charbonnier_mean_loss()

    def get_msfa(self, img_tensor, msfa_size):
        mask = torch.zeros_like(img_tensor)
        for i in range(0, msfa_size):
            for j in range(0, msfa_size):
                mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
        # buff_raw1 = mask[0, 1, :, :].cpu().detach().numpy()
        # buff_raw2 = img_tensor[0, 1, :, :].cpu().detach().numpy()
        return torch.sum(mask.mul(img_tensor), 1)

    def forward(self, X, Y):
        loss = self.mse_loss(self.get_msfa(X, self.msfa_size), self.get_msfa(Y, self.msfa_size))
        return loss

def get_sparsecube_raw(img_tensor, msfa_size):
    mask = torch.zeros_like(img_tensor)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1

    return mask.mul(img_tensor), torch.sum(mask.mul(img_tensor), 1).unsqueeze(1)

class Net_WO_WB_buff(nn.Module):
    def __init__(self, msfa_size):
        super(Net_WO_WB, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        # self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)

        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, self.mcm_ksize, padding=(self.mcm_ksize-1)//2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        # Raw_conv1 = self.mosaic_conv(y)
        # Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        return HR_4x
        # return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class Net_WO_WB(nn.Module):
    def __init__(self, msfa_size):
        super(Net_WO_WB, self).__init__()
        self.scale = 1
        self.outC = msfa_size**2
        self.mcm_ksize = msfa_size+2
        # self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        # self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_br1_front = self.make_ma_layer(branch_block_front_msfasize, msfa_size)
        # self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        # self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)

        # self.convt_br1_back = self.make_layer(branch_block_back)
        self.convt_br1_back = self.make_ma_layer(branch_block_back_msfasize, msfa_size)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        N, C, H, W = y.size()
        msfa_size = 5
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:msfa_size, 0:msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(msfa_size, msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / msfa_size), 1)
        local_weight1 = local_weight1.view(h_pattern_n * msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * msfa_size * W, -1, self.outC)
        for i in range(0, int(H / msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * msfa_size * h_pattern_n * W:(i + 1) * msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        convt_br1_front = self.convt_br1_front(Raw_conv_buff)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        return HR_4x
        # return torch.add(HR_4x, WB_norelu)
        # return WB_norelu

class HSA_Mpattern(nn.Module):
    def __init__(self, msfa_size):
        super(HSA_Mpattern, self).__init__()
        self.scale = 1
        self.msfa_size = msfa_size
        self.outC = msfa_size**2
        if msfa_size == 5:
            self.mcm_ksize = msfa_size+2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        self.att = MALayer_Conv_msfa(msfa_size, msfa_size ** 2, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_F1 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        self.convt_F2 = self.make_ma_layer(_Conv_attention_Block_msfasize, msfa_size)
        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        # print(H, W)
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size, self.msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / self.msfa_size), 1)
        # print(local_weight1.size())
        # print(h_pattern_n, self.msfa_size, W)
        local_weight1 = local_weight1.view(h_pattern_n * self.msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * self.msfa_size * W, -1, self.outC)
        for i in range(0, int(H / self.msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W:(i + 1) * self.msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        out = self.att(Raw_conv_buff)
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return WB_norelu

# class Mpattern_opt(nn.Module):
#     def __init__(self, msfa_size, att_type):
#         super(Mpattern_opt, self).__init__()
#         self.scale = 1
#         self.msfa_size = msfa_size
#         self.outC = msfa_size**2
#         self.att_type = att_type
#         if msfa_size == 5:
#             self.mcm_ksize = msfa_size+2
#         elif msfa_size == 4:
#             self.mcm_ksize = msfa_size + 1
#         self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
#         self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
#         if att_type == 'HSA':
#             self.att = MALayer_msfa(msfa_size, msfa_size ** 2, 4)
#         elif att_type == 'LSA':
#             self.att = CA_AA_par_Layer1(msfa_size, msfa_size ** 2, 4)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=True)
#         self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
#                                           bias=True)
#         if att_type == 'HSA':
#             self.convt_F1 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
#             self.convt_F2 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
#         elif att_type == 'LSA':
#             self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
#             self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
#         elif att_type == 'None':
#             self.convt_F1 = self.make_layer(_Conv_Block)
#             self.convt_F2 = self.make_layer(_Conv_Block)
#
#         self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.groups == msfa_size ** 2:
#                     c1, c2, h, w = m.weight.data.size()
#                     WB = get_WB_filter_msfa(msfa_size)
#                     for i in m.parameters():
#                         i.requires_grad = False
#                     m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def make_layer(self, block):
#         layers = []
#         layers.append(block())
#         return nn.Sequential(*layers)
#
#     def make_ma_layer(self, block, msfa_size):
#         layers = []
#         layers.append(block(msfa_size))
#         return nn.Sequential(*layers)
#
#     def forward_once(self, x):
#         x = self.conv_input(x)
#         out = self.convt_F1(x)
#         out = self.convt_F2(out)
#         return out
#
#     def repeat_y(self, y):
#         scale_int = math.ceil(self.scale)
#         N, C, H, W = y.size()
#         y = y.view(N, C, H, 1, W, 1)
#
#         y = torch.cat([y] * scale_int, 3)
#         y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)
#
#         return y.contiguous().view(-1, C, H, W)
#
#     def forward(self, data, pos_mat):
#         x, y = data
#         WB_norelu = self.WB_Conv(x)
#         N, C, H, W = y.size()
#         # print(H, W)
#         pos_mat = pos_mat.view(1, H, W, 2)
#         pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
#         pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
#         local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
#         local_weight = local_weight.view(self.msfa_size, self.msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
#         local_weight1 = local_weight.clone()
#         cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
#         cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
#                                       1).permute(0, 1, 3, 4, 2).contiguous()
#
#         h_pattern_n = 1
#         # This h_pattern_n can divide H / msfa_size as a int
#         local_weight1 = local_weight1.repeat(h_pattern_n, int(W / self.msfa_size), 1)
#         # print(local_weight1.size())
#         # print(h_pattern_n, self.msfa_size, W)
#         local_weight1 = local_weight1.view(h_pattern_n * self.msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
#         local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * self.msfa_size * W, -1, self.outC)
#         for i in range(0, int(H / self.msfa_size / h_pattern_n)):
#             cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W:(i + 1) * self.msfa_size * h_pattern_n * W, :, :]
#             if i == 0:
#                 Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
#             else:
#                 Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)
#
#         Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
#         Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
#         Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
#         Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)
#
#         if self.att_type != 'None':
#             out = self.att(Raw_conv_buff)
#         else:
#             out = Raw_conv_buff
#         out = self.relu1(out)
#         out = self.forward_once(out)
#         out = self.conv_tail(out)
#         return torch.add(out, WB_norelu)
#         # return out

class Mpattern_opt(nn.Module):
    def __init__(self, msfa_size, att_type):
        super(Mpattern_opt, self).__init__()
        self.scale = 1
        self.msfa_size = msfa_size
        self.outC = msfa_size**2
        self.att_type = att_type
        if msfa_size == 5:
            self.mcm_ksize = msfa_size+2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        if att_type == 'HSA':
            self.att = MALayer_msfa(msfa_size, msfa_size ** 2, 4)
        elif att_type == 'LSA':
            self.att = CA_AA_par_Layer1(msfa_size, msfa_size ** 2, 4)
        elif att_type == 'SE':
            self.att = SELayer(msfa_size ** 2, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        if att_type == 'HSA':
            self.convt_F1 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
        elif att_type == 'LSA':
            self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        elif att_type == 'SE':
            self.convt_F1 = self.make_layer(_Conv_SE_Block)
            self.convt_F2 = self.make_layer(_Conv_SE_Block)
        elif att_type == 'None':
            self.convt_F1 = self.make_layer(_Conv_Block)
            self.convt_F2 = self.make_layer(_Conv_Block)

        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        # print(H, W)
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size, self.msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / self.msfa_size), 1)
        # print(local_weight1.size())
        # print(h_pattern_n, self.msfa_size, W)
        local_weight1 = local_weight1.view(h_pattern_n * self.msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * self.msfa_size * W, -1, self.outC)
        for i in range(0, int(H / self.msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W:(i + 1) * self.msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        if self.att_type != 'None':
            out = self.att(Raw_conv_buff)
        else:
            out = Raw_conv_buff
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return WB_norelu


class Mpattern_opt_fast(nn.Module):
    def __init__(self, msfa_size, att_type):
        super(Mpattern_opt_fast, self).__init__()
        self.scale = 1
        self.msfa_size = msfa_size
        self.outC = msfa_size**2
        self.att_type = att_type
        if msfa_size == 5:
            self.mcm_ksize = msfa_size+2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        self.MosaicConv = nn.Conv2d(in_channels=1, out_channels=self.outC, kernel_size=self.mcm_ksize, stride=msfa_size, padding=(self.mcm_ksize - 1) // 2, bias=False)
        self.shuffleup = nn.PixelShuffle(msfa_size)
        if att_type == 'HSA':
            self.att = MALayer_msfa(msfa_size, msfa_size ** 2, 4)
        elif att_type == 'LSA':
            self.att = CA_AA_par_Layer1(msfa_size, msfa_size ** 2, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        if att_type == 'HSA':
            self.convt_F1 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
        elif att_type == 'LSA':
            self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        elif att_type == 'None':
            self.convt_F1 = self.make_layer(_Conv_Block)
            self.convt_F2 = self.make_layer(_Conv_Block)

        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size*self.msfa_size, self.outC, 1, self.mcm_ksize,self.mcm_ksize)
        # c1, c2, h, w = self.MosaicConv.weight.data.size()
        for i in range(self.msfa_size):
            for j in range(self.msfa_size):
                self.MosaicConv.weight.data = local_weight[5*i+j, :, :, :, :]
                if i ==0 and j ==0:
                    mosaicfm = self.MosaicConv(torch.roll(y, (i, j), dims=(2, 3))).unsqueeze(-1) # N, C, H/msfa_size. W/msfa_size
                else:
                    mosaicfm_single = self.MosaicConv(torch.roll(y, (i, j), dims=(2, 3))).unsqueeze(-1) # N, C, H/msfa_size. W/msfa_size
                    mosaicfm = torch.cat((mosaicfm, mosaicfm_single), dim=4)
        mosaicfm = mosaicfm.permute(0, 1, 4, 2, 3).view(N*self.outC, self.msfa_size**2, int(H/self.msfa_size), int(W/self.msfa_size))
        mosaicfm = self.shuffleup(mosaicfm).view(N, self.outC, 1, H, W).squeeze(2)
        if self.att_type != 'None':
            out = self.att(mosaicfm)
        else:
            out = mosaicfm
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return WB_norelu

class Mpattern_opt_fast2(nn.Module):
    def __init__(self, msfa_size, att_type):
        super(Mpattern_opt_fast2, self).__init__()
        self.scale = 1
        self.msfa_size = msfa_size
        self.outC = msfa_size**2
        self.att_type = att_type
        if msfa_size == 5:
            self.mcm_ksize = msfa_size+2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        self.MosaicConv = nn.Conv2d(in_channels=msfa_size**2, out_channels=self.outC*msfa_size**2, kernel_size=self.mcm_ksize, stride=msfa_size, padding=0, bias=False, groups=msfa_size**2)
        self.shuffleup = nn.PixelShuffle(int(msfa_size))
        if att_type == 'HSA':
            self.att = MALayer_msfa(msfa_size, msfa_size ** 2, 4)
        elif att_type == 'LSA':
            self.att = CA_AA_par_Layer1(msfa_size, msfa_size ** 2, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        if att_type == 'HSA':
            self.convt_F1 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
        elif att_type == 'LSA':
            self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        elif att_type == 'None':
            self.convt_F1 = self.make_layer(_Conv_Block)
            self.convt_F2 = self.make_layer(_Conv_Block)

        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2 and m.kernel_size == (2*msfa_size-1,2*msfa_size-1):
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = x.size()
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size*self.msfa_size, self.outC, 1, self.mcm_ksize,self.mcm_ksize)
        local_weight = local_weight.view(self.msfa_size*self.msfa_size*self.outC, 1, self.mcm_ksize,self.mcm_ksize)
        mosaicfm = self.MosaicConv(y)
        for c in range(0, self.outC, 1):
            if c == 0:
                order = list(range(c,self.outC*self.msfa_size**2+c, self.outC))
            else:
                order = order + list(range(c,self.outC*self.msfa_size**2+c, self.outC))
        mosaicfm = mosaicfm[:, order, :, :]
        mosaicfm = self.shuffleup(mosaicfm)
        if self.att_type != 'None':
            out = self.att(mosaicfm)
        else:
            out = mosaicfm
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return WB_norelu

class Mpattern_opt_newMCM(nn.Module):
    # this version is package the mcm in Mpattern_opt_fast2, making mcm as an easy-using and fast module
    def __init__(self, msfa_size, att_type, conv_type, inC):
        super(Mpattern_opt_newMCM, self).__init__()
        self.scale = 1
        self.msfa_size = msfa_size
        self.inC = inC
        self.outC = msfa_size**2
        self.att_type = att_type
        if msfa_size == 5:
            self.mcm_ksize = msfa_size + 2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.MosaicConv = ConvMosaic_new2(in_channels=self.inC, out_channels=self.outC, msfa_size=msfa_size, kernel_size=self.mcm_ksize, stride=1, padding=(self.mcm_ksize - 1) // 2, bias=False)
        if att_type == 'HSA':
            self.att = MALayer_msfa(msfa_size, self.outC, 4)
        elif att_type == 'LSA':
            self.att = CA_AA_par_Layer1(msfa_size, self.outC, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=self.outC, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        if att_type == 'HSA':
            if conv_type == 'st':
                self.convt_F1 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
                self.convt_F2 = self.make_ma_layer(_Conv_HSA_Block_msfasize, msfa_size)
            elif conv_type == 'mcm':
                self.convt_F1 = self.make_ma_layer(_MosaicConv_HSA_Block_msfasize, msfa_size)
                self.convt_F2 = self.make_ma_layer(_MosaicConv_HSA_Block_msfasize, msfa_size)
        elif att_type == 'LSA':
            self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
            self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        elif att_type == 'None':
            self.convt_F1 = self.make_layer(_Conv_Block)
            self.convt_F2 = self.make_layer(_Conv_Block)

        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2 and m.kernel_size == (2*msfa_size-1,2*msfa_size-1):
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        mosaicfm = self.MosaicConv(y)
        if self.att_type != 'None':
            out = self.att(mosaicfm)
        else:
            out = mosaicfm
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return out

if __name__ == '__main__':

    # torch.cuda.empty_cache()
    msfa_size = 2
    speed_test_mode = False
    cuda_id = 'cuda:1'
    batchsize = 3
    if speed_test_mode:
        net = Mpattern_opt_fast2(msfa_size=msfa_size, att_type='HSA').cuda(cuda_id)
    else:
        net = Mpattern_opt_CFA(msfa_size=msfa_size, att_type='HSA').cuda(cuda_id)
        inC = 1
        # net = Mpattern_opt_newMCM(msfa_size=msfa_size, att_type='HSA', conv_type='st', inC=inC).cuda(cuda_id)

    sparse_raw_syn = torch.randn((batchsize, msfa_size**2, 1000, 800)).cuda(cuda_id)
    if speed_test_mode:
        raw_syn = torch.randn((batchsize, msfa_size**2, 1300, 1300)).cuda(cuda_id)
    else:
        raw_syn = torch.randn((batchsize, inC, 1000, 800)).cuda(cuda_id)
    scale_coord_map = torch.randn((1, 1000*800, 2)).cuda(cuda_id)
    flops, params = profile(net, inputs=([sparse_raw_syn, raw_syn], scale_coord_map,))
    print('flops, params:', flops, params)
    # x = net(x)
    # print(x.shape)
