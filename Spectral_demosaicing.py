import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time, math
from os import listdir
from os.path import join
from PIL import Image
import random
import os
import h5py
import struct
import datetime
from astropy.modeling import models, fitting

def randcrop_4cube(a, b, c, d, crop_size, msfa_size):
    [wid, hei, bands] = a.shape
    # Width = random.randint(0, wid - crop_size - 1)
    Width = random.randint(0, (wid - crop_size)//msfa_size)*msfa_size
    # Height = random.randint(0, hei - crop_size - 1)
    Height = random.randint(0, (hei - crop_size)//msfa_size)*msfa_size
    return a[Width:(Width + crop_size),  Height:(Height + crop_size), :], \
           b[Width:(Width + crop_size), Height:(Height + crop_size), :], \
           c[Width:(Width + crop_size), Height:(Height + crop_size), :], \
           d[Width:(Width + crop_size), Height:(Height + crop_size), :]

def randcrop(a, crop_size):
    [wid, hei, nband]=a.shape
    crop_size1 = crop_size
    Width = random.randint(0, wid - crop_size1 )
    Height = random.randint(0, hei - crop_size1 )

    return a[Width:(Width + crop_size1),  Height:(Height + crop_size1), :]

def msfaTOcube(raw, msfa_size):
    mask = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=raw.dtype)
    cube = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=raw.dtype)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1

    for i in range(msfa_size**2):
        cube[:, :, i] = raw * (mask[:, :, i])
    return cube

def msfaTOcube_tensor(raw, msfa_size):
    mask = torch.zeros((msfa_size**2, raw.shape[1], raw.shape[2]), dtype=raw.dtype)
    cube = torch.zeros((msfa_size**2, raw.shape[1], raw.shape[2]), dtype=raw.dtype)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
    raw = raw.repeat(msfa_size**2, 1, 1)
    cube = raw * mask
    return cube

def quadBayerTOcube(image):
    img = np.expand_dims(image, axis=2)
    img = np.repeat(img, 3, axis=2)
    # Quad R 
    img[::4, ::4, 1:3] = 0
    img[1::4, 1::4, 1:3] = 0
    img[::4, 1::4, 1:3] = 0
    img[1::4, ::4, 1:3] = 0

    # Quad B 
    img[3::4, 2::4, 0:2] = 0
    img[3::4, 3::4, 0:2] = 0
    img[2::4, 3::4, 0:2] = 0
    img[2::4, 2::4, 0:2] = 0

    # Quad G12
    img[1::4, 2::4, ::2] = 0
    img[1::4, 3::4, ::2] = 0
    img[::4, 2::4, ::2] = 0
    img[::4, 3::4, ::2] = 0

    # Quad G21
    img[2::4, 1::4, ::2] = 0
    img[3::4, 1::4, ::2] = 0
    img[2::4, ::4, ::2] = 0
    img[3::4, ::4, ::2] = 0

    return img

def mask_input(GT_image, msfa_size):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1
    input_image = mask * GT_image
    return input_image

def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor

def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor

def DataMaker_Singleimg(single_image, msfa_size, patch_size=120, patch_stride = 60):
    single_image = single_image/255.0
    #Todo: This label is sparse, which is not correct
    label = msfaTOcube(single_image, msfa_size).transpose(2, 0, 1) #16*H*W
    label = torch.from_numpy(label).float()
    label = torch.unsqueeze(label, 0)  # 1*16*H*W
    patch_label = nn.functional.unfold(label, kernel_size=patch_size, padding=msfa_size, stride=patch_stride)
    patch_label = patch_label.reshape(msfa_size**2, patch_size, patch_size, -1)
    patch_label = patch_label.permute(3, 0, 1, 2)

    single_image = torch.from_numpy(single_image).float()
    single_image = torch.unsqueeze(single_image, 0)  # 1*H*W
    single_image = torch.unsqueeze(single_image, 0)  # 1*1*H*W
    patch_raw = nn.functional.unfold(single_image, kernel_size=patch_size, padding=msfa_size, stride=patch_stride)
    patch_raw = patch_raw.reshape(1, patch_size, patch_size, -1)
    patch_raw = patch_raw.permute(3, 0, 1, 2)
    pr = patch_raw.numpy()
    return [patch_raw, patch_label, patch_label]

def get_filename(path, filetype=None):
    name=[]
    files = os.listdir(path)
    files.sort()
    for i in files:
        if filetype:
            if os.path.splitext(i)[1] == filetype:
                name.append(i.replace(filetype, ''))
        elif os.path.splitext(i)[1] == '':
            name.append(i)
    return name

def loadCube(path):
    """
    Load a spectral cube from Matlab HDF5 format .mat file
    :param path: Path of souce file
    :return: spectral cube (cube) and bands (bands)
    """
    with h5py.File(path, 'r') as mat:
        cube = np.array(mat['cube']).T
        cube_bands = np.array(mat['bands']).squeeze()
    return cube, cube_bands

def loadMosaic(path):
    """
    Load a spectral cube from Matlab HDF5 format .mat file
    :param path: Path of souce file
    :return: spectral cube (cube) and bands (bands)
    """
    with h5py.File(path, 'r') as mat:
        mosaic = np.array(mat['mosaic']).T
    return mosaic

def compute_psnr(a, b, peak):
    """
    compute the peak SNR between two arrays

    :param a: first array
    :param b: second array with the same shape
    :param peak: scalar of peak signal value (e.g. 255, 1023)

    :return: psnr (scalar)
    """
    sqrd_error = compute_mse(a, b)
    mse = sqrd_error.mean()
    # TODO do we want to take psnr of every pixel first and then mean?
    # return 10 * np.log10((peak ^ 2) / mse), mse
    return 10 * np.log10((peak ** 2) / mse), mse

def compute_mse(a, b):
    """
    Compute the mean squared error between two arrays

    :param a: first array
    :param b: second array with the same shape

    :return: MSE(a, b)
    """
    assert a.shape == b.shape
    diff = a - b
    return np.power(diff, 2)

def saveCubeBands(path, cube, bands=None, norm_factor=None):
    """
    Save a spectra cube in Matlab HDF5 format
    :param path: Destination filename as full path
    :param cube: Spectral cube as Numpy array
    :param bands: Bands of spectral cube as Numpy array
    :param norm_factor: Normalization factor to source image counts
    """
    hdf5storage.write({u'cube': cube,
                       u'bands': bands,
                       u'norm_factor': norm_factor}, '.',
                       path, matlab_compatible=True)

def saveCube(path, cube, bands=None):
    """
    Save a spectra cube in Matlab HDF5 format
    :param path: Destination filename as full path
    :param cube: Spectral cube as Numpy array
    """
    hdf5storage.write({u'cube': cube.T,
                       u'bands': bands}, '.',
                       path, matlab_compatible=True)

def flatten_and_normalize(arr):
    """
    Vectorize a NxMxC matrix and normalize it for error calculation
    :param arr: Array to vectorize
    :return: Normalized N*MxC array
    """
    h, w, c = arr.shape
    arr = arr.reshape([h * w, c])
    norms = np.linalg.norm(arr, ord=2, axis=-1)
    norms[norms == 0] = 1 # remove zero division problems

    return arr / norms[:, np.newaxis]

def compute_sam(a, b):
    """
    spectral angle mapper

    :param a: first array
    :param b: second array with the same shape

    :return: mean of per pixel SAM
    """
    assert a.shape == b.shape
    # normalize each array per pixel, so the dot product will be determined only by the angle
    a = flatten_and_normalize(a)
    b = flatten_and_normalize(b)
    angles = np.sum(a * b, axis=1)
    angles = np.clip(angles, -1, 1)
    sams   = np.arccos(angles)

    return sams.mean()

def read_bin_file(filepath):
    '''
        read '.bin' file to 2-d numpy array

    :param path_bin_file:
        path to '.bin' file

    :return:
        2-d image as numpy array (float32)

    '''

    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]

    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32)

    return data_2d

def save_bin(filepath, arr):
    '''
        save 2-d numpy array to '.bin' files with uint16

    @param filepath:
        expected file path to store data

    @param arr:
        2-d numpy array

    @return:
        None

    '''

    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape

    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

def loadCube_nobands(path):
    """
    Load a spectral cube from Matlab HDF5 format .mat file
    :param path: Path of souce file
    :return: spectral cube (cube) and bands (bands)
    """
    with h5py.File(path, 'r') as mat:
        cube = np.array(mat['cube'])
    return cube

def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y%m%d')
    time_str = cur_time.strftime('%H%M%S')

    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + '_' + time_str

def ergas_matlab(GT,P,r=1):
    n_samples = GT.shape[0] * GT.shape[1]
    nb = GT.shape[2]
    #RMSE
    aux = np.sum(np.sum((P - GT)**2, 0), 0)/n_samples
    rmse_per_band = np.sqrt(aux)
    # ergas
    mean_y = np.sum(np.sum(GT, 0), 0)/n_samples
    ergas = 100*r*np.sqrt(np.sum((rmse_per_band/mean_y)**2)/nb)

    return ergas

def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)

def seed_set(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # if d_min < 0:
    #     origndata += torch.abs(d_min)
    #     d_min = origndata.min()

def tensor_normal(origndata):
    d_min = origndata.min()
    d_max = origndata.max()
    dst = d_max - d_min
    norm_data = (origndata - d_min).true_divide(dst)
    return norm_data

def tensor_normal_3cube(origndata, origndata1, origndata2):

    d_min = origndata.min()
    d_min1 = origndata1.min()
    d_min2 = origndata2.min()
    d_min = torch.min(torch.tensor([d_min, d_min1, d_min2]))

    if d_min < 0:
        origndata += torch.abs(d_min)
        origndata1 += torch.abs(d_min)
        origndata2 += torch.abs(d_min)
        d_min = origndata.min()
        d_min1 = origndata1.min()
        d_min2 = origndata2.min()
        d_min = torch.min(torch.tensor([d_min, d_min1, d_min2]))

    d_max = origndata.max()
    d_max1 = origndata1.max()
    d_max2 = origndata2.max()
    d_max = torch.max(torch.tensor([d_max, d_max1, d_max2]))

    dst = d_max - d_min
    norm_data = (origndata - d_min).true_divide(dst)
    norm_data1 = (origndata1 - d_min).true_divide(dst)
    norm_data2 = (origndata2 - d_min).true_divide(dst)
    return norm_data, norm_data1, norm_data2

def tensor_normal_3cube_bandwise(origndata, origndata1, origndata2):
    C = origndata.shape[-3]
    for ch in range(C):
        origndata[..., ch, :, :], origndata1[..., ch, :, :], origndata2[..., ch, :, :] = tensor_normal_3cube(origndata[..., ch, :, :], origndata1[..., ch, :, :], origndata2[..., ch, :, :])
    return origndata, origndata1, origndata2

def fit_gaosi(data, bins):
    hx, xedge = np.histogram(data, bins)
    xedge = (xedge[1:] + xedge[:-1]) / 2

    g_init = models.Gaussian1D(amplitude=np.max(hx), mean=np.mean(data), stddev=np.std(data))
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, xedge, hx)

    return g.mean.value, g.stddev.value, g

def maskedinput_and_returnmask(GT_image, msfa_size, dm_mode='msfa'):
    mask1 = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask1[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1

    if dm_mode == 'cfa_rggb':
        mask = mask1[:, :, [0, 1, 3]]
        mask[:, :, 1] = mask1[:, :, 1] + mask1[:, :, 2]
    elif dm_mode == 'msfa':
        mask = mask1

    input_image = mask * GT_image
    return input_image, mask

def create_image_block(image, row_number, col_number, row_axis=0, col_axis=1):
    # image should be h,w,c, and split to [row_number][row_number][patchx patchy c]
    block_row = np.array_split(image, row_number, axis=row_axis)  # 垂直方向切割，得到很多横向长条
    print(image.shape)
    img_blocks = []
    for block in block_row:
        block_col = np.array_split(block, col_number, axis=col_axis)  # 水平方向切割，得到很多图像块
        img_blocks += [block_col]
    return img_blocks

def recover_image_from_blocks(img_blocks):
    rowimg_blocks = []
    for block_col in img_blocks:
        block = np.hstack(block_col)  # 水平方向切割，得到很多图像块
        rowimg_blocks += [block]
    image = np.vstack(rowimg_blocks)
    return image

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

def pixelshuffledown_torchtensor(x, scale):
    b, C, h, w = x.size()
    Cout = C * scale ** 2
    hout = h // scale
    wout = w // scale
    x = x.contiguous().view(b, C, hout, scale, wout, scale)
    x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
    x = x.contiguous().view(b, Cout, hout, wout)
    return x


if __name__ == '__main__':
    mos = np.random.rand(8,16)
