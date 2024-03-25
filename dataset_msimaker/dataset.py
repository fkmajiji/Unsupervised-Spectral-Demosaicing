from scipy.interpolate import interp1d
from os import listdir
from os.path import join
import torch
import torch.utils.data as data
from libtiff import TIFFfile
from PIL import Image
import numpy as np
import random
from Spectral_demosaicing import loadCube, mask_input, msfaTOcube, pixel_shuffle_inv
from torchvision.transforms import Compose, ToTensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".mat", '.h5'])

def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    # tif = TIFFfile(filepath+'/IMECMine_D65.tif')
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    # img_test = Image.fromarray(img[:,:,1])
    return img

def randcrop(a, crop_size):
    [wid, hei, nband]=a.shape
    crop_size1 = crop_size
    # print(wid, hei, crop_size1)
    Width = random.randint(0, wid - crop_size1 - 1)
    Height = random.randint(0, hei - crop_size1 - 1)

    return a[Width:(Width + crop_size1),  Height:(Height + crop_size1), :]

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def randcrop_multi_cube(a,b, crop_size, msfa_size):
    [wid, hei, _]=a.shape
    Width = random.randint(0, (wid - crop_size)//msfa_size)*msfa_size
    Height = random.randint(0, (hei - crop_size)//msfa_size)*msfa_size
    return a[Width:(Width + crop_size),  Height:(Height + crop_size),:],\
           b[Width:(Width + crop_size),  Height:(Height + crop_size),:]

def randcrop_realmosaic(a, crop_size, msfa_size):
    [wid, hei]=a.shape
    # Width = random.randint(0, wid - crop_size - 1)
    Width = random.randint(0, (wid - crop_size)//msfa_size)*msfa_size
    # Height = random.randint(0, hei - crop_size - 1)
    Height = random.randint(0, (hei - crop_size)//msfa_size)*msfa_size
    return a[Width:(Width + crop_size),  Height:(Height + crop_size)]

class DatasetFromFolder(data.Dataset):
    def __init__(self, msfa_size, image_dir,norm_flag, patch_size=120, input_transform=None, target_transform=None, augment=False):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        print(self.image_filenames)
        random.shuffle(self.image_filenames)
        print(self.image_filenames)
        #ToDo 确认这里是否需要随机打乱文件，由于不同光照的存在
        self.msfa_size = msfa_size
        self.crop_size = calculate_valid_crop_size(patch_size, msfa_size)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag
        self.illum_aug_flag = True
    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        input_image = input_image.astype(np.float32)

        if self.norm_flag:
            norm_name = 'maxnorm'
            max_raw = np.max(input_image)
            max_subband = np.max(np.max(input_image, axis=0), 0)
            norm_factor = max_raw / max_subband
            for bn in range(self.msfa_size**2):
                input_image[:, :, bn] = input_image[:, :, bn] * norm_factor[bn]
        input_image = randcrop(input_image, self.crop_size)
        if self.augment:
            if np.random.uniform() < 0.5:
                input_image = np.fliplr(input_image)
            if np.random.uniform() < 0.5:
                input_image = np.flipud(input_image)
            # ToDo 增强方式是否足够
            input_image = np.rot90(input_image, k=np.random.randint(0, 4))
        target = input_image.copy()
        #ToDo 确认这里的mask
        ###原本的im_gt_y按照实际相机滤波阵列排列
        input_image = mask_input(target, self.msfa_size)
        ###按照实际相机滤波阵列排列逆还原为从大到小的顺序
        #input_image = reorder_imec(input_image)
        #target = reorder_imec(target)
        if self.input_transform:
            raw = input_image.sum(axis=2)
            # raw = self.input_transform(raw)/255.0
            raw = torch.Tensor(raw)/255.0
            raw = raw.unsqueeze(0)
            input_image = self.input_transform(input_image)/255.0

        if self.target_transform:
            target = self.target_transform(target)/255.0

        return raw, input_image, target

    def __len__(self):
        return len(self.image_filenames)

class RawDatasetFromFolder(data.Dataset):
    def __init__(self, msfa_size, image_dir,norm_flag,patch_size=120, input_transform=None, target_transform=None):
        super(RawDatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        print(self.image_filenames)
        random.shuffle(self.image_filenames)
        print(self.image_filenames)
        #ToDo: 确认这里是否需要随机打乱文件，由于不同光照的存在
        self.msfa_size = msfa_size
        self.crop_size = calculate_valid_crop_size(100, msfa_size)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.norm_flag = norm_flag
        self.illum_aug_flag = True
    def __getitem__(self, index):
        #TODO 注意读图这里的行列问题，弄反则会引起排布的变化
        raw = Image.open(self.image_filenames[index])
        raw = np.array(raw)
        raw = raw.astype(np.float32)
        # if self.norm_flag:
        #     norm_name = 'maxnorm'
        #     max_raw = np.max(input_image)
        #     max_subband = np.max(np.max(input_image, axis=0), 0)
        #     norm_factor = max_raw / max_subband
        #     for bn in range(self.msfa_size**2):
        #         input_image[:, :, bn] = input_image[:, :, bn] * norm_factor[bn]

        raw = randcrop_realmosaic(raw, self.crop_size, self.msfa_size)
        target = msfaTOcube(raw, self.msfa_size)
        # target = input_image.copy()
        ###原本的im_gt_y按照实际相机滤波阵列排列
        # input_image = mask_input(target, self.msfa_size)
        input_image = target.copy()
        if self.input_transform:
            raw = self.input_transform(raw)/255.0
            # raw = torch.Tensor(raw)
            # raw = raw.unsqueeze(0)
            input_image = self.input_transform(input_image)/255.0

        if self.target_transform:
            target = self.target_transform(target)/255.0

        return raw, input_image, target

    def __len__(self):
        return len(self.image_filenames)

def ToTensor_transform( ):
    return Compose([
        ToTensor(),
    ])

class NTIREDatasetFromFolder(data.Dataset):
    def __init__(self,  image_dir,msfa_size, norm_flag, patch_size=120, augment=False,
                 input_transform=ToTensor_transform(), target_transform=ToTensor_transform(),
                 ):
        super(NTIREDatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        # self.wupuimage_filenames = []
        # self.puimage_filenames = []
        # for x in listdir(image_dir):
        #     cube_path = join(image_dir, x)
        #     mosaic_path = cube_path.replace('spectral_16', 'mosaic', 1)
        #     mosaic_path = mosaic_path.replace('_16.', '.raw.')
        #     raw_oth = loadMosaic(mosaic_path)
        #     if np.max(raw_oth)<(2 ** 12 - 1):
        #         self.wupuimage_filenames.append(join(image_dir, x))
        #     else:
        #         self.puimage_filenames.append(join(image_dir, x))
        random.shuffle(self.image_filenames)
        print(self.image_filenames)
        # ToDo 确认这里是否需要随机打乱文件，由于不同光照的存在
        self.msfa_size = msfa_size
        self.crop_size = calculate_valid_crop_size(patch_size, msfa_size)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag
        self.illum_aug_flag = True

    def __getitem__(self, index):
        target = loadCube(self.image_filenames[index])[0].astype(np.float32)
        target = randcrop(target, self.crop_size)
        # print(target.shape)
        if self.augment:
            if np.random.uniform() < 0.5:
                target = np.fliplr(target).copy()
            if np.random.uniform() < 0.5:
                target = np.flipud(target).copy()
            # ToDo 增强方式是否足够
            k_rot = np.random.randint(0, 4)
            target = np.rot90(target, k=k_rot).copy()

        input_image = mask_input(target, self.msfa_size)
        raw1 = input_image.sum(axis=2)

        if self.input_transform:
            raw1 = torch.Tensor(raw1)
            raw1 = raw1.unsqueeze(0)
            input_image = self.input_transform(input_image)

        if self.target_transform:
            target = self.target_transform(target)
        return raw1, input_image, target

    def __len__(self):
        return len(self.image_filenames)


