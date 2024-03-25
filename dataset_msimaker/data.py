import sys
sys.path.append('../')
from torchvision.transforms import Compose, ToTensor
from .dataset import DatasetFromFolder,  RawDatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform( ):
    return Compose([
        ToTensor(),
    ])


def target_transform( ):
    return Compose([
        # CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set_opt(train_dir, msfa_size,norm_flag, augment_flag, patch_size):
    return DatasetFromFolder(msfa_size, train_dir, norm_flag,patch_size=patch_size,
                             input_transform=input_transform(),
                             target_transform=target_transform(),
                             augment=augment_flag
                             )

def get_test_set_opt(test_dir, msfa_size,norm_flag, patch_size):
    return DatasetFromFolder(msfa_size, test_dir, norm_flag,patch_size = patch_size,
                             input_transform=input_transform( ),
                             target_transform=target_transform( )
                             )

def get_real_mosaic_training_set_opt(train_ss_dir, msfa_size,norm_flag, patch_size):

    return RawDatasetFromFolder(msfa_size, train_ss_dir, norm_flag, patch_size=patch_size,
                             input_transform=input_transform( ),
                             target_transform=target_transform( )
                             )