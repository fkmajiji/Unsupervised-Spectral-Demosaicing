###有坐标图输入的单模型多图评价方法
import sys,os
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time
import matplotlib.pyplot as plt
from libtiff import TIFFfile, TIFFimage
from PIL import Image
from Spectral_demosaicing import input_matrix_wpn as input_matrix_wpn_msfasize
from Spectral_demosaicing import msfaTOcube, get_filename, pixel_shuffle_inv
import torch.nn as nn
from collections import OrderedDict
import pandas as pd

def load_img(filepath):
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    return img

periodic_avg_dict = OrderedDict()
for epoch_num in range(10, 30, 10):
    type_name = 'ICVL_LSA_5_EItrain_Transrandom_alpha1_st1_240324_220911'
    epoch_num = str(epoch_num)
    parser = argparse.ArgumentParser(description="USD real dataset")
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument("--model", default="checkpoint/"+type_name+"/De_happy_model_epoch_"+str(epoch_num)+".pth", type=str, help="model path")
    parser.add_argument("--dataset", default="Mosaic25", type=str, help="dataset name, Default: Mosaic25")
    parser.add_argument("--msfa_size", default=5, type=int, help="scale factor, Default: 5")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    opt = parser.parse_args()
    cuda = True
    save_flag = False
    show_flag = False
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print(opt.model)
    model = torch.load(opt.model)["model"]

    avg_psnr_predicted = 0.0
    avg_psnr_PPID = 0.0
    avg_sam_predicted = 0.0
    avg_sam_PPID = 0.0
    avg_ssim_predicted = 0.0
    avg_ssim_PPID = 0.0
    avg_ergas_predicted = 0.0
    avg_elapsed_time = 0.0
    cube_pdc_var_avg = 0.0
    sample_num = 0

    if opt.dataset == 'Mosaic25':
        if opt.msfa_size == 5:
            testimg_path = '/data1/fengkai/dataset/25bands/600_875/paper_dataset/'
            name_list = get_filename(testimg_path, '.tif')
            save_path = 'results/real/'
        elif opt.msfa_size == 4:
            testimg_path = '/data1/user1/dataset/test_real_mine/'
            name_list = get_filename(testimg_path, '.tif')
            save_path = 'results/real/'

    with torch.no_grad():
        for image_name in name_list:
                print("Processing ", image_name)
                sample_num = sample_num + 1
                image_name = image_name.split('.', 1)[0]

                raw = Image.open(testimg_path + image_name + '.tif')
                if raw.mode == 'L':
                    raw_max_theory = 255.0
                elif raw.mode == 'I;16':
                    raw_max_theory = 1023.0
                raw = np.array(raw)

                print(np.max(raw), np.min(raw))

                im_l_y = msfaTOcube(raw, opt.msfa_size)
                buff = im_l_y[:, :, 1]
                im_l_y = im_l_y.astype(float)
                raw = raw.astype(float)

                im_input = im_l_y / raw_max_theory
                raw = raw / raw_max_theory

                max_new = np.max(im_input)
                print(max_new)
                im_input = im_input / max_new

                im_l_y = im_l_y.transpose(2, 0, 1)
                im_input = im_input.transpose(2, 0, 1)
                raw = raw.transpose(0, 1)

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1],
                                                                             im_input.shape[2])
                raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])
                h, w = raw.size()[-2:]
                h_pattern_n = 1
                int_size = h_pattern_n * opt.msfa_size
                paddingBottom = int(np.ceil(h / int_size) * int_size - h)
                im_input = nn.ZeroPad2d((0, 0, 0, paddingBottom))(im_input)
                raw = nn.ZeroPad2d((0, 0, 0, paddingBottom))(raw)

                scale_coord_map = input_matrix_wpn_msfasize(raw.shape[2], raw.shape[3], opt.msfa_size)
                if cuda:
                    model = model.cuda()
                    im_input = im_input.cuda()
                    raw = raw.cuda()
                    scale_coord_map = scale_coord_map.cuda()
                else:
                    model = model.cpu()

                start_time = time.time()
                HR_4x = model([im_input, raw], scale_coord_map)
                elapsed_time = time.time() - start_time

                HR_4x = HR_4x[..., :h, :w]
                raw = raw[..., :h, :w]

                HR_4x = HR_4x.cpu()
                im_h_y = HR_4x.data[0].numpy().astype(np.float32)
                raw = raw.cpu().data[0].numpy().astype(np.float32)

                cube_pdc_var = np.ones((1, opt.msfa_size**2))
                for bn in range(opt.msfa_size**2):
                    singleband_pdc_avg = pixel_shuffle_inv(np.expand_dims((np.expand_dims(im_h_y[bn,:,:], 0)),0), opt.msfa_size)
                    singleband_pdc_avg = np.mean(np.mean(singleband_pdc_avg, -1), -1)
                    singleband_pdc_avg = singleband_pdc_avg.var(axis=1)
                    cube_pdc_var[0, bn] = singleband_pdc_avg
                rawmosaic_pdc_avg = pixel_shuffle_inv(np.expand_dims(raw[:, :, :],0), opt.msfa_size)
                rawmosaic_pdc_avg = np.mean(np.mean(rawmosaic_pdc_avg, -1), -1)

                im_h_y = im_h_y * 255.
                im_h_y = np.rint(im_h_y)
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y.astype(np.uint8)
                im_h_y = im_h_y.astype(np.float)

                raw = raw * 255.
                raw[raw < 0] = 0
                raw[raw > 255.] = 255.

                im_input = im_input.cpu()
                im_input = im_input.data[0].numpy().astype(np.float32)
                im_input = im_input * 255.
                im_input[im_input < 0] = 0
                im_input[im_input > 255.] = 255.

                if save_flag:
                    tiff = TIFFimage(im_h_y.astype(np.uint8), description='')
                    tiff.write_file((save_path + image_name + '_' + type_name + epoch_num + '.tif'),
                                    compression='none')
                    del tiff  # flushes data to disk

                if show_flag:
                    if opt.msfa_size == 5:
                        buff = np.concatenate((im_h_y[22:23, :, :], im_h_y[12:13, :, :], im_h_y[4:5, :, :])).transpose(1, 2, 0)
                    elif opt.msfa_size == 4:
                        buff = np.concatenate((im_h_y[14:15, :, :], im_h_y[2:3, :, :], im_h_y[9:10, :, :])).transpose(1, 2, 0)

                    plt.imshow(buff.astype(np.uint8))
                    plt.show()

                avg_elapsed_time += elapsed_time
                cube_pdc_var_avg += np.mean(cube_pdc_var, 1)

                del HR_4x
                del raw
                del im_input

        cube_pdc_var_avg2 = cube_pdc_var_avg / sample_num
        print("cube_pdc_var_avg2=", cube_pdc_var_avg2)
        periodic_avg_dict[epoch_num] = cube_pdc_var_avg2
        data_frame = pd.DataFrame(
            data=periodic_avg_dict, index=range(1, 1 + 1))
        data_frame.to_csv((save_path + type_name + '.csv'), index_label='index')

print("Dataset=", opt.dataset)
print("It takes average {}s for processing".format(avg_elapsed_time/len(name_list)))
