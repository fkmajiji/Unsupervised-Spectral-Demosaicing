import sys,os
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import matplotlib.pyplot as plt
from libtiff import TIFFfile, TIFFimage
from sklearn.metrics import mean_squared_error
from Spectral_demosaicing import input_matrix_wpn as input_matrix_wpn_msfasize
from Spectral_demosaicing import pixel_shuffle_inv, loadCube, mask_input, get_filename, ergas_matlab
import torch.nn as nn
from collections import OrderedDict
import pandas as pd

def load_img(filepath):
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    return img

def psnr(x_true, x_pred):
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true=x_true[:,:,:]
    for k in range(n_bands):
        x_true_k = x_true[  :, :,k].reshape([-1])
        x_pred_k = x_pred[  :, :,k,].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )


        MAX_k = np.max(x_true_k)
        if MAX_k != 0 :
            # PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            PSNR[k] = 10 * math.log10(math.pow(255, 2) / MSE[k])
            #print ('P', PSNR[k])
        else:
            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    # print('psnr', psnr)
    # print('mse', mse)
    return psnr, mse

def sam(x_true,x_pre):
    buff1 = x_true*x_pre
    buff_sin = x_true[:,:,0]
    buff_sin1 = x_pre[:, :, 0]
    buff2 = np.sum(buff1, 2)
    buff2[buff2 == 0] = 2.2204e-16
    buff4 = np.sqrt(np.sum(x_true * x_true, 2))
    buff4[buff4 == 0] = 2.2204e-16
    buff5 = np.sqrt(np.sum(x_pre * x_pre, 2))
    buff5[buff5 == 0] = 2.2204e-16
    buff6 = buff2/buff4
    buff8 = buff6/buff5
    buff8[buff8 > 1] = 1
    buff10 = np.arccos(buff8)
    buff9 = np.mean(np.arccos(buff8))
    SAM = (buff9) * 180 / np.pi
    return SAM

def ssim(x_true,x_pre):

    num=x_true.shape[2]
    ssimm=np.zeros(num)
    c1=0.0001
    c2=0.0009
    n=0
    for x in range(x_true.shape[2]):
        z = np.reshape(x_pre[:, :,x], [-1])
        sa=np.reshape(x_true[:,:,x],[-1])
        y=[z,sa]
        cov=np.cov(y)
        oz=cov[0,0]
        osa=cov[1,1]
        ozsa=cov[0,1]
        ez=np.mean(z)
        esa=np.mean(sa)
        ssimm[n]=((2*ez*esa+c1)*(2*ozsa+c2))/((ez*ez+esa*esa+c1)*(oz+osa+c2))
        n=n+1
    SSIM=np.mean(ssimm)
    # print ('SSIM',SSIM)
    return SSIM
periodic_avg_dict = OrderedDict()

type_name_list = ['ICVL_LSA_5_EItrain_Transrandom_alpha1_st1_240324_220911']
for type_name in type_name_list:
    for epoch_num in range(10, 30, 10):
        parser = argparse.ArgumentParser(description="USD syn dataset")
        parser.add_argument("--cuda", action="store_true", help="use cuda?")
        parser.add_argument("--model", default="checkpoint/"+type_name+"/De_happy_model_epoch_"+str(epoch_num)+".pth", type=str, help="model path")
        parser.add_argument("--msfa_size", default=5, type=int, help="scale factor, Default: 4")
        parser.add_argument("--dataset", default="ICVL", type=str, help="NTIRE, ICVL")
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        opt = parser.parse_args()
        cuda = True
        save_flag = False
        show_img = False
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        print(opt.model)
        model = torch.load(opt.model)["model"]

        avg_psnr_predicted = 0.0
        avg_sam_predicted = 0.0
        avg_ssim_predicted = 0.0
        avg_ergas_predicted = 0.0
        avg_sei = 0.0
        avg_elapsed_time = 0.0
        sample_num = 0
        if opt.dataset == 'ICVL':
            testimg_path = '/data1/fengkai/dataset/ICVL/IMEC25_600/test/'
            opt.ext = '.tif'
            save_path = 'results/syn/'
        elif opt.dataset == 'NTIRE':
            testimg_path = '/data2/fengkai/dataset/NRITE/valid_spectral_16/'
            opt.ext = '.mat'
            save_path = 'results/syn/'

        name_list = get_filename(testimg_path, opt.ext)
        with torch.no_grad():
            for image_name in name_list:
                    print("Processing ", image_name)
                    sample_num = sample_num + 1
                    image_name = image_name.split('.', 1)[0]
                    if opt.dataset == 'ICVL':
                        im_gt_y = load_img(testimg_path + image_name + opt.ext)
                    elif opt.dataset == 'NTIRE':
                        im_gt_y = loadCube(testimg_path + image_name + opt.ext)[0].astype(np.float32)
                    im_gt_y = im_gt_y[0:(im_gt_y.shape[0]//opt.msfa_size)*opt.msfa_size, 0:(im_gt_y.shape[1]//opt.msfa_size)*opt.msfa_size, :]
                    max_new = np.max(im_gt_y)
                    im_gt_y = im_gt_y / max_new * 255
                    im_gt_y = im_gt_y.transpose(1, 0, 2)

                    im_l_y = mask_input(im_gt_y, opt.msfa_size)
                    im_gt_y = im_gt_y.astype(float)
                    im_l_y = im_l_y.astype(float)

                    im_input = im_l_y / 255.


                    im_gt_y = im_gt_y.transpose(2, 0, 1)
                    im_l_y = im_l_y.transpose(2, 0, 1)
                    im_input = im_input.transpose(2, 0, 1)
                    raw = im_input.sum(axis=0)
                    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])
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
                        im_input = im_input.cpu()
                        raw = raw.cpu()
                        scale_coord_map = scale_coord_map.cpu()

                    start_time = time.time()
                    HR_4x = model([im_input, raw], scale_coord_map)
                    HR_4x = HR_4x[..., :h, :w]
                    elapsed_time = time.time() - start_time

                    HR_4x = HR_4x.cpu()
                    im_h_y = HR_4x.data[0].numpy().astype(np.float32)

                    cube_pdc_var = np.ones((1, opt.msfa_size ** 2))
                    for bn in range(opt.msfa_size ** 2):
                        singleband_pdc_avg = pixel_shuffle_inv(np.expand_dims((np.expand_dims(im_h_y[bn, :, :], 0)), 0),
                                                               opt.msfa_size)
                        singleband_pdc_avg = np.mean(np.mean(singleband_pdc_avg, -1), -1)
                        singleband_pdc_avg = singleband_pdc_avg.var(axis=1)
                        cube_pdc_var[0, bn] = singleband_pdc_avg
                    cube_pdc_var_avg = np.mean(cube_pdc_var, 1)
                    print('SEI_singleimage=', cube_pdc_var_avg)

                    im_h_y = im_h_y * 255.
                    im_h_y = np.rint(im_h_y)
                    im_h_y[im_h_y < 0] = 0
                    im_h_y[im_h_y > 255.] = 255.
                    im_h_y = im_h_y.astype(np.uint8)
                    im_h_y = im_h_y.astype(np.float)

                    raw = raw.cpu()
                    raw = raw.data[0].numpy().astype(np.float32)
                    raw = raw * 255.
                    raw[raw < 0] = 0
                    raw[raw > 255.] = 255.

                    im_input = im_input.cpu()
                    im_input = im_input.data[0].numpy().astype(np.float32)
                    im_input = im_input * 255.
                    im_input[im_input < 0] = 0
                    im_input[im_input > 255.] = 255.

                    im_gt_y = im_gt_y.astype(np.uint8)
                    im_gt_y = im_gt_y.astype(np.float)
                    [psnr_predicted, mse] = psnr(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
                    print("PSNR_singleimage=", psnr_predicted)
                    ssim_predicted = ssim(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
                    sam_predicted = sam(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
                    ergas_predicted = ergas_matlab(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))

                    if save_flag:
                        tiff = TIFFimage(im_h_y.astype(np.uint8), description='')
                        tiff.write_file((save_path + image_name + '_' + type_name + epoch_num + '.tif'),
                                        compression='none')
                        del tiff  # flushes data to disk

                    avg_psnr_predicted += psnr_predicted
                    avg_sam_predicted += sam_predicted
                    avg_ssim_predicted += ssim_predicted
                    avg_ergas_predicted += ergas_predicted
                    avg_sei += cube_pdc_var_avg
                    avg_elapsed_time += elapsed_time

                    if show_img:
                        nband = 12
                        fig = plt.figure()
                        ax = plt.subplot(221)
                        # ax.imshow(im_gt_y[nband, :, :], cmap='gray')
                        if opt.msfa_size == 5:
                            buff = np.concatenate((im_gt_y[22:23, :, :], im_gt_y[12:13, :, :], im_gt_y[4:5, :, :])).transpose(1, 2, 0)
                        elif opt.msfa_size == 4:
                            buff = np.concatenate((im_gt_y[0:1, :, :], im_gt_y[7:8, :, :], im_gt_y[14:15, :, :])).transpose(1, 2, 0)
                        ax.imshow(buff.astype(np.uint8))
                        ax.set_title("GT")

                        ax = plt.subplot(222)
                        ax.imshow(im_input[nband, :, :], cmap='gray')
                        ax.set_title("Input(one band)")

                        ax = plt.subplot(223)
                        ax.imshow(raw[0, :, :], cmap='gray')
                        ax.set_title("Input(raw)")

                        ax = plt.subplot(224)
                        if opt.msfa_size == 5:
                            buff = np.concatenate((im_h_y[22:23, :, :], im_h_y[12:13, :, :], im_h_y[4:5, :, :])).transpose(1, 2, 0)
                        elif opt.msfa_size == 4:
                            buff = np.concatenate((im_h_y[0:1, :, :], im_h_y[7:8, :, :], im_h_y[14:15, :, :])).transpose(1, 2, 0)
                        ax.imshow(buff.astype(np.uint8))
                        ax.set_title(opt.model)
                        plt.show()

                    del HR_4x
                    del raw
                    del im_input
            avg_psnr_predicted_save = avg_psnr_predicted / sample_num
            print("PSNR_predicted=", avg_psnr_predicted_save)
            print("SSIM_predicted=", avg_ssim_predicted / sample_num)
            print("SAM_predicted=", avg_sam_predicted / sample_num)
            print("ERGAS_predicted=", avg_ergas_predicted / sample_num)
            print("SEI_predicted=", avg_sei / sample_num)
            cube_pdc_var_avg2 = avg_sei / sample_num
            periodic_avg_dict[epoch_num] = np.concatenate((cube_pdc_var_avg2, np.array([avg_psnr_predicted_save])))
            data_frame = pd.DataFrame(
                data=periodic_avg_dict, index=range(1, 2 + 1))
            data_frame.to_csv((save_path + type_name + '.csv'), index_label='index')
            avg_psnr_predicted = 0
            avg_ssim_predicted = 0
            avg_sam_predicted = 0
            avg_ergas_predicted = 0
            avg_sei = 0
            sample_num = 0

print("Dataset=", opt.dataset)
print("It takes average {}s for processing".format(avg_elapsed_time/len(name_list)))
