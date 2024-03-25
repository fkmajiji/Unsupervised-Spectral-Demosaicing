import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from lapsrn import Mpattern_opt, L1_Charbonnier_mean_loss, reconstruction_loss, get_sparsecube_raw, Mpattern_opt_fast2, Mpattern_opt_newMCM
from tqdm import tqdm
import pandas as pd
from dataset_msimaker.data import get_training_set_opt, get_test_set_opt, get_real_mosaic_training_set_opt
from dataset_msimaker.dataset import NTIREDatasetFromFolder
from torch.utils.data import DataLoader
from math import log10
from Spectral_demosaicing import input_matrix_wpn as input_matrix_wpn_msfasize, get_datetime_str
import numpy
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description="unsupervised spectral demosaicing")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--msfa_size', '-uf',  type=int, default=5, help="the size of square msfa")
parser.add_argument("--dataset", default="ICVL", type=str, help="dataset. Can be NTIRE/ICVL/Mosaic25")
parser.add_argument("--model", default="LSA", type=str, help="model. Can be LSA/HSA/SE/St")
parser.add_argument("--train_type", default="EItrain", type=str, help="EItrain or Suptrain or mixSuptrain")

def main() -> object:

    global opt, model
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    opt = parser.parse_args()
    opt.norm_flag = False
    opt.augment_flag = False
    opt.lr_drate = 0.5
    opt.alpha = 1
    opt.step_num = 1
    opt.dt = get_datetime_str()
    if opt.train_type == 'EItrain':
        if opt.dataset == 'ICVL':
            opt.batchSize = 1
            opt.nEpochs = 25000
            opt.save_interval = 10
        elif opt.dataset == 'NTIRE':
            opt.batchSize = 16
            opt.nEpochs = 8000
            opt.save_interval = 10
        elif opt.dataset == 'Mosaic25':
            opt.batchSize = 1
            opt.nEpochs = 14000
            opt.save_interval = 100

        opt.lr = 1e-4
        opt.step = 100000
        opt.tran_type = 'random' # can be globalshift / random
        opt.save_dir = 'checkpoint/'+opt.dataset+'_'+opt.model + '_' + str(opt.msfa_size) + '_' +opt.train_type+'_Trans' + opt.tran_type + '_alpha' + str(opt.alpha)+ '_st' + str(opt.step_num) + '_'+opt.dt + '/'
    elif opt.train_type == 'Suptrain':
        opt.batchSize = 32
        opt.nEpochs = 7500
        opt.lr = 2e-3
        if opt.dataset == 'ICVL':
            opt.step = 3000
            opt.nEpochs = 20000
            opt.save_interval = 50
        elif opt.dataset == 'NTIRE':
            opt.step = 1000
            opt.nEpochs = 1200
            opt.save_interval = 50

        opt.save_dir = 'checkpoint/' + opt.dataset + '_' + opt.model + '_' + str(opt.msfa_size) + '_' + opt.train_type + '_'+opt.dt +'/'

    cuda = True
    opt.cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    if opt.dataset == 'ICVL':
        opt.train_ps = 100
        opt.val_ps = 512
        train_set = get_training_set_opt("/data1/fengkai/dataset/ICVL/IMEC25_600/train",
                                         opt.msfa_size, opt.norm_flag, opt.augment_flag, opt.train_ps)
        test_set = get_test_set_opt("/data1/fengkai/dataset/ICVL/IMEC25_600/test",
                                    opt.msfa_size, opt.norm_flag, opt.val_ps)
    elif opt.dataset == 'NTIRE':
        opt.train_ps = 160
        opt.val_ps = 440
        train_set = NTIREDatasetFromFolder("/data2/fengkai/dataset/NRITE/train_spectral_16",
                                         opt.msfa_size, opt.norm_flag, opt.train_ps, opt.augment_flag)
        test_set = NTIREDatasetFromFolder("/data2/fengkai/dataset/NRITE/valid_spectral_16",
                                    opt.msfa_size, opt.norm_flag, opt.val_ps)
    elif opt.dataset == 'Mosaic25':
        opt.train_ps = 100
        opt.val_ps = 500
        train_set = get_real_mosaic_training_set_opt("/data2/fk/dataset/25bands/static_zoom/train/buff", opt.msfa_size,
                                                     opt.norm_flag, patch_size=opt.train_ps)
        test_set = get_test_set_opt("/data2/fk/dataset/TT59/TT59_600_875_nossfnorm_01norm/test_small",
                                    opt.msfa_size, opt.norm_flag, opt.val_ps)

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)


    print(opt)

    print("===> Building model")
    if opt.model == 'HSA':
        model = Mpattern_opt(opt.msfa_size, 'HSA')
    elif opt.model == 'LSA':
        model = Mpattern_opt(opt.msfa_size, 'LSA')
    elif opt.model == 'SE':
        model = Mpattern_opt(opt.msfa_size, 'SE')
    elif opt.model == 'St':
        model = Mpattern_opt(opt.msfa_size, 'None')
    elif opt.model == 'fastHSA':
        # model = Mpattern_opt_fast2(msfa_size=5, att_type='HSA')
        model = Mpattern_opt_newMCM(msfa_size=opt.msfa_size, att_type='HSA', conv_type='st', inC=1)
    elif opt.model == 'fastLSA':
        # model = Mpattern_opt_fast2(msfa_size=5, att_type='LSA')
        model = Mpattern_opt_newMCM(msfa_size=opt.msfa_size, att_type='LSA', conv_type='st', inC=1)
    elif opt.model == 'fastSt':
        model = Mpattern_opt_fast2(msfa_size=5, att_type='None')

    criterion = L1_Charbonnier_mean_loss()
    criterion1 = reconstruction_loss(opt.msfa_size)
    print(model)
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()
    else:
        model = model.cpu()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    save_opt(opt)
    print("===> Training")
    results = {'im_loss': [], 're_loss': [], 'all_loss': [], 'psnr': []}
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # if epoch % 50 == 1:
        test_results = test(testing_data_loader, optimizer, model, criterion, criterion1, epoch, opt.nEpochs, opt.msfa_size)
        if opt.train_type == 'EItrain':
            running_results = train_EI_optstep(training_data_loader, optimizer, model, criterion, criterion1, epoch, opt.nEpochs, opt.msfa_size, opt)
        elif opt.train_type == 'Suptrain':
            running_results = train_sup(training_data_loader, optimizer, model, criterion, criterion1, epoch, opt.nEpochs, opt.msfa_size)

        results['im_loss'].append(running_results['im_loss'] / running_results['batch_sizes'])
        results['re_loss'].append(running_results['re_loss'] / running_results['batch_sizes'])
        results['all_loss'].append(running_results['all_loss'] / running_results['batch_sizes'])
        results['psnr'].append(test_results['psnr'] / test_results['batch_sizes'] / opt.msfa_size ** 2)
        if epoch % opt.save_interval == 0:
            save_checkpoint(model, epoch)
        if epoch!=0:
            save_statistics(opt, results, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.lr_drate ** (epoch // opt.step))
    return lr

def train_EI_optstep(training_data_loader, optimizer, model, criterion, criterion1, epoch, num_epochs, msfa_size, opt):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'im_loss': 0, 're_loss': 0, 'all_loss': 0}
    model.train()

    for batch in train_bar:

        raw_syn, sparse_raw_syn, FD_pair = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        N, C, H, W = batch[0].size()
        running_results['batch_sizes'] += N

        scale_coord_map = input_matrix_wpn_msfasize(H, W, msfa_size)

        if opt.cuda:
            sparse_raw_syn = sparse_raw_syn.cuda()
            raw_syn = raw_syn.cuda()
            FD_pair = FD_pair.cuda()
            scale_coord_map = scale_coord_map.cuda()

        loss_x4 = 0
        # if 'fast' in opt.model:
        #     if opt.msfa_size == 5:
        #         mcm_ksize = opt.msfa_size + 2
        #     elif opt.msfa_size == 4:
        #         mcm_ksize = opt.msfa_size + 1
        #     pad_size = (mcm_ksize - 1) // 2
        #     raw_syn = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(raw_syn)
        #     raw_cube = raw_syn.repeat(1, opt.msfa_size ** 2, 1, 1)
        #     for i in range(opt.msfa_size):
        #         for j in range(opt.msfa_size):
        #             if i == 0 and j == 0:
        #                 continue
        #             else:
        #                 raw_single = torch.roll(raw_syn, (-i, -j), dims=(2, 3))
        #                 raw_cube[:, i * opt.msfa_size + j, :, :] = raw_single
        #     HR_4x = model([sparse_raw_syn, raw_cube], scale_coord_map)
        # else:
        firstcube = model([sparse_raw_syn, raw_syn], scale_coord_map)

        # flops, params = profile(model, inputs=([sparse_raw_syn, raw_syn], scale_coord_map,))
        # print('flops, params:', flops, params)

        loss_raw = criterion1(firstcube, FD_pair)
        loss = loss_raw.clone()

        for step_id in range(opt.step_num):
            if step_id == 0:
                curinput = firstcube
            else:
                curinput = curoutcube

            new_lable = transform_opt(curinput, msfa_size, opt)

            N, C, H, W = new_lable.size()
            scale_coord_map = input_matrix_wpn_msfasize(H, W, msfa_size)
            scale_coord_map = scale_coord_map.cuda()
            input, input_raw = get_sparsecube_raw(new_lable, opt.msfa_size)
            # if 'fast' in opt.model:
            #     if opt.msfa_size == 5:
            #         mcm_ksize = opt.msfa_size + 2
            #     elif opt.msfa_size == 4:
            #         mcm_ksize = opt.msfa_size + 1
            #     pad_size = (mcm_ksize - 1) // 2
            #     input_raw = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(input_raw)
            #     raw_cube = input_raw.repeat(1, opt.msfa_size ** 2, 1, 1)
            #     for i in range(opt.msfa_size):
            #         for j in range(opt.msfa_size):
            #             if i == 0 and j == 0:
            #                 continue
            #             else:
            #                 raw_single = torch.roll(input_raw, (-i, -j), dims=(2, 3))
            #                 raw_cube[:, i * opt.msfa_size + j, :, :] = raw_single
            #     FD_net = model([input, raw_cube], scale_coord_map)
            # else:
            curoutcube = model([input, input_raw], scale_coord_map)
            loss += opt.alpha/opt.step_num * criterion(curoutcube, new_lable)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running_results['im_loss'] += loss_x4.item()
        running_results['re_loss'] += loss_raw.item()
        running_results['all_loss'] += loss.item()
        train_bar.set_description(desc='[%d/%d] cube: %.4f L_real_raw: %.4f L_all: %.4f' % (
            epoch, num_epochs, running_results['im_loss'] / running_results['batch_sizes'],
            running_results['re_loss'] / running_results['batch_sizes'],
            running_results['all_loss'] / running_results['batch_sizes']))
    return running_results

def train_sup(training_data_loader, optimizer, model, criterion, criterion1, epoch, num_epochs, msfa_size):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'im_loss': 0, 're_loss': 0, 'all_loss': 0}
    model.train()

    # for batch in training_data_loader:
    for batch in train_bar:
    # for batch_num, (input_raw, input, target) in enumerate(training_data_loader):

        input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        # batch_size = batch[0].shape[0]
        # running_results['batch_sizes'] += batch_size
        N, C, H, W = batch[0].size()
        # input_raw, input, label_x4 = input_raw.to(device_flag),input.to(device_flag), target.to(device_flag)
        # N, C, H, W  = target.shape
        running_results['batch_sizes'] += N

        scale_coord_map = input_matrix_wpn_msfasize(H, W, msfa_size)

        if opt.cuda:
            input = input.cuda()
            input_raw = input_raw.cuda()
            label_x4 = label_x4.cuda()
            scale_coord_map = scale_coord_map.cuda()

        # [Raw_conv,HR_4x] = model([input, input_raw], scale_coord_map)
        #
        # loss_raw = 0.125 * criterion1(Raw_conv, label_x4)
        # loss_x4 = 0.125 * criterion(HR_4x, label_x4)
        # loss = loss_x4 + loss_raw
        # optimizer.zero_grad()
        #
        # loss_raw.backward(retain_graph=True)
        # loss_x4.backward()
        # # loss.backward()
        # optimizer.step()
        #
        # running_results['im_loss'] += loss_x4.item()
        # running_results['re_loss'] += loss_raw.item()
        # running_results['all_loss'] += loss.item()
        if 'fast' in opt.model:
            if opt.msfa_size == 5:
                mcm_ksize = opt.msfa_size + 2
            elif opt.msfa_size == 4:
                mcm_ksize = opt.msfa_size + 1
            pad_size = (mcm_ksize - 1) // 2
            input_raw = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(input_raw)
            raw_cube = input_raw.repeat(1, opt.msfa_size ** 2, 1, 1)
            for i in range(opt.msfa_size):
                for j in range(opt.msfa_size):
                    if i == 0 and j == 0:
                        continue
                    else:
                        raw_single = torch.roll(input_raw, (-i, -j), dims=(2, 3))
                        raw_cube[:, i * opt.msfa_size + j, :, :] = raw_single
            HR_4x = model([input, raw_cube], scale_coord_map)
        else:
            HR_4x = model([input, input_raw], scale_coord_map)

        loss_x4 = criterion(HR_4x, label_x4)
        loss_raw = 0
        loss = loss_x4
        optimizer.zero_grad()

        # loss_x2.backward(retain_graph=True)
        # loss_x4.backward()

        loss.backward()
        optimizer.step()

        running_results['im_loss'] += loss_x4.item()
        # running_results['re_loss'] += loss_raw.item()
        running_results['all_loss'] += loss.item()
        train_bar.set_description(desc='[%d/%d] Loss_im: %.5f Loss_re: %.1f Loss_all: %.5f' % (
            epoch, num_epochs, running_results['im_loss'] / running_results['batch_sizes'],
            running_results['re_loss'] / running_results['batch_sizes'],
            running_results['all_loss'] / running_results['batch_sizes']))
        # progress_bar(batch_num, len(training_data_loader), 'Loss: %.4f' % (running_results['im_loss'] / running_results['batch_sizes']))
    return running_results

def test(testing_data_loader, optimizer, model, criterion, criterion1, epoch, num_epochs, msfa_size):

    # lr = adjust_learning_rate(optimizer, epoch-1)
    #
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr

    # print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 'psnr': 0, 'mse': 0}
    model.eval()

    with torch.no_grad():
        # for batch in training_data_loader:
        for batch in test_bar:
        # for batch_num, (input_raw, input, target) in enumerate(training_data_loader):

            input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            # batch_size = batch[0].shape[0]
            # running_results['batch_sizes'] += batch_size
            N, C, H, W = batch[0].size()
            # input_raw, input, label_x4 = input_raw.to(device_flag),input.to(device_flag), target.to(device_flag)
            # N, C, H, W  = target.shape
            test_results['batch_sizes'] += N

            scale_coord_map = input_matrix_wpn_msfasize(H, W, msfa_size)

            if opt.cuda:
                input = input.cuda()
                input_raw = input_raw.cuda()
                label_x4 = label_x4.cuda()
                scale_coord_map = scale_coord_map.cuda()

            # input_raw = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16,17,18,19,20],[21,22,23,24,25]]).cuda()
            # input_raw = input_raw.unsqueeze(0)
            # input_raw = input_raw.unsqueeze(0)
            # input_raw = input_raw.repeat(1, 1, 3, 3)

            # if 'fast' in opt.model:
            #
            #     if opt.msfa_size == 5:
            #         mcm_ksize = opt.msfa_size + 2
            #     elif opt.msfa_size == 4:
            #         mcm_ksize = opt.msfa_size + 1
            #     pad_size = (mcm_ksize - 1) // 2
            #     input_raw = nn.ZeroPad2d((pad_size, pad_size, pad_size, pad_size))(input_raw)
            #     raw_cube = input_raw.repeat(1, opt.msfa_size ** 2, 1, 1)
            #     for i in range(opt.msfa_size):
            #         for j in range(opt.msfa_size):
            #             if i == 0 and j == 0:
            #                 continue
            #             else:
            #                 raw_single = torch.roll(input_raw, (-i, -j), dims=(2, 3))
            #                 raw_cube[:, i * opt.msfa_size + j, :, :] = raw_single
            #     HR_4x = model([input, raw_cube], scale_coord_map)
            # else:

            HR_4x = model([input, input_raw], scale_coord_map)

            for nbatch in range(N):
                for nb in range(msfa_size**2):
                    sb_mse = ((HR_4x[nbatch, nb, :, :] - label_x4[nbatch, nb, :, :]) ** 2).data.mean()
                    psnr = 10 * log10(1 / sb_mse.item())
                    test_results['psnr'] += psnr

            test_bar.set_description(desc='[%d/%d] psnr: %.4f ' % (
            epoch, num_epochs, test_results['psnr']/test_results['batch_sizes']/(msfa_size**2)))
    return test_results

def transform_opt(HR_4x, msfa_size, opt):
    if opt.tran_type == 'random':
        tran_type = random.choice([0, 1, 3, 5])
    elif opt.tran_type == 'rotation':
        tran_type = 0
    elif opt.tran_type == 'flip':
        tran_type = 1
    elif opt.tran_type == 'resize':
        tran_type = 3
    elif opt.tran_type == 'globalshift':
        tran_type = 4
    elif opt.tran_type == 'patternshift':
        tran_type = 5
    else:
        raise Exception("wrong tran type")

    if tran_type == 0:
        ## rotation transf
        rn = random.randint(1, 3)
        new_lable = torch.rot90(HR_4x, rn, [2, 3])

    if tran_type == 1:
        if numpy.random.uniform() < 0.5:
            new_lable = torch.flip(HR_4x, [2])
        else:
            new_lable = torch.flip(HR_4x, [3])

    scale_lib = [0.2, 0.25, 0.5, 2, 3, 4]

    if tran_type == 3:
        ## resize transf
        while 1:
            scale_num = random.randint(0, len(scale_lib)-1)
            new_lable = torch.nn.functional.interpolate(HR_4x, scale_factor=scale_lib[scale_num], mode='nearest')
            N, C, H, W = new_lable.size()
            if H >= (opt.train_ps-opt.msfa_size)*scale_lib[0] and H <= opt.train_ps*scale_lib[-1]:  # opt.train_ps*scale_lib[0]*0.5: # value of ICVL_LSA_5_EItrain_Transrandomnew_upd7_alpha1_st1_230721_150236 maybe 40 or opt.train_ps*scale_lib[0]
                break
        new_lable = random_crop_4DTensor(new_lable, (H // opt.msfa_size) * opt.msfa_size)

    if tran_type == 4:
        ## golbal shift transf used in EI
        new_lable = shift_random(HR_4x, n_trans=1, max_offset=0)

    if tran_type == 5:
        ## new shift have just (msfa_size-1)*(msfa_size-1) transformations
        while 1:
            i = random.randint(0, msfa_size-1)
            j = random.randint(0, msfa_size-1)
            if i != 0 or j != 0:
                break
        # print('newshift', i, j)
        new_lable = torch.roll(HR_4x, (-i, -j), (2, 3))
        N, C, H, W = new_lable.size()
        new_lable = new_lable[:, :, 0: (H - opt.msfa_size), 0: (W - opt.msfa_size)]

    return new_lable

def random_crop_4DTensor(a, crop_size):
    N, C, hei, wid=a.size()
    Height = random.randint(0, hei - crop_size)
    Width = random.randint(0, wid - crop_size)
    return a[:, :, Height:(Height + crop_size), Width:(Width + crop_size)]

def shift_random(x, n_trans=5, max_offset=0):
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - 1, 'n_shifts should less than {}'.format(H-1)

    if max_offset==0:
        shifts_row = random.sample(list(np.concatenate([-1*np.arange(1, H), np.arange(1, H)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1*np.arange(1, W), np.arange(1, W)])), n_trans)
    else:
        assert max_offset<=min(H,W), 'max_offset must be less than min(H,W)'
        shifts_row = random.sample(list(np.concatenate([-1*np.arange(1, max_offset), np.arange(1, max_offset)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1*np.arange(1, max_offset), np.arange(1, max_offset)])), n_trans)

    x = torch.cat([x if n_trans == 0 else torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) for sx, sy in
                   zip(shifts_row, shifts_col)], dim=0)
    return x


def save_checkpoint(model, epoch):
    model_folder = opt.save_dir
    model_out_path = model_folder + "De_happy_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_opt(opt):
    statistics_folder = opt.save_dir
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=vars(opt), index=range(1, 2))
    # data_frame = pd.DataFrame(
    #     data={'Loss_im': results['im_loss'], 'Loss_re': results['re_loss'], 'Loss_all': results['all_loss']},
    #     index=range(opt.start_epoch, epoch+1))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_opt.csv', index_label='Epoch')
    print("save--opt")

def save_statistics(opt, results, epoch):
    statistics_folder = opt.save_dir
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=results,index=range(opt.start_epoch, epoch + 1))
    # data_frame = pd.DataFrame(
    #     data={'Loss_im': results['im_loss'], 'Loss_re': results['re_loss'], 'Loss_all': results['all_loss']},
    #     index=range(opt.start_epoch, epoch+1))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_train_results.csv', index_label='Epoch')
    # print("saveLoss")
if __name__ == "__main__":
    main()
