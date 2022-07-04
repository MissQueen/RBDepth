import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from data.fifa_dataset import *
from data.nyu_dataset import *
from models.depth_net import *
import argparse
import numpy as np


class NYUEvaluation():
    def __init__(self, opt, e):
        test_set = nyuDataset(opt)
        batchsize = opt.batch_size
        dataloader = DataLoader(test_set, batchsize, shuffle=False)
        d_model = DepthNet().cuda()
        weights = torch.load(os.path.join(opt.d_model_path, str(e) + '_depthnet.pth'))
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        d_model.load_state_dict(weights_dict)

        # d_model.load_state_dict(torch.load(os.path.join(opt.d_model_path, str(e)+'_depthnet.pth')))
        d_model.eval()
        result = [[] for i in range(9)]
        self.test_name = []
        self.test_rmse = []
        for i, data in enumerate(dataloader):
            img, depth = data['src'], data['depth']
            depth = depth.float()
            img = torch.autograd.Variable(img).cuda()
            depth = torch.autograd.Variable(depth).cuda()
            pred_depth = d_model(img)
            pred_depth = torch.nn.functional.interpolate(pred_depth[-1],
                                                         size=[448, 560],
                                                         mode='bilinear',
                                                         align_corners=True)
            pred_depth = np.squeeze(pred_depth.cpu().detach().numpy())
            gt_np = np.squeeze(depth.cpu().detach().numpy())

            pred_depth = pred_depth*255.0
            pred_depth[pred_depth<0] = 0.
            pred_depth[pred_depth>255] = 255.
            # pred_depth = pred_depth/1000.0
            # gt_np = gt_np/1000.0
            # pred_depth *= 8.0
            min_depth = 1e-3
            max_depth = 10.0
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            # gt_np *= 8.0
            gt_np[gt_np < min_depth] = min_depth
            gt_np[gt_np > max_depth] = max_depth

            pred_depth /= 1000.0
            gt_np /= 1000.0
            print('pred_depth:',pred_depth)
            print('gt:', gt_np)

            test_result = self.compute_metrics(gt_np, pred_depth)
            self.test_name.append(data['img_name'][0])
            self.test_rmse.append(test_result[4])

            for it, item in enumerate(test_result):

                result[it].append(item)

        self.print_result(result)
        self.write2txt(self.test_name, self.test_rmse, opt, e)

    def write2txt(self,test_name, test_rmse, opt, e):
        if not os.path.exists(opt.test_out_path):
            os.makedirs(opt.test_out_path)
        txt = open(os.path.join(opt.test_out_path,str(e)+'_20220111_nyu-src-depth_rmse.txt'),'w')

        for i in range(len(test_rmse)):
            txt.write(test_name[i])
            txt.write(' ')
            txt.write(str(test_rmse[i]))
            txt.write('\n')
        txt.close()
        pass
    def compute_metrics(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean((np.abs(gt - pred) / gt))
        # print('abs_rel:',abs_rel)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        # print('sq_rel:',sq_rel)
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        # print('rmse:',rmse)
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        # print('rmse_log:',rmse_log)
        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        # print('silog:',silog)
        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        # print('log_10:',log_10)
        return [a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel]

    def print_result(self, result):
        a1 = np.array(result[0]).mean()
        a2 = np.array(result[1]).mean()
        a3 = np.array(result[2]).mean()
        abs_rel = np.array(result[3]).mean()
        rmse = np.array(result[4]).mean()
        log_10 = np.array(result[5]).mean()
        rmse_log = np.array(result[6]).mean()
        silog = np.array(result[7]).mean()
        sq_rel = np.array(result[8]).mean()
        print('a1:', a1, 'a2:', a2, 'a3:', a3, 'abs_rel:', abs_rel, 'rmse:', rmse, 'log_10:', log_10, 'rmse_log:',
              rmse_log, 'silog:', silog, 'sq_rel:', sq_rel)

class Evaluation():
    def __init__(self, opt, e):
        test_set = fifaDataset(opt)
        batchsize = opt.batch_size
        dataloader = DataLoader(test_set, batchsize, shuffle=False)
        d_model = DepthNet().cuda()
        d_model.load_state_dict(torch.load(os.path.join(opt.d_model_path, str(e)+'_depthnet.pth')))
        d_model.eval()
        result = [[] for i in range(9)]
        self.test_name = []
        self.test_rmse = []
        for i, data in enumerate(dataloader):
            img, depth = data['src'], data['depth']
            depth = depth.float()
            img = torch.autograd.Variable(img).cuda()
            depth = torch.autograd.Variable(depth).cuda()
            pred_depth = d_model(img)
            pred_depth = torch.nn.functional.interpolate(pred_depth[-1],
                                                         size=[352, 640],
                                                         mode='bilinear',
                                                         align_corners=True)
            pred_depth = np.squeeze(pred_depth.cpu().detach().numpy())
            gt_np = np.squeeze(depth.cpu().detach().numpy())

            # print('gt_np:', gt_np)
            # pred_depth += 1.0
            # pred_depth /= 2.0
            pred_depth *= 255.0
            min_depth = 1e-3
            max_depth = 255.0
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            gt_np[gt_np < min_depth] = min_depth
            # print('gt_np_min:',np.min(gt_np))
            # print('pre_depth_min:',np.min(pred_depth))
            # print('pred_depth:', pred_depth)
            test_result = self.compute_metrics(gt_np, pred_depth)
            self.test_name.append(data['img_name'][0])
            self.test_rmse.append(test_result[4])
            # print(self.test_rmse)
            for it, item in enumerate(test_result):
                # print('it:',it,'item:',item)
                result[it].append(item)
        # print(np.array(result).shape)
        self.print_result(result)
        self.write2txt(self.test_name, self.test_rmse, opt, e)

    def write2txt(self,test_name, test_rmse, opt, e):
        if not os.path.exists(opt.test_out_path):
            os.makedirs(opt.test_out_path)
        txt = open(os.path.join(opt.test_out_path,str(e)+'_1-7_non-reflectance_0.5d-1s-depth_rmse.txt'),'w')
        # print('******',np.array(test_rmse))
        for i in range(len(test_rmse)):
            txt.write(test_name[i])
            txt.write(' ')
            txt.write(str(test_rmse[i]))
            txt.write('\n')
        txt.close()
        pass
    def compute_metrics(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean((np.abs(gt - pred) / gt))
        # print('abs_rel:',abs_rel)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        # print('sq_rel:',sq_rel)
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        # print('rmse:',rmse)
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        # print('rmse_log:',rmse_log)
        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        # print('silog:',silog)
        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        # print('log_10:',log_10)
        return [a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel]

    def print_result(self, result):
        a1 = np.array(result[0]).mean()
        a2 = np.array(result[1]).mean()
        a3 = np.array(result[2]).mean()
        abs_rel = np.array(result[3]).mean()
        rmse = np.array(result[4]).mean()
        log_10 = np.array(result[5]).mean()
        rmse_log = np.array(result[6]).mean()
        silog = np.array(result[7]).mean()
        sq_rel = np.array(result[8]).mean()
        print('a1:', a1, 'a2:', a2, 'a3:', a3, 'abs_rel:', abs_rel, 'rmse:', rmse, 'log_10:', log_10, 'rmse_log:',
              rmse_log, 'silog:', silog, 'sq_rel:', sq_rel)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='NYU', help='path to dataset FIFA')
    parser.add_argument('--dataset_root', type=str, default='/data/home/lichuxuan/depth_estimation/nyu-files', help='path to dataset FIFA')
    parser.add_argument('--src_fifa_dataset', type=str, default='/disk1/lcx/fifa', help='path to dataset FIFA')
    parser.add_argument('--reflectance_fifa_dataset', type=str,
                        default='/disk1/lcx/fifa_datasets/12_21_r_640_360/reflectance', help='path to dataset FIFA')
    parser.add_argument('--edge_fifa_dataset', type=str, default='/disk1/lcx/fifa_datasets/12-25_edge_640_360',
                        help='path to edge dataset FIFA')
    parser.add_argument('--src_nyu_dataset', type=str, default='/data/home/lichuxuan/depth_estimation/data')
    parser.add_argument('--reflectance_nyu_dataset', type=str, default='')
    parser.add_argument('--edge_nyu_dataset', type=str, default='/data/home/lichuxuan/depth_estimation/nyu-files/edge')

    # parser.add_argument('--dataset_root', type=str, default='/disk1/lcx/ppt_result', help='path to dataset FIFA')
    parser.add_argument('--device', type=str, default='cuda', help='deivce')
    parser.add_argument('--isTrain', type=bool, default=False, help='phase')
    # parser.add_argument('--isTrain', default='True', action='store_true', help='train or test')
    parser.add_argument('--checkpoints_dir', type=str, default='/data/home/lichuxuan/depth_estimation/nyu_dataset/nyu-model',
                        help='Models are saved here')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--epochs', type=int, default=120, help='epochs of training')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_R_gradient', type=float, default=10.,
                        help='weight for reflectance gradient loss')
    parser.add_argument('--lambda_I_L2', type=float, default=1., help='weight for illumination L2 loss')
    parser.add_argument('--lambda_I_smooth', type=float, default=10, help='weight for Illumination smooth loss')
    parser.add_argument('--lambda_ifm', type=float, default=100, help='weight for pm loss')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--n_downsample', type=int, default=2, help='min 2')
    parser.add_argument('--r_enc_n_res', type=int, default=4, help='reflectance encoder resblock layers')
    parser.add_argument('--r_dec_n_res', type=int, default=0, help='reflectance decoder resblock layers')
    parser.add_argument('--i_enc_n_res', type=int, default=0, help='illumination encoder resblock layers')
    parser.add_argument('--i_dec_n_res', type=int, default=0, help='illumination decoder resblock layers')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--pad_type', type=str, default='reflect', help='pad_type')
    parser.add_argument('--activ', type=str, default='lrelu', help='activ')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--display_ncols', type=int, default=6,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

    # network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # training parameters
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--g_lr_ratio', type=float, default=1.0, help='a ratio for changing learning rate of generator')
    parser.add_argument('--d_lr_ratio', type=float, default=1.0,
                        help='a ratio for changing learning rate of discriminator')
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=40,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--save_iter_model', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--train_mode', type=str, default='depth', help='reflectance | depth')
    parser.add_argument('--r_model_path', type=str, default='/disk1/lcx/model_pth/cusdepth/12-14-modelpath/reflectance',
                        help='path to reflectance model')
    parser.add_argument('--d_model_path', type=str, default='/data/home/lichuxuan/depth_estimation/nyu_dataset/nyu-model/depth',
                        help='path to depthnet model')
    parser.add_argument('--test_file_path', type=str, default='/data/home/lichuxuan/depth_estimation/nyu_dataset/nyu-test-ppt/',
                        help='path to depthnet model')
    parser.add_argument('--test_out_path', type=str, default='/data/home/lichuxuan/depth_estimation/nyu_dataset/nyu-test-result/', help='path to depthnet model')

    opt = parser.parse_args()
    # for i in range(4,12):
    #     print('----epoch----',i*10)
    #     opt.epochs = i*10
    #     Evaluation(opt, i*10)
    opt.epochs = 10
    NYUEvaluation(opt, 10)
