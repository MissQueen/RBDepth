import argparse
import os
import sys
import torchvision.transforms.functional as tf
import cv2
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import numpy as np
from models.model import *
import torch.nn.functional as F
from fifa_dataset import *
from nyu_dataset import *
from train_loss import *

def adjust_gt(gt_depth, pred_depth):
	adjusted_gt = []
	for each_depth in pred_depth:
		adjusted_gt.append(F.interpolate(gt_depth, size=[each_depth.size(2), each_depth.size(3)],
								   mode='bilinear', align_corners=True))
	return adjusted_gt

def is_rank_zero(args):
    return args.rank == 0

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def validate(opt, d_model, te_set, e, device):
    with torch.no_grad():
        metrics = RunningAverageDict()
        for batch in tqdm(te_set, desc=f"Epoch:{e}.Loop:Validation") if is_rank_zero(opt) else te_set:
            img = batch['src'].to(device)
            depth = batch['depth'].cpu()
            depth = np.array(depth)
            depth = np.squeeze(depth)
            pred_depth = d_model(img)
            pred_depth = torch.nn.functional.interpolate(pred_depth[-1],
                                                         size=[opt.input_height, opt.input_width],
                                                         mode='bilinear',
                                                         align_corners=True)
            min_val = 1e-3
            max_val = 10.0
            pred_depth = np.squeeze(pred_depth.cpu().detach().numpy())
            pred_depth[pred_depth < min_val] = min_val
            pred_depth[pred_depth > max_val] = max_val
            valid_mask = np.logical_and(depth > min_val, depth < max_val)
            if opt.dataset == 'NYU':
                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(compute_errors(depth[valid_mask], pred_depth[valid_mask]))
            return metrics.get_value()


def main_worker(gpu, ngpus_per_node, opt):
    opt.gpu = gpu
    if opt.isTrain:
        d_model = model()
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            d_model = d_model.cuda(opt.gpu)
        opt.multigpu = False
        if opt.distributed:
            opt.multigpu = True
            opt.rank = opt.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                    world_size=opt.world_size, rank=opt.rank)
            opt.batch_size = int(opt.batch_size / ngpus_per_node)
            opt.workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            torch.cuda.set_device(opt.gpu)
            d_model = nn.SyncBatchNorm.convert_sync_batchnorm(d_model)
            d_model = d_model.cuda(opt.gpu)
            d_model = torch.nn.parallel.DistributedDataParallel(d_model, device_ids=[opt.gpu], output_device=opt.gpu,
                                                                find_unused_parameters=True)
        elif opt.gpu is None:
            opt.multigpu = True
            d_model = d_model.cuda()
            d_model = torch.nn.DataParallel(d_model)

        device = opt.gpu
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        should_write = ((not opt.distributed) or opt.rank == 0)

        if opt.dataset_name == 'FIFA':
            tr_set = FifaDataLoader(opt, mode='train').data
            te_set = FifaDataLoader(opt, mode='test').data
        else:
            tr_set = NyuDataLoader(opt, mode='train').data
            te_set = NyuDataLoader(opt, mode='test').data

        batchsize = opt.batch_size
        epoch = opt.epochs
        optimizer = torch.optim.Adam(d_model.parameters(), lr=opt.lr, betas=(0.9,0.999))
        d_model.train()

        get_gradient = Sobel().cuda()

        step = 0
        best_loss = np.inf

        for e in range(epoch):
            cur_loss = 0
            for i, data in tqdm(enumerate(tr_set), desc=f"Epoch:{e+1}/{epoch}.Loop:Train",
                                total=len(tr_set)) if is_rank_zero(opt) else enumerate(tr_set):
                src = data['src'].to(device)
                depth = data['depth'].to(device)

                pred_depth = d_model(src)
                gt_depth = adjust_gt(depth, pred_depth)
                depth_loss = depth_loss.depth_loss(pred_depth, gt_depth, get_gradient)
                cur_loss += depth_loss.cpu().item()
                optimizer.zero_grad()
                depth_loss.backward()
                optimizer.step()
                step += 1

                if should_write and step % opt.validate_every == 0:
                    d_model.eval()
                    if not os.path.exists(opt.checkpoints_dir):
                        os.makedirs(opt.checkpoints_dir)
                    metrics = validate(opt, d_model, te_set, e, device)
                    print('metrics:', metrics)
                    if metrics['abs_rel'] < best_loss:
                        torch.save(d_model.state_dict(), os.path.join(opt.checkpoints_dir,'best_depthnet.pth'))
                        best_loss = metrics['abs_rel']
                    d_model.train()
            cur_loss /= len(tr_set)
            print('epoch:', e, 'loss:', cur_loss)
            torch.save(d_model.state_dict(), os.path.join(opt.checkpoints_dir, str(e) + '_depthnet.pth'))
            print('depth model saved...')

pass


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FIFA', help='dataset name')
    parser.add_argument('--dataset_root', type=str, default='/disk1/lcx/cusdepth_640_360', help='path to dataset')
    parser.add_argument('--reflectace_dataset', type=str, default='', help='path to reflectance')
    parser.add_argument('--r_model_path', type=str,
                        default='/disk1/lcx/model_pth/nyu/src-reflectance-20220109/reflectance',
                        help='path to reflectance model')
    parser.add_argument('--d_model_path', type=str, default='/disk1/lcx/RBdepth-files/models/depth',
                        help='path to depthnet model')
    parser.add_argument('--test_file_path', type=str, default='/disk1/lcx/fifa_datasets/nyu-test/usi3d-test',
                        help='path to depthnet model')
    parser.add_argument('--test_out_path', type=str,
                        default='/disk1/lcx/fifa_datasets/nyu-test/ours-result/usi3d-result',
                        help='path to depthnet model')
    parser.add_argument('--checkpoints_dir', type=str, default='/disk1/lcx/RBdepth-files/models', help='path to save pth')
    parser.add_argument('--train_mode', type=str, default='depth', help='reflectance | depth')
##################################################################################
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--epoch', type=int, default=120, help='epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_R_gradient', type=float, default=10.0)
    parser.add_argument('--lambda_I_L2', type=float, default=1.0)
    parser.add_argument('--lambda_I_smooth', type=float, default=10.0)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--r_enc_n_res', type=int, default=4)
    parser.add_argument('--r_dec_n_res', type=int, default=0)
    parser.add_argument('--i_enc_n_res', type=int, default=0)
    parser.add_argument('--i_dec_n_res', type=int, default=0)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--pad_type', type=str, default='reflect')
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
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
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
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--input_height', type=int, help='input height 416 ｜ 352', default=352)
    parser.add_argument('--input_width', type=int, help='input width 544 ｜ 640', default=640)
    ###########################################################################################
    parser.add_argument('--validate-every', '--validate_every', default=500, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    opt = parser.parse_args()

    if sys.argv.__len__()  == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = parser.parse_args([arg_filename_with_prefix])
    else:
        opt = parser.parse_args()
    opt.num_threads = opt.workers
    opt.mode = 'train'

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[','').replace(']','')
        nodes = node_str.split(',')
        opt.world_size = len(nodes)
        opt.rank = int(os.environ['SLURM_PROCID'])
    except KeyError as e:
        opt.world_size = 1
        opt.rank = 0
        nodes = ['127.0.0.1']

    if opt.distributed:
        mp.set_start_method('forkserver')

        port = np.random.randint(15000, 15025)
        opt.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(opt.dist_url)
        opt.dist_backend = 'nccl'
        opt.gpu = None
    ngpus_per_node = torch.cuda.device_count()
    opt.num_workers = opt.workers
    opt.ngpus_per_node = ngpus_per_node
    if opt.distributed:
        opt.world_size = ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        if ngpus_per_node == 1:
            opt.gpu = 0

    main_worker(opt.gpu, ngpus_per_node, opt)