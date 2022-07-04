import torch
import os
# import itertools
# import torch.nn.functional as F
# from util import distributed as du
from util import util
# from . import harmony_networks as networks
import torch.nn as nn
import util.ssim as ssim
from .harmony_networks import *
from .depth_net import *

class retinexltifpmModel(nn.Module):
    def __init__(self, opt, d_model):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(retinexltifpmModel,self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.d_model = d_model
        # self.loss_names = ['G','G_L1','G_R_grident','G_I_L2','G_I_smooth',"IF"]
        # self.loss_names = ['G', 'G_L1', 'G_R_grident', 'G_I_smooth']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['harmonized', 'src_img', 'reflectance', 'illumination']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # self.model_names = ['G']
        # self.opt.device = self.device


        # self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = opt.r_enc_n_res
        r_dec_n_res = opt.r_dec_n_res
        i_enc_n_res = opt.i_enc_n_res
        i_dec_n_res = opt.i_dec_n_res

        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc,
                                              self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res,
                                              self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ,
                                              pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res,
                                               opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ,
                                               pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res,
                                               self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln',
                                               opt.activ, pad_type=opt.pad_type)

        self.cur_device = torch.cuda.current_device()
        # self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        # if self.ismaster:
        #     print(self.netG)

        if self.isTrain:
            # if self.ismaster == 0:
            #     util.saveprint(self.opt, 'netG', str(self.netG))
            #     # define loss functions
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            self.criterionDSSIM_CS = ssim.DSSIM(mode='c_s').to(self.device)

            self.optimizer_G = torch.optim.Adam(params=list(self.reflectance_enc.parameters())
                                                       +list(self.reflectance_dec.parameters())
                                                       +list(self.illumination_enc.parameters())
                                                       +list(self.illumination_dec.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        # self.comp = input['comp'].to(self.device)
        # self.real = input['real'].to(self.device)
        # self.inputs = input['inputs'].to(self.device)
        # self.mask = input['mask'].to(self.device)
        # self.image_paths = input['img_path']
        #
        # self.mask_r = F.interpolate(self.mask, size=[64,64])
        # self.mask_r_32 = F.interpolate(self.mask, size=[32,32])
        # self.real_r = F.interpolate(self.real, size=[32,32])
        # self.real_gray = util.rgbtogray(self.real_r)
        # self.src_img = input['src'].to(self.device)
        self.src_img = input['src'].cuda()
        # self.depth_img = input['depth'].to(self.device)
        self.depth_img = input['depth'].cuda()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.harmonized, self.reflectance, self.illumination, self.ifm_mean = self.netG(self.inputs, self.mask_r, self.mask_r_32)
        # if not self.isTrain:
        #     self.harmonized = self.comp*(1-self.mask) + self.harmonized*self.mask
        # self.harmonized, self.reflectance, self.illumination = self.netG(self.src_img)
        r_content = self.reflectance_enc(self.src_img)
        i_content = self.illumination_enc(self.src_img)

        # r_content = self.reflectanceRec(r_content, fg_mask=mask_r, attScore=match_score)

        self.reflectance = self.reflectance_dec(r_content)
        self.reflectance = self.reflectance / 2 + 0.5

        # i_content = self.lightingRes(i_content, fg_pooling, bg_pooling, mask_r)
        # i_content = self.illuminationRec(i_content, fg_mask=mask_r, attScore=match_score.detach())

        self.illumination = self.illumination_dec(i_content)
        self.illumination = self.illumination / 2 + 0.5

        self.harmonized = self.reflectance * self.illumination

        return self.harmonized, self.reflectance, self.illumination

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # self.loss_IF = self.criterionDSSIM_CS(self.ifm_mean, self.real_gray)*self.opt.lambda_ifm

        self.loss_G_L1 = self.criterionL1(self.harmonized, self.src_img) * self.opt.lambda_L1
        self.loss_G_R_grident = self.gradient_loss(self.reflectance, self.src_img) * self.opt.lambda_R_gradient
        self.loss_G_I_L2 = self.criterionL2(self.illumination, self.src_img)*self.opt.lambda_I_L2
        self.loss_G_I_smooth = util.compute_smooth_loss(self.illumination) * self.opt.lambda_I_smooth
        # assert 0
        self.loss_D_L1 = self.criterionL1(self.d_model(self.reflectance), self.depth_img) * self.opt.lambda_D_L1

        self.loss_G = self.loss_G_L1 + self.loss_G_R_grident + self.loss_G_I_smooth + self.loss_G_I_L2 + self.loss_D_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        return self.loss_G

    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x + g_y

    def save_pth(self,dir,e):
        torch.save(self.reflectance_enc.state_dict(), os.path.join(dir,str(e)+'_r_enc.pth'))
        torch.save(self.reflectance_dec.state_dict(), os.path.join(dir,str(e)+'_r_dec.pth'))
        torch.save(self.illumination_enc.state_dict(), os.path.join(dir,str(e)+'_i_enc.pth'))
        torch.save(self.illumination_dec.state_dict(), os.path.join(dir,str(e)+'_i_dec.pth'))




