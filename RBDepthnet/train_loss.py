# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim as _ssim
import numpy as np



class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv=nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_kx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edge_ky=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k=np.stack((edge_kx, edge_ky))

        edge_k=torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight=nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x):
        out=self.edge_conv(x)
        out=out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

def depth_loss(output, depth_gt, get_gradient):
    depth_loss = torch.nn.L1Loss()
    losses = []
    ssim_loss = []
    grad_loss = []
    for depth_index in range(len(output)):
        loss = depth_loss(output[depth_index], depth_gt[depth_index])
        losses.append(loss)
        ssim_loss.append(torch.clip((1 - _ssim.ssim(depth_gt[depth_index], output[depth_index])) * 0.5, 0.0, 1.0))
        depth_t = torch.from_numpy(np.array(depth_gt[depth_index]))
        depth_o = torch.from_numpy(np.array(output[depth_index]))

        depth_grad = get_gradient(depth_t)
        output_grad = get_gradient(depth_o)

        depth_grad_dx = depth_grad[:,0,:,:].contiguous().view_as(depth_t)
        depth_grad_dy = depth_grad[:,1,:,:].contiguous().view_as(depth_t)
        output_grad_dx = output_grad[:,0,:,:].contiguous().view_as(depth_t)
        output_grad_dy = output_grad[:,1,:,:].contiguous().view_as(depth_t)
        loss_dx = torch.log(torch.abs(output_grad_dx-depth_grad_dx)+0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy-depth_grad_dy)+0.5).mean()
        grad_loss.append(loss_dx+loss_dy)

    a1 = 1
    a2 = 1
    a3 = 1

    loss_d = sum(losses)
    loss_s = sum(ssim_loss)
    loss_g = sum(grad_loss)
    print('loss_d:', loss_d.item(), 'loss_s:', loss_s.item(), 'loss_g:', loss_g.item())
    total_loss = a1 * loss_d + a2 * loss_s + a3 * loss_g
    return total_loss

