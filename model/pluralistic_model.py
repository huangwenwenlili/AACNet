import torch
import torch.nn as nn
from .base_model import BaseModel
from . import network, base_function, external_function,aacnet

from util import task
import itertools
import math
from model.loss import AdversarialLoss, PerceptualLoss, StyleLoss
import torch.nn.functional as F


def weights_init(m, init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun(m)


class Pluralistic(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""

    def name(self):
        return "Pluralistic Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two',
                                help='training strategies with one path or two paths')
            parser.add_argument('--lambda_per', type=float, default=1, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_l1', type=float, default=1, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=0.1, help='weight for generation loss')
            parser.add_argument('--lambda_sty', type=float, default=250, help='weight for generation loss')
            parser.add_argument('--lambda_consist', type=float, default=1, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['app_g', 'ad_g', 'img_d', 'per', 'sty', 'consist']
        self.visual_names = ['img_m', 'img_truth', 'img_out', 'img_g']
        # self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['G', 'D']
        self.distribution = []
        self.istrans = False

        self.net_G = aacnet.define_g(gpu_ids=opt.gpu_ids)  # ada-RS-ab-B-4

        # define the discriminator model
        self.net_D = network.define_d_msg(gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define the loss functions
            self.GANloss = AdversarialLoss(type='nsgan')
            self.L1loss = torch.nn.L1Loss()
            self.per = PerceptualLoss()
            self.sty = StyleLoss()
            # define the optimizer

            self.optimizer_G = torch.optim.AdamW(
                itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.AdamW(
                itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.lr, betas=(0.5, 0.9))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1  # src
        # self.img_truth = self.img
        self.img_m = self.mask * self.img_truth
        # self.img_c = (1 - self.mask) * self.img_truth

        # get multiple scales image ground truth and mask for training
        # self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        # self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)

    def test(self):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        self.net_G.eval()
        if self.istrans:
            # trans
            self.img_g, self.consistency_features = self.net_G(self.img_m, self.mask)
        else:
            self.img_g, self.x_outs = self.net_G(self.img_m, self.mask)

        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask
        self.save_results(self.img_out, data_name='out')

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process

        if self.istrans:
            # trans
            self.img_g, self.consistency_features = self.net_G(self.img_m, self.mask)
        else:
            self.img_g, self.x_outs = self.net_G(self.img_m, self.mask)

        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        real_b_t = real
        x_outs_t = fake
        real_xs = {}
        real_xs['x_out_L32'] = F.interpolate(real_b_t, size=x_outs_t['x_out_L32'].size()[2:]).detach()
        real_xs['x_out_L64'] = F.interpolate(real_b_t, size=x_outs_t['x_out_L64'].size()[2:]).detach()
        real_xs['x_out_L128'] = F.interpolate(real_b_t, size=x_outs_t['x_out_L128'].size()[2:]).detach()
        real_xs['x_out_L256'] = F.interpolate(real_b_t, size=x_outs_t['x_out_L256'].size()[2:]).detach()

        x_outs_t_detach = {}
        for key, value in x_outs_t.items():
            x_outs_t_detach[key] = x_outs_t[key].detach()

        pred_real, _ = netD(real_xs)
        pred_fake, _ = netD(x_outs_t_detach)
        D_loss = (self.GANloss(pred_real, True, True) + self.GANloss(pred_fake, False, True)) / 2

        # # Real
        # D_real = netD(real)
        # #D_real_loss = self.GANloss(D_real, True, True)
        # # fake
        # D_fake = netD(fake.detach())
        # #D_fake_loss = self.GANloss(D_fake, False, True)
        # # loss for discriminator
        # D_loss = (self.GANloss(D_real, True, True) + self.GANloss(D_fake, False, True)) / 2

        D_loss.backward()
        # D_loss.backward(retain_graph=False) #default false, true

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.x_outs)
        # self.loss_img_d_rec = self.backward_D_basic(self.net_D_rec, self.img_truth, self.img_rec[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # generator adversarial loss
        base_function._freeze(self.net_D)

        # g loss fake
        D_fake, features = self.net_D(self.x_outs)

        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        # rec loss fake
        # D_fake = self.net_D_rec(self.img_rec[-1])
        # D_real = self.net_D_rec(self.img_truth)
        # self.loss_ad_rec = self.L2loss(D_fake, D_real) * self.opt.lambda_g

        # calculate l1 loss ofr multi-scale outputs
        totalG_loss = 0
        self.loss_app_g = self.L1loss(self.img_truth, self.img_g) * self.opt.lambda_l1
        total_lap_level_loss = 0
        for key, value in self.x_outs.items():
            # for w_l in range(self.w_level, 1, -1):
            if key != 'x_out_L256':
                t_fake_out = self.x_outs[key]

                # mask_re = F.interpolate(self.mask, size=t_fake_out.size()[2:])
                real_B_re = F.interpolate(self.img_truth, size=t_fake_out.size()[2:])
                # comp_B_re = t_fake_out * (1 - mask_re) + real_B_re * mask_re

                mult_l1_loss = self.L1loss(real_B_re, t_fake_out)
                # hole_loss = self.l1_loss(real_B_re, t_fake_out, (1 - mask_re))

                total_lap_level_loss += mult_l1_loss
        self.loss_app_g += total_lap_level_loss * self.opt.lambda_l1

        self.loss_per = self.per(self.img_g, self.img_truth) * self.opt.lambda_per
        self.loss_sty = self.sty(self.img_truth * (1 - self.mask), self.img_g * (1 - self.mask)) * self.opt.lambda_sty


        totalG_loss = self.loss_app_g + self.loss_per + self.loss_sty + self.loss_ad_g

        # totalG_loss.backward(retain_graph=True) #scale
        totalG_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()