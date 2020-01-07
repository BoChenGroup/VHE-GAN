from __future__ import print_function

import scipy.special as ss

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p

from tensorboardX import summary
from tensorboardX import FileWriter

from model import G_NET, D_NET64, D_NET128, D_NET256, INCEPTION_V3, MLP_ENCODER_IMG, myLoss

from six.moves import xrange

from winPGBN_sampler import PGBN_sampler


# ################## Shared functions ###################
def negative_log_posterior_probability(predictions, num_splits=1):
    scores = []
    for i in xrange(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    scores = []
    for i in xrange(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    netEn_img = MLP_ENCODER_IMG()
    netEn_img.apply(weights_init)
    netEn_img = torch.nn.DataParallel(netEn_img, device_ids=gpus)
    print(netEn_img)

    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())

    for i in xrange(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in xrange(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    if cfg.TRAIN.NET_MLP_IMG != '':
        state_dict = torch.load(cfg.TRAIN.NET_MLP_IMG)
        netEn_img.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_MLP_IMG)

    inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        netEn_img = netEn_img.cuda()
        for i in xrange(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    return netG, netsD, netEn_img, inception_model, len(netsD), count


def define_optimizers(netG, netsD, netIMG):
    optimizersD = []
    num_Ds = len(netsD)
    for i in xrange(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerEnG = optim.Adam([{'params': netIMG.parameters()},
                               {'params': netG.parameters()}],
                              lr=cfg.TRAIN.GENERATOR_LR,
                              betas=(0.5, 0.999))

    return optimizersD, optimizerEnG


def save_model(netIMG, netG, avg_param_G, netsD, epoch, count, model_dir):
    torch.save(
        netIMG.state_dict(),
        '%s/netIMG_%d_%d.pth' % (model_dir, epoch, count))
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d_%d.pth' % (model_dir, epoch, count))
    for i in xrange(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save En/G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/count_%09dreal_samples.png' % (image_dir, count),
        normalize=True)

    for i in xrange(num_imgs):
        fake_img = fake_imgs[i][0:num]
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples_%s.png' %
            (image_dir, count, i), normalize=True)


# ################# Text to image task############################ #
class WHAI_GAN_Trainer(object):
    def __init__(self, output_dir, data_loader):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.log_dir = os.path.join(output_dir, 'Log')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.log_dir)
        self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        real_vimgs, wrong_vimgs = [], []
        imgs, texts, w_imgs, _ = data
        if cfg.CUDA:
            vtxts = Variable(texts).cuda()
        else:
            vtxts = Variable(texts)
        for i in xrange(3):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
        return imgs, vtxts, real_vimgs, wrong_vimgs

    def train_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_tgpu[0].size(0)
        criterion, mu = self.criterion_1, self.mu_theta1

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_tgpu[idx]
        wrong_imgs = self.wrong_tgpu[idx]
        fake_imgs = self.fake_imgs[idx]
        
        netD.zero_grad()
        
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())
        
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(fake_logits[1], fake_labels)
            
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            
            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward()
        # update parameters
        optD.step()
        # log
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, float(errD.data[0]))
            self.summary_writer.add_summary(summary_D, count)
        return float(errD)

    def train_Gnet(self, count):
        errG_total = 0
        flag = count % 100
        batch_size = self.real_tgpu[0].size(0)
        criterion_1, mu = self.criterion_1, self.mu_theta1

        params = [self.shape1, self.scale1, self.Phi1, self.theta1, self.txtbow]
        criterion, optEnG, netG = self.criterion_2, self.optimizerEnG, self.netG
        loss, theta1_KL, Likelihood, p1, p2, p3, shape1, scale1 = criterion(params)

        real_labels = self.real_labels[:batch_size]
        for i in xrange(self.num_Ds):
            outputs = self.netsD[i](self.fake_imgs[i], mu)
            errG = criterion_1(outputs[0], real_labels)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                             criterion_1(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG
            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_D, count)

        errG_total += loss
        optEnG.zero_grad()

        errG_total.backward()

        optEnG.step()
        return float(errG_total), theta1_KL, Likelihood, loss, p1, p2, p3, shape1, scale1

    def updatePhi(self, miniBatch, Phi, Theta, MBratio, MBObserved, NDot):
        Xt = miniBatch
        Xt_to_t1, WSZS = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), Phi.astype('double'), Theta.astype('double'))

        EWSZS = WSZS
        EWSZS = MBratio * EWSZS

        if (MBObserved == 0):
            NDot = EWSZS.sum(0)
        else:
            NDot = (1 - self.ForgetRate[MBObserved]) * NDot + self.ForgetRate[MBObserved] * EWSZS.sum(0)
        tmp = EWSZS + self.eta
        tmp = (1 / NDot) * (tmp - tmp.sum(0) * Phi)
        tmp1 = (2 / NDot) * Phi

        tmp = Phi + self.epsit[MBObserved] * tmp + np.sqrt(self.epsit[MBObserved] * tmp1) * np.random.randn(
            Phi.shape[0], Phi.shape[1])
        Phi = PGBN_sampler.ProjSimplexSpecial(tmp, Phi, 0)

        return Phi, NDot

    def train(self):
        self.netG, self.netsD, self.netIMG, self.inception_model,\
        self.num_Ds, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizersD, self.optimizerEnG = \
            define_optimizers(self.netG, self.netsD, self.netIMG)

        self.criterion_1 = nn.BCELoss()
        self.criterion_2 = myLoss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))

        # Prepare PHI
        real_min = np.float64(2.2e-308)
        Phi1 = 0.2 + 0.8 * np.float64(np.random.rand(1000, 256))
        Phi1 = Phi1 / np.maximum(real_min, Phi1.sum(0))
        if cfg.CUDA:
            self.Phi1 = Variable(Phi1).cuda()
            self.criterion_1.cuda()
            self.criterion_2.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // self.num_batches

        batch_length = self.num_batches
        self.NDot = 0
        self.ForgetRate = np.power((0 + np.linspace(1, cfg.TRAIN.MAX_EPOCH*int(batch_length),
                                    cfg.TRAIN.MAX_EPOCH * int(batch_length))), -0.7)
        self.eta = 0.1
        epsit = np.power((20 + np.linspace(1, cfg.TRAIN.MAX_EPOCH * int(batch_length),
                                    cfg.TRAIN.MAX_EPOCH * int(batch_length))), -0.7)
        self.epsit = 1 * epsit / epsit[0]

        num_total_samples = batch_length * self.batch_size

        start_t = time.time()

        for epoch in xrange(start_epoch, self.max_epoch):
            LL = 0
            KL = 0
            LS = 0
            p1 = 0
            p2 = 0
            p3 = 0
            DL = 0
            GL = 0
            shape = []
            scale = []
            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.img_tcpu, self.txtbow, self.real_tgpu, self.wrong_tgpu = self.prepare_data(data)

                #######################################################
                # (1) Get conv hidden units
                ######################################################
                _, self.flat = self.inception_model(self.real_tgpu[-1])

                #######################################################
                # (2) Get shape, scale and sample of theta
                ######################################################
                self.theta1, self.shape1, self.scale1 = self.netIMG(self.flat)

                shape1 = self.shape1.detach().cpu().numpy()
                scale1 = self.scale1.detach().cpu().numpy()
                mu_theta1 = scale1 * ss.gamma(1 + 1 / shape1)
                self.mu_theta1 = torch.tensor(mu_theta1, dtype=torch.float32)
                #######################################################
                # (3) Generate fake images
                ######################################################
                self.fake_imgs, _ = self.netG(self.mu_theta1.detach())

                #######################################################
                # (4) Update D network
                ######################################################
                errD_total = 0
                for i in xrange(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                #######################################################
                # (5) Update G network (or En network): maximize log(D(G(z)))
                ######################################################
                errG_total, self.KL1, self.LL, self.LS, self.p1, self.p2, self.p3, shape1, scale1 = self.train_Gnet(count)
                LL += self.LL
                KL += self.KL1
                LS += self.LS
                p1 += self.p1
                p2 += self.p2
                p3 += self.p3
                shape.append(shape1)
                scale.append(scale1)

                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                DL += errD_total
                GL += errG_total

                #######################################################
                # (6) Update Phi
                #######################################################
                input_txt = np.array(np.transpose(self.txtbow.cpu().numpy()), order='C').astype('double')
                Phi1 = np.array(self.Phi1.cpu().numpy(), order='C').astype('double')
                Theta1 = np.array(np.transpose(self.theta1.cpu().numpy()), order='C').astype('double')
                phi1, self.NDot = self.updatePhi(input_txt, Phi1, Theta1, int(batch_length), count, self.NDot)
                self.Phi1 = torch.tensor(phi1, dtype=torch.float32).cuda()

                # for inception score
                pred, _ = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total)
                    summary_G = summary.scalar('G_loss', errG_total)
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netIMG, self.netG, avg_param_G, self.netsD, epoch, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)

                    self.fake_imgs, _ = \
                        self.netG(self.theta1)

                    save_img_results(self.img_tcpu, self.fake_imgs, self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        m_incep = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(m_incep, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(m_nlpp, count)
                        #
                        predictions = []
                count += 1

            end_t = time.time()
            LS = LS / num_total_samples
            LL = LL / num_total_samples
            KL = KL / num_total_samples
            DL = DL / num_total_samples
            GL = GL / num_total_samples
            print('Epoch: %d/%d,   Time elapsed: %.4fs\n'
                  '* Batch Train Loss: %.6f          (LL: %.6f, KL: %.6f, Loss_D:'
                  '%.2f Loss_G: %.2f)\n' % (epoch, self.max_epoch, end_t - start_t,
                                            LS, LL, KL, DL, GL))
            start_t = time.time()
            if epoch % 50 == 0:
                save_model(self.netIMG, self.netG, avg_param_G, self.netsD, epoch, count, self.model_dir)
        # save the model at the last updating
        save_model(self.netIMG, self.netG, avg_param_G, self.netsD, epoch, count, self.model_dir)
        self.summary_writer.close()