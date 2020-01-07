from __future__ import print_function

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

from model import G_NET_256, G_NET_128, G_NET_64, D_THETA3_NET16, D_THETA3_NET32,\
    D_THETA3_NET64, D_THETA2_NET32, D_THETA2_NET64, D_THETA2_NET128, D_THETA1_NET64,\
    D_THETA1_NET128, D_THETA1_NET256, INCEPTION_V3, MLP_ENCODER, myLoss

from six.moves import xrange
from winPGBN_sampler import PGBN_sampler


# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    img_hat_transpose = img_hat.transpose(1, 2)
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


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


def load_network(gpus):
    netEn = MLP_ENCODER(cfg.TEXT.DIMENSION_THETA1, cfg.TEXT.DIMENSION_THETA2, cfg.TEXT.DIMENSION_THETA3)
    netEn.apply(weights_init)
    netEn = torch.nn.DataParallel(netEn, device_ids=gpus)
    print(netEn)

    if cfg.TRAIN.NET_MLP_EN != '':
        state_dict = torch.load(cfg.TRAIN.NET_MLP_EN)
        netEn.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_MLP_EN)

    netsG = []
    netsG.append(G_NET_64())
    netsG.append(G_NET_128())
    netsG.append(G_NET_256())
    for i in xrange(len(netsG)):
        netsG[i].apply(weights_init)
        netsG[i] = torch.nn.DataParallel(netsG[i], device_ids=gpus)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_THETA3_NET16(cfg.TEXT.DIMENSION_THETA3))
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_THETA3_NET32(cfg.TEXT.DIMENSION_THETA3))
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_THETA3_NET64(cfg.TEXT.DIMENSION_THETA3))

    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_THETA2_NET32(cfg.TEXT.DIMENSION_THETA2))
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_THETA2_NET64(cfg.TEXT.DIMENSION_THETA2))
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_THETA2_NET128(cfg.TEXT.DIMENSION_THETA2))

    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_THETA1_NET64(cfg.TEXT.DIMENSION_THETA1))
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_THETA1_NET128(cfg.TEXT.DIMENSION_THETA1))
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_THETA1_NET256(cfg.TEXT.DIMENSION_THETA1))

    for i in xrange(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.COUNT != '':
        count = np.load('%s' % (cfg.TRAIN.COUNT))
    if cfg.TRAIN.NET_G != '':
        for i in xrange(len(netsG)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_G, i))
            state_dict = torch.load('%s%d' % (cfg.TRAIN.NET_G, i))
            netsG[i].load_state_dict(state_dict)

    if cfg.TRAIN.NET_D != '':
        for i in xrange(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    if cfg.INCEPTION:
        inception_model = INCEPTION_V3()
        if cfg.CUDA:
            inception_model = inception_model.cuda()
        inception_model.eval()
    else:
        inception_model = None

    if cfg.CUDA:
        netEn.cuda()
        for i in xrange(len(netsG)):
            netsG[i].cuda()
        for i in xrange(len(netsD)):
            netsD[i].cuda()

    return netEn, netsG, netsD, len(netsD), inception_model, count


def define_optimizers(netEn, netsG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in xrange(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizersG = []
    num_Gs = len(netsG)
    for i in xrange(num_Gs):
        opt = optim.Adam(netsG[i].parameters(),
                         lr=cfg.TRAIN.GENERATOR_LR,
                         betas=(0.5, 0.999))
        optimizersG.append(opt)

    optimizerEn = optim.Adam(netEn.parameters(),
                             lr=cfg.TRAIN.ENCODER_LR)

    return optimizerEn, optimizersG, optimizersD


def save_model(netEn, netsG, avg_param_G, netsD, count, model_dir):
    np.save('%s/count.npy' % model_dir, count)
    torch.save(netEn.state_dict(),
               '%s/netEn.pth' % model_dir)
    for i in xrange(len(netsG)):
        load_params(netsG[i], avg_param_G[i])
        torch.save(
            netsG[i].state_dict(),
            '%s/netG%d.pth' % (model_dir, i))
    for i in xrange(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save En/G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir):
    num = cfg.TRAIN.VIS_COUNT

    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/count_%09d_real_samples.png' % (image_dir, count),
        normalize=True)

    for i in xrange(num_imgs):
        fake_img = fake_imgs[i][0:num]

        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader):
        if cfg.TRAIN.FLAG:
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
        for i in xrange(len(imgs)):
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
        criterion, c_code = self.criterion, self.c_code[idx // 3]

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_tgpu[int((idx // 3) + idx % 3)]
        wrong_imgs = self.wrong_tgpu[int((idx // 3) + idx % 3)]
        fake_imgs = self.fake_imgs[idx]

        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logits = netD(real_imgs, c_code.detach())
        wrong_logits = netD(wrong_imgs, c_code.detach())
        fake_logits = netD(fake_imgs.detach(), c_code.detach())

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

    def train_Gnet(self, idx, count):
        optG = self.optimizersG[idx]
        optG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_tgpu[0].size(0)
        criterion, c_code = self.criterion, self.c_code[idx]
        real_labels = self.real_labels[:batch_size]
        for i in xrange(len(self.netsG)):
            outputs = self.netsD[idx * 3 + i](self.fake_imgs[idx * 3 + i], c_code)
            errG = criterion(outputs[0], real_labels)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS *\
                    criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG
            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_D, count)

        errG_total = errG_total
        errG_total.backward()
        optG.step()
        return float(errG_total)

    def train_Enet(self, count):
        errEn_total = 0
        flag = count % 100
        params = [self.shape_1, self.scale_1, self.shape_2, self.scale_2, self.shape_3, self.scale_3, self.Phi[0],
                  self.theta_1, self.Phi[1], self.theta_2, self.Phi[2], self.theta_3, self.txtbow]
        criterion, optEn, netEn = self.vae, self.optimizerEn, self.netEn
        loss, theta3_KL, theta2_KL, theta1_KL, Likelihood, Lowerbound = criterion(params)
        errEn_total = errEn_total + loss
        netEn.zero_grad()
        # backward
        errEn_total.backward()
        # update parameters
        optEn.step()
        if flag == 0:
            summary_LS = summary.scalar('En_loss', float(loss.data.item()))
            summary_LB = summary.scalar('En_lowerbound', float(Lowerbound.data.item()))
            summary_LL = summary.scalar('En_likelihood', float(Likelihood.data.item()))
            summary_KL1 = summary.scalar('En_kl1', float(theta1_KL.data.item()))
            summary_KL2 = summary.scalar('En_kl2', float(theta2_KL.data.item()))
            summary_KL3 = summary.scalar('En_kl3', float(theta3_KL.data.item()))
            self.summary_writer.add_summary(summary_LS, count)
            self.summary_writer.add_summary(summary_LB, count)
            self.summary_writer.add_summary(summary_LL, count)
            self.summary_writer.add_summary(summary_KL1, count)
            self.summary_writer.add_summary(summary_KL2, count)
            self.summary_writer.add_summary(summary_KL3, count)
        return theta1_KL, theta2_KL, theta3_KL, Likelihood, Lowerbound, loss

    def updatePhi(self, miniBatch, Phi, Theta, MBratio, MBObserved):
        real_min = 1e-6
        Xt = miniBatch

        for t in range(len(Phi)):
            if t == 0:
                self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), Phi[t], Theta[t])
            else:
                self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(self.Xt_to_t1[t - 1], Phi[t],
                                                                                  Theta[t])

            self.EWSZS[t] = MBratio * self.WSZS[t]

            if (MBObserved == 0):
                self.NDot[t] = self.EWSZS[t].sum(0)
            else:
                self.NDot[t] = (1 - self.ForgetRate[MBObserved]) * self.NDot[t] + self.ForgetRate[MBObserved] * \
                               self.EWSZS[t].sum(0)

            tmp = self.EWSZS[t] + self.eta[t]
            tmp = (1 / (self.NDot[t] + real_min)) * (tmp - tmp.sum(0) * Phi[t])
            tmp1 = (2 / (self.NDot[t] + real_min)) * Phi[t]
            tmp = Phi[t] + self.epsit[MBObserved] * tmp + np.sqrt(self.epsit[MBObserved] * tmp1) * np.random.randn(
                Phi[t].shape[0], Phi[t].shape[1])
            Phi[t] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[t], 0)

        return Phi

    def train(self):
        self.netEn, self.netsG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = []
        for i in xrange(len(self.netsG)):
            avg_param_G.append(copy_G_params(self.netsG[i]))

        self.optimizerEn, self.optimizersG, self.optimizersD = \
            define_optimizers(self.netEn, self.netsG, self.netsD)

        self.criterion = nn.BCELoss()
        self.vae = myLoss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))

        predictions = []
        count = start_count
        start_epoch = start_count // self.num_batches

        batch_length = self.num_batches

        self.Phi = []
        self.eta = []
        K = [256, 128, 64]
        real_min = np.float64(2.2e-308)
        eta = np.ones(3) * 0.1
        for i in range(3):
            self.eta.append(eta[i])
            if i == 0:
                self.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(1000, K[i])))
            else:
                self.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(K[i - 1], K[i])))
            self.Phi[i] = self.Phi[i] / np.maximum(real_min, self.Phi[i].sum(0))

        self.NDot = [0] * 3
        self.Xt_to_t1 = [0] * 3
        self.WSZS = [0] * 3
        self.EWSZS = [0] * 3

        self.ForgetRate = np.power((0 + np.linspace(1, cfg.TRAIN.MAX_EPOCH * int(batch_length),
                                                    cfg.TRAIN.MAX_EPOCH * int(batch_length))), -0.7)
        epsit = np.power((20 + np.linspace(1, cfg.TRAIN.MAX_EPOCH * int(batch_length),
                                           cfg.TRAIN.MAX_EPOCH * int(batch_length))), -0.7)
        self.epsit = 1 * epsit / epsit[0]

        num_total_samples = batch_length * self.batch_size

        if cfg.CUDA:
            for i in xrange(len(self.Phi)):
                self.Phi[i] = Variable(torch.from_numpy(self.Phi[i]).float()).cuda()
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        for epoch in xrange(start_epoch, self.max_epoch):
            start_t = time.time()
            LL = 0
            KL1 = 0
            KL2 = 0
            KL3 = 0
            LS = 0
            DL = 0
            GL = 0
            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.img_tcpu, self.txtbow, self.real_tgpu, self.wrong_tgpu = self.prepare_data(data)

                #######################################################
                # (1) Get conv hidden units
                ######################################################
                _, self.flat = self.inception_model(self.real_tgpu[-1])

                self.theta_1, self.shape_1, self.scale_1, self.theta_2,\
                self.shape_2, self.scale_2, self.theta_3, self.shape_3,\
                self.scale_3 = self.netEn(self.flat)

                self.txt_embedding = []
                self.txt_embedding.append(self.theta_3.detach())
                self.txt_embedding.append(self.theta_2.detach())
                self.txt_embedding.append(self.theta_1.detach())
                #######################################################
                # (2) Generate fake images
                ######################################################
                tmp = []
                self.c_code = []
                x_embedding = None
                for it in xrange(len(self.netsG)):
                    fake_imgs, c_code, x_embedding = \
                        self.netsG[it](self.txt_embedding[it], x_embedding)
                    tmp.append(fake_imgs)
                    self.c_code.append(c_code)

                self.fake_imgs = []
                for it in xrange(len(tmp)):
                    for jt in xrange(len(tmp[it])):
                        self.fake_imgs.append(tmp[it][jt])

                #######################################################
                # (3) Update En network
                ######################################################
                self.KL1, self.KL2, self.KL3, self.LL, self.LB, self.LS = self.train_Enet(count)
                LL += self.LL
                KL1 += self.KL1
                KL2 += self.KL2
                KL3 += self.KL3
                LS += self.LS

                if count % 100 == 0:
                    print(self.LS)
                    print(self.KL1)
                    print(self.KL2)
                    print(self.KL3)

                #######################################################
                # (4) Update Phi
                #######################################################
                input_txt = np.array(np.transpose(self.txtbow.cpu().numpy()), order='C').astype('double')
                Phi = []
                theta = []
                self.theta = [self.theta_1, self.theta_2, self.theta_3]
                for i in xrange(len(self.Phi)):
                    Phi.append(np.array(self.Phi[i].cpu().numpy(), order='C').astype('double'))
                    theta.append(np.array(np.transpose(self.theta[i].detach().cpu().numpy()), order='C').astype('double'))
                phi = self.updatePhi(input_txt, Phi, theta, int(batch_length), count)
                for i in xrange(len(phi)):
                    self.Phi[i] = torch.tensor(phi[i], dtype=torch.float32).cuda()

                #######################################################
                # (5) Update D network
                ######################################################
                errD_total = 0
                for i in xrange(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD
                DL += errD_total

                #######################################################
                # (6) Update G network: maximize log(D(G(z)))
                ######################################################
                errG_total = 0
                for i in xrange(len(self.netsG)):
                    errG = self.train_Gnet(i, count)
                    errG_total += errG
                    for p, avg_p in zip(self.netsG[i].parameters(), avg_param_G[i]):
                        avg_p.mul_(0.999).add_(0.001, p.data)
                GL += errG_total

                # for inception score
                if cfg.INCEPTION:
                    pred, _ = self.inception_model(self.fake_imgs[-1].detach())
                    predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total)
                    summary_G = summary.scalar('G_loss', errG_total)
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netEn, self.netsG, avg_param_G, self.netsD, epoch, self.model_dir)
                    # Save images
                    backup_para = []
                    for i in xrange(len(self.netsG)):
                        backup_para.append(copy_G_params(self.netsG[i]))
                        load_params(self.netsG[i], avg_param_G[i])

                    x_embedding = None
                    self.fake_imgs = []
                    for it in xrange(len(self.netsG)):
                        fake_imgs, _, x_embedding = self.netsG[it](self.txt_embedding[it], x_embedding)
                        self.fake_imgs.append(fake_imgs[-1])
                    save_img_results(self.img_tcpu, self.fake_imgs, len(self.netsG),
                                     count, self.image_dir)

                    for i in xrange(len(self.netsG)):
                        load_params(self.netsG[i], backup_para[i])

                    if cfg.INCEPTION:
                        # Compute inception score
                        if len(predictions) > 500:
                            predictions = np.concatenate(predictions, 0)
                            mean, std = compute_inception_score(predictions, 10)
                            m_incep = summary.scalar('Inception_mean', mean)
                            self.summary_writer.add_summary(m_incep, count)

                            mean_nlpp, std_nlpp = \
                                negative_log_posterior_probability(predictions, 10)
                            m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                            self.summary_writer.add_summary(m_nlpp, count)

                            predictions = []

                count = count + 1

            end_t = time.time()
            LS = LS / num_total_samples
            LL = LL / num_total_samples
            KL1 = KL1 / num_total_samples
            KL2 = KL2 / num_total_samples
            KL3 = KL3 / num_total_samples
            DL = DL / num_total_samples
            GL = GL / num_total_samples
            print('Epoch: %d/%d,   Time elapsed: %.4fs\n'
                  '* Batch Train Loss: %.6f          (LL: %.6f, KL1: %.6f, KL2: %.6f,'
                  'KL3: %.6f, Loss_D: %.2f Loss_G: %.2f)\n' % (epoch, self.max_epoch,
                                                               end_t - start_t, LS, LL,
                                                               KL1, KL2, KL3, DL, GL))
        save_model(self.netEn, self.netsG, avg_param_G, self.netsD, epoch, self.model_dir)
        self.summary_writer.close()