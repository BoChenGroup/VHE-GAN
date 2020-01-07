import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo


# ############################## For Compute inception score ##############################
class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

    def forward(self, input):
        x = input * 0.5 + 0.5
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x, f = self.model(x)
        x = nn.Softmax()(x)
        return x, f


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()

    def forward(self, text_embedding):
        c_code = text_embedding
        return c_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, code_dim, numdiv, num_residual=cfg.GAN.R_NUM):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.num_div = numdiv
        self.code_dim = code_dim
        self.num_residual = num_residual
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + code_dim
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        ndiv = self.num_div
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        if self.code_dim == 64:
            self.upsample1 = upBlock(ngf, ngf // 2)
            self.upsample2 = upBlock(ngf // 2, ngf // 4)
        if self.code_dim == 128:
            self.upsample1 = upBlock(ngf, ngf // 2)
            self.upsample2 = upBlock(ngf // 2, ngf // 4)

            self.jointConv1 = Block3x3_relu(3, ngf // ndiv)

            self.jointConv2 = Block3x3_relu(ngf // 4 + ngf // (ndiv // 4), ngf // 4)
            self.residual = self._make_layer(ResBlock, ngf // 4)

            self.downsample1 = downBlock(ngf // ndiv, ngf // (ndiv // 2))
            self.downsample2 = downBlock(ngf // (ndiv // 2), ngf // (ndiv // 4))
            self.upsample3 = upBlock(ngf // 4, ngf // 8)
        if self.code_dim == 256:
            self.upsample1 = upBlock(ngf, ngf // 2)
            self.upsample2 = upBlock(ngf // 2, ngf // 4)

            self.jointConv1 = Block3x3_relu(3, ngf // ndiv)

            self.jointConv2 = Block3x3_relu(ngf // 4 + ngf // (ndiv // 8), ngf // 4)
            self.residual = self._make_layer(ResBlock, ngf // 4)

            self.downsample1 = downBlock(ngf // ndiv, ngf // (ndiv // 2))
            self.downsample2 = downBlock(ngf // (ndiv // 2), ngf // (ndiv // 4))
            self.downsample3 = downBlock(ngf // (ndiv // 4), ngf // (ndiv // 8))
            self.upsample3 = upBlock(ngf // 4, ngf // 8)
            self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, c_code=None, x_var=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = c_code
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        if c_code.size(1) == 128:
            if x_var is not None:
                x_var = self.jointConv1(x_var)
                x_var = self.downsample1(x_var)
                x_var = self.downsample2(x_var)
                out_code = torch.cat((out_code, x_var), 1)
                out_code = self.jointConv2(out_code)
                out_code = self.residual(out_code)
                out_code = self.upsample3(out_code)
        if c_code.size(1) == 256:
            if x_var is not None:
                x_var = self.jointConv1(x_var)
                x_var = self.downsample1(x_var)
                x_var = self.downsample2(x_var)
                x_var = self.downsample3(x_var)

                out_code = torch.cat((out_code, x_var), 1)
                out_code = self.jointConv2(out_code)
                out_code = self.residual(out_code)
                out_code = self.upsample3(out_code)
                out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, code_dim, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = code_dim
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((c_code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET_64(nn.Module):
    def __init__(self):
        super(G_NET_64, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_theta3_f3 = INIT_STAGE_G(self.gf_dim * 4, 64, 1)
            self.h_theta3_f2 = NEXT_STAGE_G(self.gf_dim, 64)
            self.h_theta3_f1 = NEXT_STAGE_G(self.gf_dim // 2, 64)
            self.img_theta3_net3 = GET_IMAGE_G(self.gf_dim)
            self.img_theta3_net2 = GET_IMAGE_G(self.gf_dim // 2)
            self.img_theta3_net1 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, text_embedding=None, x_embedding=None):
        if cfg.GAN.B_CONDITION is not None:
            c_code = self.ca_net(text_embedding)
        fake_imgs = []

        h_theta3_f3 = self.h_theta3_f3(c_code)
        fake_theta3_img3 = self.img_theta3_net3(h_theta3_f3)
        fake_imgs.append(fake_theta3_img3)

        h_theta3_f2 = self.h_theta3_f2(h_theta3_f3, c_code)
        fake_theta3_img2 = self.img_theta3_net2(h_theta3_f2)
        fake_imgs.append(fake_theta3_img2)

        h_theta3_f1 = self.h_theta3_f1(h_theta3_f2, c_code)
        fake_theta3_img1 = self.img_theta3_net1(h_theta3_f1)
        fake_imgs.append(fake_theta3_img1)

        x_embedding = fake_theta3_img1
        x_embedding = x_embedding.detach()

        return fake_imgs, c_code, x_embedding


class G_NET_128(nn.Module):
    def __init__(self):
        super(G_NET_128, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 1:
            self.h_theta2_f3 = INIT_STAGE_G(self.gf_dim * 8, 128, 256)
            self.h_theta2_f2 = NEXT_STAGE_G(self.gf_dim, 128)
            self.h_theta2_f1 = NEXT_STAGE_G(self.gf_dim // 2, 128)
            self.img_theta2_net3 = GET_IMAGE_G(self.gf_dim)
            self.img_theta2_net2 = GET_IMAGE_G(self.gf_dim // 2)
            self.img_theta2_net1 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, text_embedding=None, x_embedding=None):
        if cfg.GAN.B_CONDITION is not None:
            c_code = self.ca_net(text_embedding)
        fake_imgs = []

        h_theta2_f3 = self.h_theta2_f3(c_code, x_embedding)
        fake_theta2_img3 = self.img_theta2_net3(h_theta2_f3)
        fake_imgs.append(fake_theta2_img3)

        h_theta2_f2 = self.h_theta2_f2(h_theta2_f3, c_code)
        fake_theta2_img2 = self.img_theta2_net2(h_theta2_f2)
        fake_imgs.append(fake_theta2_img2)

        h_theta2_f1 = self.h_theta2_f1(h_theta2_f2, c_code)
        fake_theta2_img1 = self.img_theta2_net1(h_theta2_f1)
        fake_imgs.append(fake_theta2_img1)

        x_embedding = fake_theta2_img1
        x_embedding = x_embedding.detach()
        return fake_imgs, c_code, x_embedding


class G_NET_256(nn.Module):
    def __init__(self):
        super(G_NET_256, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 2:
            self.h_theta1_f3 = INIT_STAGE_G(self.gf_dim * 16, 256, 512)
            self.h_theta1_f2 = NEXT_STAGE_G(self.gf_dim, 256)
            self.h_theta1_f1 = NEXT_STAGE_G(self.gf_dim // 2, 256)
            self.img_theta1_net3 = GET_IMAGE_G(self.gf_dim)
            self.img_theta1_net2 = GET_IMAGE_G(self.gf_dim // 2)
            self.img_theta1_net1 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, text_embedding=None, x_embedding=None):
        if cfg.GAN.B_CONDITION is not None:
            c_code = self.ca_net(text_embedding)
        fake_imgs = []

        h_theta1_f3 = self.h_theta1_f3(c_code, x_embedding)
        fake_theta1_img3 = self.img_theta1_net3(h_theta1_f3)
        fake_imgs.append(fake_theta1_img3)

        h_theta1_f2 = self.h_theta1_f2(h_theta1_f3, c_code)
        fake_theta1_img2 = self.img_theta1_net2(h_theta1_f2)
        fake_imgs.append(fake_theta1_img2)

        h_theta1_f1 = self.h_theta1_f1(h_theta1_f2, c_code)
        fake_theta1_img1 = self.img_theta1_net1(h_theta1_f1)
        fake_imgs.append(fake_theta1_img1)

        return fake_imgs, c_code, None


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# Downsale the spatial size by a factor of 8
def encode_image_by_8times(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# Downsale the spatial size by a factor of 4
def encode_image_by_4times(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For THETA3-16/32/64 images
class D_THETA3_NET16(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA3_NET16, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s4 = encode_image_by_4times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 2 + efg, ndf * 2)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s4(x_var)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA3_NET32(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA3_NET32, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s4 = encode_image_by_4times(ndf)
        self.img_code_s8 = downBlock(ndf * 2, ndf * 4)
        self.img_code_s8_1 = Block3x3_leakRelu(ndf * 4, ndf * 2)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 2 + efg, ndf * 2)
            self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s4(x_var)
        x_code = self.img_code_s8(x_code)
        x_code = self.img_code_s8_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA3_NET64(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA3_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s4 = encode_image_by_4times(ndf)
        self.img_code_s8 = downBlock(ndf * 2, ndf * 4)
        self.img_code_s16 = downBlock(ndf * 4, ndf * 8)
        self.img_code_s16_1 = Block3x3_leakRelu(ndf * 8, ndf * 4)
        self.img_code_s16_2 = Block3x3_leakRelu(ndf * 4, ndf * 2)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 2 + efg, ndf * 2)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s4(x_var)
        x_code = self.img_code_s8(x_code)
        x_code = self.img_code_s16(x_code)
        x_code = self.img_code_s16_1(x_code)
        x_code = self.img_code_s16_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For THETA2-32/64/128 images
class D_THETA2_NET32(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA2_NET32, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 4 + efg, ndf * 4)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA2_NET64(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA2_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf)
        self.img_code_s16 = downBlock(ndf * 4, ndf * 8)
        self.img_code_s16_1 = Block3x3_leakRelu(ndf * 8, ndf * 4)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 4 + efg, ndf * 4)
            self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)
        x_code = self.img_code_s16(x_code)
        x_code = self.img_code_s16_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA2_NET128(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA2_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf)
        self.img_code_s16 = downBlock(ndf * 4, ndf * 8)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.img_code_s32_2 = Block3x3_leakRelu(ndf * 8, ndf * 4)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 4 + efg, ndf * 4)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)
        x_code = self.img_code_s16(x_code)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        x_code = self.img_code_s32_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For THETA1-64/128/256 images
class D_THETA1_NET64(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA1_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA1_NET128(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA1_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


class D_THETA1_NET256(nn.Module):
    def __init__(self, code_dim):
        super(D_THETA1_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = code_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# ############################WHAI##############################################
real_max = torch.tensor(10.0, dtype=torch.float32)
real_min = torch.tensor(1e-10, dtype=torch.float32)
eulergamma = torch.tensor(0.5772, dtype=torch.float32)
Theta1Scale_prior = torch.tensor(1.0, dtype=torch.float32)
Theta2Scale_prior = torch.tensor(1.0, dtype=torch.float32)
prior_shape = torch.tensor(0.01, dtype=torch.float32)
prior_scale = torch.tensor(1, dtype=torch.float32)


def log_max(x):
    tmp = torch.log(torch.max(x, real_min.cuda()))
    return tmp


def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    part1 = eulergamma.cuda() * (1-1/Wei_shape)+ log_max(Wei_scale/Wei_shape)+1+Gam_shape * torch.log(Gam_scale)
    part2 = -torch.lgamma(Gam_shape)+(Gam_shape - 1) * (log_max(Wei_scale) - eulergamma.cuda() / Wei_shape)
    KL = part1 + part2 - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
    return KL


class activation(nn.Module):
    def __init__(self):
        super(activation, self).__init__()

    def forward(self, x):
        return F.tanh(x)


class repmat(nn.Module):
    def __init__(self, dim):
        super(repmat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.repeat(1, self.dim)


class MLP_ENCODER(nn.Module):
    def __init__(self, theta1Dim, theta2Dim, theta3Dim):
        super(MLP_ENCODER, self).__init__()
        self.tD1 = theta1Dim
        self.tD2 = theta2Dim
        self.tD3 = theta3Dim

        self.h_mlp_1 = nn.Sequential(
            nn.Linear(2048, 512),
            activation()
        )

        self.shape_net_mlp_1 = nn.Sequential(
            nn.Linear(512, 1),
            repmat(256)
        )

        self.scale_net_mlp_1 = nn.Sequential(
            nn.Linear(512, 256)
        )

        self.h_mlp_2 = nn.Sequential(
            nn.Linear(512, 256),
            activation()
        )

        self.shape_net_mlp_2 = nn.Sequential(
            nn.Linear(256, 1),
            repmat(128)
        )

        self.scale_net_mlp_2 = nn.Sequential(
            nn.Linear(256, 128)
        )

        self.h_mlp_3 = nn.Sequential(
            nn.Linear(256, 128),
            activation()
        )

        self.shape_net_mlp_3 = nn.Sequential(
            nn.Linear(128, 1),
            repmat(64)
        )

        self.scale_net_mlp_3 = nn.Sequential(
            nn.Linear(128, 64)
        )

    def reparameterize(self, Wei_shape, Wei_scale, K_dim):
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(cfg.TRAIN.BATCH_SIZE, K_dim).uniform_()
        else:
            eps = torch.FloatTensor(cfg.TRAIN.BATCH_SIZE, K_dim).uniform_()
        theta = Wei_scale * torch.pow(-log_max(1 - eps), 1/Wei_shape)
        return theta

    def forward(self, flat):
        h_1 = self.h_mlp_1(flat)
        shape_1 = self.shape_net_mlp_1(h_1)
        scale_1 = self.scale_net_mlp_1(h_1)
        scale_1 = torch.max(torch.log(1 + torch.exp(scale_1)), real_min.cuda())
        shape_1 = torch.min(torch.max(torch.log(1 + torch.exp(shape_1)), real_min.cuda()), real_max.cuda())
        theta_1 = self.reparameterize(shape_1, scale_1, cfg.TEXT.DIMENSION_THETA1)

        h_2 = self.h_mlp_2(h_1)
        shape_2 = self.shape_net_mlp_2(h_2)
        scale_2 = self.scale_net_mlp_2(h_2)
        scale_2 = torch.max(torch.log(1 + torch.exp(scale_2)), real_min.cuda())
        shape_2 = torch.min(torch.max(torch.log(1 + torch.exp(shape_2)), real_min.cuda()), real_max.cuda())
        theta_2 = self.reparameterize(shape_2, scale_2, cfg.TEXT.DIMENSION_THETA2)

        h_3 = self.h_mlp_3(h_2)
        shape_3 = self.shape_net_mlp_3(h_3)
        scale_3 = self.scale_net_mlp_3(h_3)
        scale_3 = torch.max(torch.log(1 + torch.exp(scale_3)), real_min.cuda())
        shape_3 = torch.min(torch.max(torch.log(1 + torch.exp(shape_3)), real_min.cuda()), real_max.cuda())
        theta_3 = self.reparameterize(shape_3, scale_3, cfg.TEXT.DIMENSION_THETA3)

        return theta_1, shape_1, scale_1, theta_2, shape_2, scale_2, theta_3, shape_3, scale_3


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, params):
        shape1, scale1, shape2, scale2, shape3, scale3, phi1, theta1, phi2, theta2, phi3, theta3,\
        t_txt = params
        t_txt = torch.tensor(t_txt, dtype=torch.float32).cuda()
        theta3_KL = torch.sum(KL_GamWei(prior_shape.cuda(), prior_scale.cuda(), shape3, scale3))
        theta2_KL = torch.sum(KL_GamWei(torch.matmul(theta3, torch.transpose(phi3, 1, 0)), Theta2Scale_prior.cuda(),
                                        shape2, scale2))
        theta1_KL = torch.sum(KL_GamWei(torch.matmul(theta2, torch.transpose(phi2, 1, 0)), Theta1Scale_prior.cuda(),
                                        shape1, scale1))
        Likelihood = torch.sum(
            t_txt * log_max(torch.matmul(theta1, torch.transpose(phi1, 1, 0))) - torch.matmul(theta1, torch.transpose(phi1, 1, 0)) -
            torch.lgamma(((t_txt + 1))))
        loss = cfg.TRAIN.COEFF.KL_theta3_LOSS * theta3_KL + cfg.TRAIN.COEFF.KL_theta2_LOSS *\
               theta2_KL + cfg.TRAIN.COEFF.KL_theta1_LOSS * theta1_KL + Likelihood
        Lowerbound = Likelihood + theta3_KL + theta2_KL + theta1_KL
        return -loss, theta3_KL, theta2_KL, theta1_KL, Likelihood, Lowerbound