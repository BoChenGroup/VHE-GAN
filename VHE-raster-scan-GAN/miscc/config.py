from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'pgbn-hierachical'
__C.MODE_MODEL = 'serialHierachicalModel'
__C.CONFIG_NAME = '3stages'
__C.DATA_DIR = '../data/birds'

__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 4
__C.INCEPTION = True

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 5
__C.TREE.BASE_SIZE = 16
__C.TREE.NORM = False

# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = True
__C.TEST.SAMPLE_NUM = 30000


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 8
__C.TRAIN.MAX_EPOCH = 300
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 1e-4
__C.TRAIN.FLAG = True
__C.TRAIN.COUNT = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_D = ''
__C.TRAIN.NET_MLP_EN = ''

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.UNCOND_LOSS = 1.0
__C.TRAIN.COEFF.KL_theta3_LOSS = 0.01
__C.TRAIN.COEFF.KL_theta2_LOSS = 0.01
__C.TRAIN.COEFF.KL_theta1_LOSS = 0.01

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 0
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2
__C.GAN.B_CONDITION = True

__C.TEXT = edict()
__C.TEXT.DIMENSION_THETA3 = 64
__C.TEXT.DIMENSION_THETA2 = 128
__C.TEXT.DIMENSION_THETA1 = 256
