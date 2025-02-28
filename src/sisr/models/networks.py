import logging

import torch

from src.sisr.models import modules as M

logger = logging.getLogger("base")

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["type"]
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(**setting)
    return netG


# Discriminator
def define_L(opt):
    opt_net = opt["network_L"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
