import argparse
import logging
import math
import numpy as np
import os
import sys
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

import src.latent_sr.options as option
from src.latent_sr.models import create_model

import src.utils as util
from src.data import create_dataloader, create_dataset
from src.data.data_sampler import DistIterSampler

from src.data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.", default="latent_sisr.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    args.opt = os.path.join('src/latent_sr/options/train/', args.opt)
    opt = option.parse(args.opt, ts, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]
    torch.manual_seed(seed)

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)
        
    model = create_model(opt) 
    device = model.device
    print(device)

    model.print_network()

if __name__ == "__main__":
    main()