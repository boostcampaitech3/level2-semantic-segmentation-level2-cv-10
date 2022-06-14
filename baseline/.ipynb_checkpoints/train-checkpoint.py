import torch
import torch.nn as nn
import numpy as np
import argparse
import numpy as np

from config_parser import ConfigParser
import data_loader.dataloader as module_dataloader

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # [1] Logger
    logger = config.get_logger('train')
    
    # [2] DataLoader
    train_loader = config.init_obj(module_dataloader, 'train_loader')
    val_loader = config.init_obj(module_dataloader, 'val_loader')
    
    # [3] Model
    
    # [4] criterion
    
    # [5] metrics
    
    # [6] optimizer
    
    # [7] lr_scheduler
    
    # [8] set up Trainer
    
    # [9] train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-l', '--loss', default=None, type=str)
    parser.add_argument('-e', '--epochs', default=None, type=int)
    parser.add_argument('-lr', '--learning_rate', default=None, type=float)
    parser.add_argument('-bs', "--batch_size", default=None, type=int)

    args = parser.parse_args()
    config = ConfigParser(args)

    main(config)




