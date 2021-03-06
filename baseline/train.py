import torch
import torch.nn as nn
import numpy as np
import argparse
import numpy as np

import segmentation_models_pytorch as smp

import data_loader as module_dataloader
import loss as module_loss
import metric as module_metric
from trainer import Trainer
from models import build_smp_model


from config_parser import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # [0] device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # [1] Logger
    logger = config.get_logger('train')

    # [2] DataLoader
    train_loader = config.init_obj(module_dataloader, 'train_loader')
    val_loader = config.init_obj(module_dataloader, 'val_loader')

    # [3] SMP Model
    model_config = config['model']
    model = build_smp_model(model_config)
    model = model.to(device)

    # [4] loss_fn
    loss_fn = getattr(module_loss, config['loss'])

    # [5] metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # [6] optimizer
    optimizer = config.init_obj(torch.optim, 'optimizer', model.parameters())

    # [7] lr_scheduler
    lr_scheduler = config.init_obj(torch.optim.lr_scheduler, 'lr_scheduler', optimizer)

    # [8] set up Trainer
    trainer = Trainer(model, model_config, loss_fn, optimizer, lr_scheduler, metrics,
                      config=config,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader)

    # [9] train
    trainer.train()


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




