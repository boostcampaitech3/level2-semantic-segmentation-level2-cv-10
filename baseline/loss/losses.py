import torch
import torch.nn as nn

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)


