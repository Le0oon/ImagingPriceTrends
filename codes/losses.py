""" Loss 损失函数 """

import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # 权重因子用于分类交叉熵损失
        self.beta = beta    # 权重因子用于MSE损失
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, targets, decoder_outputs, decoder_targets):

        ce_loss = self.cross_entropy_loss(logits, targets)
        mse_loss = self.mse_loss(decoder_outputs, decoder_targets)

        total_loss = self.alpha * ce_loss + self.beta * mse_loss

        return total_loss


def loss(args):
    loss_lower = args.loss.lower()

    if loss_lower == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_lower == 'mse':
        loss = nn.MSELoss()
    elif loss_lower == 'l1':
        loss = nn.L1Loss()
    else:
        assert False and "Invalid optimizer"
    fp = open('output.log', 'a+')
    print(f'loss is {loss}', file=fp)
    fp.close()
    return loss
