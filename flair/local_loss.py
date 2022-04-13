import math

import torch
from torch import nn

class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, label_num, **
        # shape(y_true) = batch_size, **
        y_pred = torch.softmax(y_pred, dim=1)

        pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))

        dsc_i = 1 - ((1 - pred_prob) * pred_prob) / ((1 - pred_prob) * pred_prob + 1)
        # print(dsc_i)
        dice_loss = dsc_i.mean()
        # dice_loss = dsc_i.sum()
        # asd
        return dice_loss