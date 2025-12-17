import torch
from torch import nn


class LossTotal(nn.Module):
    def __init__(self, weight_ba_loss, weight_ce_loss, device):
        super(LossTotal, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
        torch.nn.init.constant_(self.bn.weight, 1)
        self.bn.to(device)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_ba_loss = weight_ba_loss
        self.weight_ce_loss = weight_ce_loss

    def forward(self, y, lbl):
        lbl_float = lbl.float()

        diff = y[:, 1] - y[:, 0]  # 第1维大的为changed
        ce_loss = self.bce_loss(diff, lbl_float)
        diff = torch.unsqueeze(diff, 1)
        diff = self.bn(diff)
        iou_loss = 1 - torch.sum(torch.sigmoid(diff) * lbl_float) / torch.sum(
            torch.sigmoid(diff) + lbl_float - torch.sigmoid(diff) * lbl_float)
        loss = iou_loss * self.weight_ba_loss + ce_loss * self.weight_ce_loss
        return loss