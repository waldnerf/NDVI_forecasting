import torch
from nn.loss import soft_dtw
from nn.loss import path_soft_dtw
from torch import nn
import torch.nn.functional as F


def dilate_loss(outputs, targets, alpha=0.5, gamma=0.01):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal


class DilateLoss(nn.Module):
    def __init__(self, device, alpha=0.5, gamma=0.01, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, outputs, targets):
        # outputs, targets: shape (batch_size, N_output, 1)
        batch_size, N_output = outputs.shape[0:2]
        loss_shape = 0
        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        D = torch.zeros((batch_size, N_output, N_output)).to(self.device)
        for k in range(batch_size):
            Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = softdtw_batch(D, self.gamma)

        path_dtw = path_soft_dtw.PathDTWBatch.apply
        path = path_dtw(D, self.gamma)
        Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(self.device)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss
