import torch
import numpy as np


class DiceLoss(torch.nn.Module):

    def __init__(self, laplace=0.1):
        super(DiceLoss, self).__init__()
        self.laplace = laplace

    def dice_similarity_coefficient(self, pred, gt, alpha=0.5):
        pg = (gt * pred).mean()
        png = ((1 - gt) * pred).mean()
        gnp = ((1 - pred) * gt).mean()
        dsc = (pg + self.laplace) / (pg + (1 - alpha) * png + alpha * gnp + self.laplace)
        return dsc

    def __call__(self, pred, gt):
        return 1 - self.dice_similarity_coefficient(pred, gt)


class RootTverskyLoss(DiceLoss):

    def __init__(self, alpha=0.3, laplace=0.1):
        super(RootTverskyLoss, self).__init__(laplace)
        self.alpha = alpha

    def __call__(self, pred, gt):
        return 1 - self.dice_similarity_coefficient(pred, gt, self.alpha)


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, gt):
        beta = (1 - gt.mean()) * (1 - self.alpha)
        loss_object = - ((1 - pred) ** self.gamma * torch.log(pred) * gt).mean()
        loss_background = - (pred ** self.gamma * torch.log(1 - pred) * (1 - gt)).mean()
        return beta * loss_object + (1 - beta) * loss_background


class BiasedBCELoss(torch.nn.Module):

    def __init__(self, biased_weight=0.8, epsilon=1e-9):
        super(BiasedBCELoss, self).__init__()
        self.biased_weight = biased_weight
        self.epsilon = epsilon

    def __call__(self, pred, gt):
        beta = (1 - gt.mean()) * self.biased_weight
        loss_object = - (torch.log(pred + self.epsilon) * gt).mean()
        loss_background = - (torch.log(1 - pred + self.epsilon) * (1 - gt)).mean()
        return loss_object * beta + loss_background * (1 - beta)


def dice_similarity_coefficient(pred, gt):
    inter_section = 2 * (gt * pred).mean()
    union_section = gt.mean() + pred.mean()
    return inter_section / union_section


def precision_rate(pred, gt):
    true_pos = (gt * pred).sum()
    false_pos = ((1 - gt) * pred).sum()
    return true_pos / (true_pos + false_pos + 1e-6)


class BiasedMSELoss(torch.nn.Module):

    def __init__(self, biased_weight=0.8, epsilon=1e-3):
        super(BiasedMSELoss, self).__init__()
        self.biased_weight = biased_weight
        self.epsilon = epsilon

    def __call__(self, pred, gt):
        weight = gt + self.epsilon
        loss = (pred - gt) ** 2 * weight
        loss = (loss.mean()) ** 0.5
        return loss


class AdversarialLoss(torch.nn.Module):

    def __init__(self, label, epsilon=1e-3):
        """
            label 0: AdversarialLoss for G
            label 1: AdversarialLoss for D
        """
        super(AdversarialLoss, self).__init__()
        if label in ["D", "G"]:
            self.label = label
        else:
            assert "The label of the AdversarialLoss should be D or G."
        self.epsilon = epsilon

    def __call__(self, real, fake):
        if self.label == "D":
            loss_real = - torch.log(real + self.epsilon)
            loss_fake = torch.log(fake + self.epsilon)
        else:
            loss_real = 0.
            loss_fake = - torch.log(fake + self.epsilon)
        return (loss_real + loss_fake).sum()


if __name__ == '__main__':
    # D:
    # real = torch.Tensor([1.])
    # fake = torch.Tensor([0.])

    # real = torch.Tensor([.5])
    # fake = torch.Tensor([.5])
    # print(AdversarialLoss(1)(real, fake))

    # G
    real = torch.Tensor([1.])
    fake = torch.Tensor([0.])
    print(AdversarialLoss(0)(real, fake))

    real = torch.Tensor([1.])
    fake = torch.Tensor([1.])
    print(AdversarialLoss(0)(real, fake))