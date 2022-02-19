import torch

from config import cfg

def smooth_l1_loss(x, t, in_weight, sigma):
    """
    这里这个sigma相当于_smooth_l1_loss在L1与L2之间切换的阈值,
    sigma为1时(默认)  smooth_l1_loss = 0.5x^2              |x| < 1
                     smooth_l1_loss = |x|-0.5             |x| >= 1
    sigma为3时,      smooth_l1_loss = 0.5x^2 * sigma^2    |x| < 1/sigma^2
                    smooth_l1_loss = |x|- 0.5/sigma^2   |x| >= 1/sigma^2
    在该份代码中计算rpn网络的损失时 sigma为3,roi网络的损失时 sigma为1
    """

    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """
    faster-rcnn中修正系数loc的损失计算
    :param pred_loc: (h*w*9,4) rpn网络中定位卷积提供的修正系数
    :param gt_loc: (h*w*9,4) 目标修正系数
    :param gt_label: (h*w*9,) 默认值为-1,正样本1负样本0(共256个)的全体anchor的label值
    :param sigma: 调整l1与l2损失函数切换的关键系数
    :return: loss计算结果
    """

    # (h*w*9,4) 全为0
    in_weight = torch.zeros(gt_loc.shape, device=cfg.device)
    # 将gt_label为1的位置设为1
    in_weight[(gt_label > 0).reshape(-1, 1).expand_as(in_weight).to(cfg.device)] = 1

    # 计算loss
    # in_weight将作为pred和gt每两个对于元素之差的权重 in_weight * (pred-gt)
    # 因此gt_label不为1的位置不参与loss计算，即仅计算正样本loss
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    
    return loc_loss