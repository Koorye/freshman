import torch
from torch import nn
from torchvision.ops import RoIPool

from config import cfg
from model.util.tool import normal_init


class RoIHead(nn.Module):
    def __init__(self, n_class, classifier):
        """
        : param n_class: 类别数
        : param classifier: 分类器
        """
        
        super(RoIHead, self).__init__()

        self.classifier = classifier

        # 回归器和分类器
        self.loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        # 初始化权重
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class

        # VGG16的缩放因子为1/16
        # 输出尺寸为(7,7)
        self.roi = RoIPool((7, 7), 1/16)

    def forward(self, x, rois):
        """
        : param x: vgg16网络提取的特征 -> (1,c,h,w)
        : param rois: RPN网络提供的roi -> (128,4) 128个样本，每个样本拥有(xmin,ymin,xmax,ymax)
        return:
            roi_locs: RoIHead网络提供的roi修正系数 -> (128, n_class*4)
            roi_scores: RoIHead网络提供的roi各类置信度 -> (128, n_class)
        """

        # (128,1) cat (128,4) -> (128,5) -> (0,xmin,ymin,xmax,ymax)
        # 第一列全为0，表示batch_id
        rois = torch.cat(
            (torch.zeros((rois.shape[0], 1), device=cfg.device), rois), 1)

        # (n_rois,c,7,7)
        # 即每个roi都在特征图上进行一次近似的自适应最大值池化
        pool = self.roi(x, rois).to(cfg.device)
        # (n_rois,c*7*7)
        pool = pool.reshape(pool.shape[0], -1)
        fc7 = self.classifier(pool)

        roi_locs = self.loc(fc7)
        roi_scores = self.score(fc7)

        return roi_locs, roi_scores
