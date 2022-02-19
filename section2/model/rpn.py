import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from model.util.tool import normal_init
from model.util.roi_creator import RoICreator


class RPN(nn.Module):
    def __init__(self) -> None:
        super(RPN, self).__init__()

        self.feat_stride = 16

        self.ratios = torch.Tensor([.5, 1, 2])
        self.anchor_scales = torch.Tensor([8, 16, 32])
        self.anchor_base = None

        self.generate_anchor_base()

        self.roi_creator = RoICreator(self)
        self.anchor_types = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.score = nn.Conv2d(512, self.anchor_types*2, 1, 1, 0)
        self.loc = nn.Conv2d(512, self.anchor_types*4, 1, 1, 0)

        normal_init(self.conv1, 0, .01)
        normal_init(self.score, 0, .01)
        normal_init(self.loc, 0, .01)

    def forward(self, x, img_size, scale=1.):
        """
        : param x: 输入的特征图 (1,c,h,w)
        : param img_size: 原始图像的尺寸 (h,w)，创建roi时将限制在原始图像的宽高范围内
        : param scale: 缩放，原始图像预处理时的缩放倍数
        : return:
          rpn_locs: 回归系数 (n_rois,4)
          rpn_scores: 正负样本的分数 (n_rois,2)
          rois: ROI框 (n_rois,4)
          anchors: 原始框 (h*w*9,4)
        """
        # 特征图的尺寸
        b, c, h, w = x.shape

        anchor = self.create_anchor_all(h, w)

        # 回归和分类
        x = F.relu(self.conv1(x))
        # (b,36,h,w)
        rpn_locs = self.loc(x)
        # (b,18,h,w)
        rpn_scores = self.score(x)

        # (b,36,h,w) -> (b,h,w,36) -> (b,h*w*9,4)
        # 即每个batch有h*w*9个anchor
        # 每个anchor有4个位置回归信息 (dx,dy,dw,dh)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).reshape(b, -1, 4)
        # 每个anchor有2个分数信息 (b,h,w,9*2)
        # (b,18,h,w) -> (b,h,w,18)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1)

        # (b,h,w,18) -> (b,h,w,9,2)
        # 在第4维即对正样本和负样本分数作softmax
        rpn_softmax_scores = F.softmax(rpn_scores.reshape(b,h,w,self.anchor_types,2), dim=4)
        # 取所有anchor的负样本分数 (b,h,w,9)
        rpn_fg_scores = rpn_softmax_scores[:,:,:,:,1]
        # (b,h,w,9) -> (b,h*w*9)
        rpn_fg_scores = rpn_fg_scores.view(b,-1)

        # (b,h,w,18) -> (b,h*w*9,2)
        rpn_scores = rpn_scores.reshape(b,-1,2)

        # 根据回归信息和基础anchor得到roi
        # 限制roi的x,y,w,h范围
        # 按rpn_fg_scores大小截取前n个roi进行NMS
        # 截取前m个roi返回
        rois = []
        for i in range(b):
            # (n_rois, 4) -> (x1,y1,x2,y2)
            roi = self.roi_creator(rpn_locs[i].detach(),
                                   rpn_fg_scores[i].detach(),
                                   anchor, img_size, scale=scale)
            rois.append(roi)
        
        rois = torch.cat(rois, dim=0)

        return rpn_locs, rpn_scores, rois, anchor
    
    def generate_anchor_base(self):
        """
        生成基础的9种长宽、面积比的anchor坐标 坐标形式(x1,y1,x2,y2)
        """
        ratio_num = len(self.ratios)
        scale_num = len(self.anchor_scales)

        py = self.feat_stride / 2.
        px = self.feat_stride / 2.

        # (n_anchors,4) -> (xmin,ymin,xmax,ymax)
        self.anchor_base = torch.zeros((ratio_num * scale_num, 4), dtype=torch.float32, device=cfg.device)
        # 遍历每种纵横比、每种尺寸
        for i in range(ratio_num):
            for j in range(scale_num):
                # 16 * 尺寸 * 纵横比^.5 -> h
                # 16 * 尺寸 * 1/纵横比^.5 -> w
                h = self.feat_stride * self.anchor_scales[j] * torch.sqrt(self.ratios[i])
                w = self.feat_stride * self.anchor_scales[j] * torch.sqrt(1. / self.ratios[i])

                # 每个特征点基于box中心进行生成anchor
                index = i * len(self.anchor_scales) + j
                self.anchor_base[index, 0] = px - w / 2.
                self.anchor_base[index, 1] = py - h / 2.
                self.anchor_base[index, 2] = px + w / 2.
                self.anchor_base[index, 3] = py + h / 2.

    def create_anchor_all(self, feature_h, feature_w):
        """
        生成相对于整张图片来说的全部anchors
        :param feature_h: 经过特征提取网络之后的features的高
        :param feature_w: 经过特征提取网络之后的features的宽
        :return: 布满整张图片的所有anchors
        """

        # 0 ~ h*16，步长为16 -> [0,16,32,...,(h-1)*16] (h,)
        # 0 ~ w*16，步长为16 -> [0,16,32,...,(w-1)*16] (w,)
        shift_y = torch.arange(0, feature_h * self.feat_stride, self.feat_stride, dtype=torch.float32,device=cfg.device)
        shift_x = torch.arange(0, feature_w * self.feat_stride, self.feat_stride, dtype=torch.float32,device=cfg.device)

        # (h,w) (h,w)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        # (h*w) <stack> (h*w) -> (h*w,2) -> (h*w,4)
        # 表示xmin,ymin,xmax,ymax分别需要平移的坐标
        shift = torch.stack((torch.flatten(shift_x), torch.flatten(shift_y)), 1).repeat(1, 2)

        # (9,4) + (h*w,1,4) -> (1,9,4) + (h*w,9,4) -> (h*w,9,4)
        # 通过相加实现平移
        anchor = self.anchor_base + shift[:, None, :]
        # (h*w,9,4) -> (h*w*9,4) 即 (n_anchors,4)
        anchor = anchor.reshape((-1, 4))

        return anchor
