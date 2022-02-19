import torch

from config import cfg
from model.util.box_tool import box2loc, box_iou, unmap, get_inside_index


class AnchorTargetCreator(object):
    """
    生成Anchor对应的Target
    Anchor是本来就存在的
    在训练RPN网络时，需要准备一些target_loc和target_label来和rpn网络生成rpn_loc和rpn_label来计算rpn网络的损失
    先从基础anchor中提取出内部框的索引inside_index，并修改基础anchor为内部anchor
    计算anchor与target_boxes的iou值，返回每个anchor与target_boxes的最大iou索引argmax_ious以及满足iou条件的不超过256个的正负样本索引
    创建一个默认值为-1的长度为len(anchor)的基础label,并根据前面得到的正负样本的索引,分别给label中正样本赋1负样本赋0
    根据target_box与前一步得到的argmax_ious求出与target_box最匹配的anchor,然后求出真实修正系数
    最后将在内部anchor上求出的内部loc与内部label映射回基础loc(除内部loc外默认为0)与基础label(除内部label外默认为-1)上
    最终返回基础loc与基础label
    """

    def __init__(self):
        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio = 0.5

    def __call__(self, target_box, anchor, img_size):
        """
        :param target_box: 真实标注框
        :param anchor: 初步生成的基础框
        :param img_size: 图片输入尺寸
        :return: rpn_box到target_box的真实矫正系数(针对所有anchor的),和是否含有目标的label(1=有, 0=无, -1=忽略)
        """

        img_h, img_w = img_size

        # 筛选不超出图片边界的anchor
        inside_index = get_inside_index(anchor, img_h, img_w)
        anchor_inside = anchor[inside_index]

        # 根据anchor和真实标注框
        # 筛选每个anchor与所有标注框的iou最大值的索引 (n_anchors,)
        # 以及标签：1为正样本，0为负样本，-1为忽略 (n_anchors,)
        argmax_ious, label = self._create_label(anchor_inside, target_box)

        # 计算所有内部框到每个框最匹配(iou)的target_box的修正系数
        loc = box2loc(anchor_inside, target_box[argmax_ious])

        # 将内部筛选后的label和loc映射到原始完整的label和loc
        label = unmap(label, anchor, inside_index)
        loc = unmap(loc, anchor, inside_index)

        return loc, label

    def _create_label(self, anchor, target_box):
        """
        :param anchor: 图片内部的内部anchors (n_anchors,4)
        :param target_box: 真实标注框 (n_targets,4)
        :return: 
          argmax_ious: 每个anchor与所有target_box的最大iou索引
          label: 所有anchor对应的标签1是正样本，0是负样本，-1表示忽略
        """

        # 创建anchor数量的label，值为-1，表示忽略
        # (n_anchors,)
        label = -torch.ones((len(anchor),), dtype=torch.int32, device=cfg.device)

        # 计算内框和标注框的iou
        ious = box_iou(anchor, target_box)
        # 计算每个anchor与target_boxes iou的最大值和对应索引
        max_ious, argmax_ious = ious.max(dim=1)

        # 计算每个target_box与所有anchors的最大iou
        gt_max_ious, _ = ious.max(dim=0)
        # 这里gt_max_ious的最大值可能不是唯一的,所以需要把全部的最大iou都找出来作为target_anchor的索引
        gt_argmax_ious = torch.nonzero(torch.eq(ious, gt_max_ious))

        # 首先分配负样本,以便正样本可以覆盖它们(某些情况下最大IOU可能小于neg_iou_thresh)
        # 负样本: iou小于neg_iou_thresh的anchor
        label[max_ious < self.neg_iou_thresh] = 0

        # 每个target_box最大iou的anchor或大于iou阈值的anchor均设为正样本
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 如果正样本超过理论值则随机丢弃多余的正样本
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = torch.nonzero(label == 1).squeeze()
        pos_num = pos_index.numel()
        if pos_num > n_pos:
            disable_index = pos_index[torch.randperm(pos_num)[:pos_num - n_pos]]
            label[disable_index] = -1

        # 如果负样本超过理论值则随机丢弃多余的负样本
        n_neg = self.n_sample - torch.sum(torch.eq(label, 1))
        neg_index = torch.nonzero(label == 0).squeeze()
        neg_num = neg_index.numel()
        if neg_num > n_neg:
            disable_index = neg_index[torch.randperm(neg_num)[
                :neg_num - n_neg]]
            label[disable_index] = -1

        return argmax_ious, label
