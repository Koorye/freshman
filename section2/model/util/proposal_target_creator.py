import torch

from config import cfg
from model.util.box_tool import box2loc, box_iou


class ProposalTargetCreator(object):
    """
    生成Proposal对应的Target
    Proposal是由rpn提供的
    先把target_box并入到roi中去,计算roi和target_box的iou,并获取每个roi和target_box的最大iou索引roi_argmaxiou 以及最大iou值roi_maxiou
    将 target_box的label通过roi_argmaxiou赋值给roi,并+1(因为0为背景类)
    根据正负样本的iou阈值得出iou中的正负样本索引(共128个)pos_index,neg_index(组成keep_index).并随机舍弃多余的样本
    从众多roi_label中根据keep_index挑选出正负样本的label,并令负样本的label为0
    并根据keep_index从roi中挑选出正负样本的roi,然后由target_box根据roi_argmaxiou和keep_index得出len(keep_index)个
    与roi正负样本对应的target_box 即target_box[roi_argmax_targets[keep_index]]
    最后根据正负样本的roi与其对应的target_box计算修正系数,然后减均值除以方差
    参数:
       n_sample (int): 每张图片理论上采集的样本数.
       pos_ratio (float): 正样本比例
       pos_iou_thresh (float): 达到正样本的IOU阈值
       neg_iou_thresh_hi (float): IOU在此区间内的属于负样本 [neg_iou_thresh_lo, neg_iou_thresh_hi).
       neg_iou_thresh_lo (float): 同上.
    """

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # 注意:在py-faster-rcnn中 该值默认为0.1
        self.loc_normalize_mean = torch.tensor(
            (0., 0., 0., 0.), device=cfg.device)
        self.loc_normalize_std = torch.tensor(
            (.1, .1, .2, .2), device=cfg.device)

    def __call__(self, roi, target_box, label):
        """
        :param roi: rpn网络提供的roi,理论上训练阶段为(2000,4) 测试阶段为(300,4)
        :param target_box: 真实标注物体的坐标 (n,4) n为一张图片中真实标注物体的个数
        :param target_label: 真实标注物体的种类索引 (n,)
        :return: 128(理论)个正负样本roi的坐标, roi与target_box修正系数, 128(理论)个正负样本roi的label
        """

        # 这里将target_box也并入到roi中去，这里可以将roi当做RPN阶段的"anchor"，只不过是动态的
        # 将target_box添加到roi中是为了更好的收敛ROIHead网络而做的操作
        # 训练初期RPN阶段提供的roi不管是从质量还是数量来说都不高,这里算是弥补了一些
        roi = torch.cat((roi, target_box), dim=0)

        # 计算正样本的数量
        pos_roi_per_image = round(self.n_sample * self.pos_ratio)

        # 计算ROI和真实坐标的iou
        iou = box_iou(roi, target_box)

        # 每个roi和target_boxes的最大iou和索引
        max_iou, gt_assignment = iou.max(dim=1)

        # 将所有种类索引+1(所有label>=1,0为下面的负样本所准备的),并且此时为所有roi赋予label
        # 值为与其iou最大的target_box的label值
        # 0作为背景类
        gt_roi_label = label[gt_assignment] + 1

        # 获取那些IOU大于pos_iou_thresh的roi索引
        pos_index = torch.nonzero(max_iou >= self.pos_iou_thresh)
        pos_num = pos_index.numel()
        # 取理论要达到的正样本数量和过滤后的正样本数量的最小值作为最终的正样本数量
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_num))
        # 随机抽取上述数量的正样本下标
        if pos_num > 0:
            pos_index = pos_index[torch.randperm(
                pos_num)[:pos_roi_per_this_image]]

        # 获取iou在[neg_iou_thresh_lo, neg_iou_thresh_hi)区间的roi索引
        neg_index = torch.nonzero((max_iou < self.neg_iou_thresh_hi)
                                  & (max_iou >= self.neg_iou_thresh_lo))

        # 获取过滤后负样本的数量
        neg_num = neg_index.numel()
        # 计算每张图片中理论上的负样本个数
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        # 取理论负样本数量和过滤后负样本数量的最小值作为最终的负样本数量
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_num))
        # 随机抽取上述数量的负样本下标
        if neg_num > 0:
            neg_index = neg_index[torch.randperm(
                neg_num)[:neg_roi_per_this_image]]

        # 将正负样本的roi索引合并到一起
        keep_index = torch.cat((pos_index, neg_index)).squeeze()
        # 从所有roi中挑选出正负样本的label
        gt_roi_label = gt_roi_label[keep_index]

        # 将负样本的label置为0
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        # 计算修正系数
        # 即roi和其最大iou的target_box的loc
        gt_roi_loc = box2loc(sample_roi, target_box[gt_assignment[keep_index]])

        # 这里的减均值除以方差以及非训练阶段roi网络最后出来的roi_loc还要乘方差加均值
        gt_roi_loc = ((gt_roi_loc - self.loc_normalize_mean)
                      / self.loc_normalize_std)

        return sample_roi, gt_roi_loc, gt_roi_label
