import numpy as np
import torch

from config import cfg

def loc2box(src_box, loc):
    """
    已知预测框和修正参数,求目标框
    利用平移和尺度放缩修正P_box以得到^G_box 然后将^G_box与G_box进行计算损失
    参考 https://blog.csdn.net/zijin0802034/article/details/77685438/
    ^G_box_x = p_w*t_x + p_x`
    ^G_box_y = p_h*t_y + p_y`
    ^G_box_w = p_w*exp(t_w)`
    ^G_box_h = p_h*exp(t_h)`
    : param src_bbox (array): `p_{xmin}, p_{ymin}, p_{xmax}, p_{ymax}`. (R,4)
    : param loc (array): `t_x, t_y, t_w, t_h`.  (R,4)
    return: dst_box 修正后的^G_box -> (R, 4)，R是预测的框数量，第二维度的数据形式与src_bbox相同
    """
    # (x1,y1,x2,y2) -> (x,y,w,h)
    src_w = src_box[:, 2] - src_box[:, 0]
    src_h = src_box[:, 3] - src_box[:, 1]
    src_x = src_box[:, 0] + 0.5 * src_w
    src_y = src_box[:, 1] + 0.5 * src_h

    dst_x = loc[:, 0] * src_w + src_x
    dst_y = loc[:, 1] * src_h + src_y
    dst_w = torch.exp(loc[:, 2]) * src_w
    dst_h = torch.exp(loc[:, 3]) * src_h

    # (x,y,w,h) -> (x1,y1,x2,y2)
    dst_box = torch.zeros(loc.shape, dtype=loc.dtype, device=cfg.device)
    dst_box[:, 0] = dst_x - 0.5 * dst_w
    dst_box[:, 1] = dst_y - 0.5 * dst_h
    dst_box[:, 2] = dst_x + 0.5 * dst_w
    dst_box[:, 3] = dst_y + 0.5 * dst_h

    return dst_box

def box2loc(src_box, dst_box):
    """
    已知真实框和预测框求出其修正参数
    :param src_box: shape -> (R, 4) x1y1x2y2.
    :param dst_box: 同上
    :return: 修正系数 shape -> (R, 4)
    """
    src_w = src_box[:, 2] - src_box[:, 0]
    src_h = src_box[:, 3] - src_box[:, 1]
    src_x = src_box[:, 0] + 0.5 * src_w
    src_y = src_box[:, 1] + 0.5 * src_h

    dst_w = dst_box[:, 2] - dst_box[:, 0]
    dst_h = dst_box[:, 3] - dst_box[:, 1]
    dst_x = dst_box[:, 0] + 0.5 * dst_w
    dst_y = dst_box[:, 1] + 0.5 * dst_h

    dx = (dst_x - src_x) / (src_w + 1e-8)
    dy = (dst_y - src_y) / (src_h + 1e-8)
    dw = torch.log(dst_w / (src_w + 1e-8))
    dh = torch.log(dst_h / (src_h + 1e-8))

    loc = torch.stack((dx, dy, dw, dh), dim=1)
    return loc

def box_iou(box_a, box_b):
    # 计算 N个box与M个box的iou需要使用到numpy的广播特性
    # tl为交叉部分左上角坐标最大值, tl.shape -> (N,M,2)
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    # br为交叉部分右下角坐标最小值
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    # 第一个axis是指定某一个box内宽高进行相乘,第二个axis是筛除那些没有交叉部分的box
    # 这个 < 和 all(axis=2) 是为了保证右下角的xy坐标必须大于左上角的xy坐标,否则最终没有重合部分的box公共面积为0
    area_i = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2)
    # 分别计算bbox_a,bbox_b的面积,以及最后的iou
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], dim=1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], dim=1)
    iou = area_i / (area_a[:, None] + area_b - area_i)
    return iou

def box_iou_numpy(box_a, box_b):
    # 计算 N个box与M个box的iou需要使用到numpy的广播特性
    # tl为交叉部分左上角坐标最大值, tl.shape -> (N,M,2)
    lt = np.maximum(box_a[:, np.newaxis, :2], box_b[:, :2])
    # br为交叉部分右下角坐标最小值
    rb = np.minimum(box_a[:, np.newaxis, 2:], box_b[:, 2:])
    # 第一个axis是指定某一个box内宽高进行相乘,第二个axis是筛除那些没有交叉部分的box
    # 这个 < 和 all(axis=2) 是为了保证右下角的xy坐标必须大于左上角的xy坐标,否则最终没有重合部分的box公共面积为0
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    # 分别计算bbox_a,bbox_b的面积,以及最后的iou
    area_a = np.prod(box_a[:, 2:] - box_a[:, :2], axis=1)
    area_b = np.prod(box_b[:, 2:] - box_b[:, :2], axis=1)
    iou = area_i / (area_a[:, np.newaxis] + area_b - area_i)
    return iou

def unmap(data, anchor, inside_index):
    if len(data.shape) == 1:
        # 如果是对label进行映射,则默认值为-1(忽略样本)
        ret = torch.full_like(anchor[:,0],fill_value=-1, dtype=torch.int32)
        ret[inside_index] = data
    else:
        # 如果是对loc进行映射,则默认值为0(忽略样本)
        # ret = np.zeros((n_anchor, 4), dtype=np.float32)
        ret = torch.zeros_like(anchor)
        ret[inside_index] = data
    return ret

def get_inside_index(anchor, h, w):
    """
    筛选在图片内的anchor，即排除超出图片边界的anchor
    """
    index_inside = torch.nonzero(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= w) &
        (anchor[:, 3] <= h)
    ).squeeze()
    return index_inside