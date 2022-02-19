import torch
from torchvision.ops import nms

from model.util.box_tool import loc2box
from config import cfg

class RoICreator:
    """
    ROICreator的主要功能如下:
    1.由rpn得出的修正系数来修正基础anchor来得到roi
    2.限制roi的坐标范围
    3.剔除那些宽高小于min_size的roi
    4.根据rpn分类卷积得出的conf来从大到小进行排序,并截取前 12000个roi(如果少于12000,如8000.那么就截取前8000.下面同理)
    5.进行nms,然后截取前 2000个roi.并最终返回这些roi(训练阶段->非训练阶段,nms前12000->6000,nms后2000->300)
    """

    def __init__(self, parent_model):
        self.parent_model = parent_model
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_size = 16
        self.nms_thresh = cfg.nms_rpn

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        :param loc: rpn网络定位卷积得出的修正参数    (16650, 4)
        :param score: rpn网络分类卷积得出的置信度    (16650,)
        :param anchor: 基础生成的anchor            (16650, 4)
        :param img_size: 网络输入的尺寸 (h,w)
        :param scale: 原始图片预处理到网络输入尺寸的倍数
        :return: 经过筛选的roi 当然也可以多返回一个pred_box为是否含有物体的目标置信度
        """

        # pre_nms, post_nms分别为NMS前后保留的样本数
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        # (n_anchors,4) -> (x1,y1,x2,y2)
        roi = loc2box(anchor, loc)

        # 限制roi的坐标范围，x在0~w范围内，y在0~h范围内
        roi[:, 0:4:2].clip_(0, img_size[1]) 
        roi[:, 1:4:2].clip_(0, img_size[0])

        # 剔除宽高小于min_size的roi
        # 由于原始图像被缩放了，因此这里的min_size也要进行对应比例的缩放
        # 例如，原图太大，宽2000 -> 1000
        # 因此 min_size 16 -> 8
        # 否则相当于在宽为2000的图像剔除宽高小于2*min_size的roi
        min_size = self.min_size * scale
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        # (n_anchors,1) -> (n_anchors,)
        keep = torch.nonzero((ws >= min_size) & (hs >= min_size)).squeeze()
        roi = roi[keep]
        score = score[keep]

        # 重新根据分类置信度从大到小进行排序然后选取n_pre_nms个进行nms
        order = score.argsort(descending=True)
        order = order[:n_pre_nms]
        roi = roi[order]
        score = score[order]

        # nms后选取前n_post_nms个
        keep = nms(roi, score, self.nms_thresh).to(cfg.device)
        keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi