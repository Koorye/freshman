import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.ops import batched_nms
from config import cfg

from model.roi_head import RoIHead
from model.rpn import RPN
from model.util.anchor_target_creator import AnchorTargetCreator
from model.util.box_tool import loc2box
from model.util.loss import fast_rcnn_loc_loss
from model.util.proposal_target_creator import ProposalTargetCreator


def decom_vgg16():
    """
    截取VGG16模型，获取提取器和分类器
    """
    model = vgg16(pretrained=True)
    # 截取vgg16的前30层网络结构,因为再往后的就不需要
    # 31层为maxpool再往后就是fc层
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    # 删除最后一层以及两个dropout层
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 冻结前4层的卷积层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)

    return features, classifier


class FasterRCNN(nn.Module):
    def __init__(self) -> None:
        super(FasterRCNN, self).__init__()
        self.n_class = len(cfg.classes) + 1

        # 获取特征提取网络和分类网络
        self.extractor, self.classifier = decom_vgg16()

        self.rpn = RPN()
        self.head = RoIHead(n_class=self.n_class,
                            classifier=self.classifier)

        self.nms_thresh = cfg.nms_roi
        self.rpn_sigma = cfg.rpn_sigma
        self.roi_sigma = cfg.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.optimizer = self.get_optim()

        self.mean = torch.tensor(
            (0., 0., 0., 0.), device=cfg.device).repeat(self.n_class)[None]
        self.std = torch.tensor(
            (.1, .1, .2, .2), device=cfg.device).repeat(self.n_class)[None]

        self.score_thresh = .05

    def forward(self, x, target_boxes=None, target_labels=None, scale=1.):
        """
        : param x: 原图
        : param target_boxes: 目标框
        : param target_labels: 目标框类别
        : param scale: 原始图片预处理的缩放比
        """

        img_size = x.shape[2:]
        # 提取特征图
        features = self.extractor(x)

        # 经过RPN提取回归值和分数，并返回NMS后的预测框
        # 此处roi理论为2000个
        # (b,h*w*9,4) (b,h*w*9,2) (n_rois,4) (h*w*9,4)
        # n_rois≈2000, n_anchors≈20000
        rpn_locs, rpn_scores, rois, anchor = self.rpn(
            features, img_size, scale)

        # 如果是测试阶段，直接将特征图和RPN得到roi送入roi_head得到预测结果
        # 测试阶段返回的roi数量理论值为300
        if not self.training:
            roi_locs, roi_scores = self.head(features, rois)
            return roi_locs, roi_scores, rois

        # batch为1，故只取第一个元素
        # (n_targets,4)
        target_box = target_boxes[0]
        # (n_targets,)
        target_label = target_labels[0]
        # (h*w*9,2)
        rpn_score = rpn_scores[0]
        # (h*w*9,4)
        rpn_loc = rpn_locs[0]
        roi = rois

        # 筛选相应数量的正负样本，并计算位置回归的修正系数和标签
        # 此处得到的修正系数和标签是roi最接近的目标框的类别和偏差
        # 此处理论上筛选128个roi，包括正负样本
        # (128,4) (128,4) (128,)
        sample_roi, gt_head_loc, gt_head_label = self.proposal_target_creator(
            roi, target_box, target_label)

        # 对每个样本提取回归系数和分数
        # 理论值 (128,n_class*4), (128,n_class)
        head_loc, head_score = self.head(features, sample_roi)

        # 筛选相应数量的正负样本供RPN网络计算损失
        # (h*w*9,4) (h*w*9,) label为1,0,-1，-1表示忽略
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            target_box, anchor, img_size)

        # 使用long类型因为下面cross_entropy方法需要
        gt_rpn_label = gt_rpn_label.long()
        # 计算RPN回归损失
        rpn_loc_loss = fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        # 计算RPN分类损失，忽略label为-1
        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label.to(cfg.device), ignore_index=-1)

        # 计算roi_head的回归和分类损失
        # roi数量，理论为 (128,)
        n_sample = head_loc.shape[0]
        # (128,n_class,4) 128为参与训练的理论样本数
        head_loc = head_loc.reshape(n_sample, -1, 4)

        # 转成long以计算cross_entropy
        gt_head_label = gt_head_label.long()

        # 获取sample_roi中每个roi所对应的修正系数loc
        # 当然，正样本和负样本所获取的loc情况是不同的
        # 正样本:某个roi中类别概率最大的那个类别的loc
        # 负样本:永远是第1个loc(背景类index为0)
        # 第0维: (128,) [0,1,...,127]
        # 第1维: (128,) 每个元素为每个roi的最接近的标注框的类别 -> 0~20
        # 从而提取每个roi对应类别的修正系数loc
        # (128,4)
        head_loc = head_loc[torch.arange(
            n_sample).long().to(cfg.device), gt_head_label]

        # 计算roi_head的回归损失
        roi_loc_loss = fast_rcnn_loc_loss(
            head_loc, gt_head_loc, gt_head_label, self.roi_sigma)
        # 计算roi_head的分类损失
        roi_cls_loss = F.cross_entropy(
            head_score, gt_head_label.to(cfg.device))

        losses = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        return losses

    @torch.no_grad()
    def predict(self, imgs, sizes=None):
        """
        该方法在非训练阶段的时候使用
        :param imgs: 一个batch的图片
        :param sizes: batch中每张图片的输入尺寸
        :return: 返回所有一个batch中所有图片的坐标,类,类概率值 三个值都是list型数据,里面包含的是numpy数据
        """

        boxes, labels, scores = [], [], []

        # 因为batch_size为1所以这个循环就只循环一次
        for img, size in zip([imgs], [sizes]):
            # 图片输入宽度 / 原图宽度 -> 缩放比
            scale = img.shape[3] / size[1]

            # 根据原图和缩放比计算roi及其回归系数和分数
            # 理论值为 (300,n_class*4) (300,n_class) (300,4)
            roi_locs, roi_scores, roi = self(img, scale=scale)

            # roi_head训练时loc进行了标准化，因此预测时要对loc进行逆标准化以恢复真实的回归系数loc
            # *chenyun版本的代码中是有对训练阶段的roi_locs进行标准化的,然后再在非训练状态下进行逆向标准化
            roi_locs = (roi_locs * self.std + self.mean)

            # (300,n_class*4) -> (300,n_class,4)
            roi_locs = roi_locs.view(-1, self.n_class, 4)
            # (300,4) -> (300,1,4) -> (300,n_class,4)
            roi = roi.view(-1, 1, 4).expand_as(roi_locs)

            # 将坐标放缩回原始尺寸
            # *chenyun版本是将缩放这一步放到修正坐标之前
            # (300,n_class,4) -> (300*n_class,4)
            pred_boxes = loc2box(roi.reshape(-1, 4),
                                 roi_locs.reshape(-1, 4)) / scale

            # (300*n_class,4)
            pred_boxes = pred_boxes
            # (300*n_class,4) -> (300,n_class,4)
            pred_boxes = pred_boxes.view(-1, self.n_class, 4)

            # 限制预测框的坐标范围
            pred_boxes[:, :, 0::2].clamp_(min=0, max=size[1])
            pred_boxes[:, :, 1::2].clamp_(min=0, max=size[0])

            # 对roi_head网络预测的每类进行softmax处理
            pred_scores = F.softmax(roi_scores, dim=1)

            # 每张图片的预测结果 (m为预测目标的个数)
            # 跳过cls_id为0的pred_bbox与pred_scores，因为它是背景类
            # (m,4) (m,) (m,)
            pred_boxes, pred_label, pred_score = self._suppress(
                pred_boxes[:, 1:, :], pred_scores[:, 1:])

            boxes.append(pred_boxes)
            labels.append(pred_label)
            scores.append(pred_score)

        return boxes, labels, scores

    def _suppress(self, pred_boxes, pred_scores):
        """
        对Faster-RCNN网络最终预测的box与score进行score筛选以及nms
        1.循环所有的标注类,在循环中过滤出那些类得分在self.score_thresh之下的cls_box与cls_score。
        2.随后进行batch_nms.随后就将经过nms筛选的box,score以及新建的label分别整合到一起并返回这三个值
        :param pred_boxes: rpn网络提供的roi,经过roi_head网络提供的loc再次修正得到的 torch.Size([300, self.n_class, 4])
        :param pred_scores: roi_head网络提供各个类的置信度 torch.Size([300, self.n_class])
        :return: faster-rcnn网络预测的目标框坐标,种类,种类的置信度
        """

        # 生成20类id，不包含背景，[0,1,...,19] -> (20,) -> (1,20)
        # 之后重复为所有box生成相同的张量 -> (300,20) 300为NMS筛选后的box理论值
        cls_ids = torch.arange(
            self.n_class-1)[None].repeat(pred_boxes.shape[0], 1)

        # 过滤得分低于self.score_thresh的score
        score_keep = pred_scores > self.score_thresh
        # (300,n_class,4)
        pred_boxes = pred_boxes[score_keep].reshape(-1, 4)
        # (300,n_class,)
        pred_scores = pred_scores[score_keep].flatten()
        # (300,)
        cls_ids = cls_ids[score_keep].flatten()

        # 对pred_boxes进行NMS过滤
        # *这里使用batch_nms速度会快一些，A100上 35/sec -> 41/sec
        keep = batched_nms(pred_boxes, pred_scores,
                           cls_ids, self.nms_thresh)
        box = pred_boxes[keep].cpu().numpy()
        score = pred_scores[keep].cpu().numpy()
        label = cls_ids[keep].cpu().numpy()

        return box, label, score

    def get_optim(self):
        # 获取梯度更新的方式,以及 放大 对网络权重中 偏置项 的学习率
        lr = cfg.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value],
                                'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr,
                                'weight_decay': cfg.weight_decay}]
        if cfg.use_sgd:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(params)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    def save(self, save_path):
        save_dict = {
            'model': self.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
        }
        torch.save(save_dict, f'weights/{save_path}')
        return save_path

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
