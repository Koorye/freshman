class Config:
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    device = 'cpu'

    # RPN阶段的NMS阈值
    nms_rpn = .7

    # 测试阶段的ROI阈值
    nms_roi = .3    

    # 计算loss时rpn与roi所占的比重
    rpn_sigma = 3.
    roi_sigma = 1.

    # 权重衰减系数
    weight_decay = .0005
    # 每个epoch学习率下降的倍数
    lr_decay = .1
    # 初始学习率
    lr = 1e-3
    epoch = 15
    # 是否使用SGD优化器
    use_sgd = True

    # 数据所在目录
    train_dir = 'data/VOC2012'
    val_dir = 'data/VOC2012'

    # 图片最大与最小输入长宽尺寸
    max_size = 1000
    min_size = 600

    # 是否加载历史模型
    load_model = True


cfg = Config()
