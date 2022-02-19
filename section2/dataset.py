from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from PIL import Image
import torch.nn.functional as F
import random
import glob
import xml.etree.ElementTree as ET
import os


class ListDataset(Dataset):
    def __init__(self, cfg, split='trainval', is_train=False):
        self.data_dir = cfg.train_dir if is_train else cfg.val_dir
        id_list_file = os.path.join(cfg.train_dir, f'{split}.txt')
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.ignore_difficult = is_train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.ToTensor = T.ToTensor()
        self.cls_list = cfg.classes
        self.min_size = cfg.min_size
        self.max_size = cfg.max_size

    def __getitem__(self, i):
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            if self.ignore_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(float(bndbox_anno.find(tag).text)) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.cls_list.index(name))
        
        box = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')

        img = self.ToTensor(Image.open(img_file))  # 自带归一化
        _, in_h, in_w = img.shape
        # preprocess img 缩放到最小比例,这样最终长和宽都能放缩到规定的尺寸
        scale1 = self.min_size / min(in_h, in_w)
        scale2 = self.max_size / max(in_h, in_w)
        scale = min(scale1, scale2)
        out_h, out_w = round(in_h * scale), round(in_w * scale)
        img = F.interpolate(img.unsqueeze(0), size=(out_h, out_w), mode="bilinear",align_corners=True).squeeze(0)
        img = self.normalize(img).numpy()
        if self.is_train:
            box *= scale
            # 需要将后续返回的img替换为img.copy()
            # 因为给定numpy数组的某些步幅为负(img[:, ::-1, :]和img[:, :, ::-1])。numpy官方当前不支持此功能。
            img, params = random_flip(img, x_random=True, return_param=True)
            box = flip_bbox(box, (out_h, out_w), x_flip=params['x_flip'])
            return img.copy(), box.copy(), label.copy(), scale
        else:
            return img, (in_h, in_w), box, label, difficult

    def __len__(self):
        return len(self.ids)

# 为测试图片准备
class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob("%s/*.*" % folder_path)

    def __getitem__(self, index):
        img_path = self.files[index]
        # 这里使用convert是防止使用png图片或其他格式时会有多个通道而引起的报错,
        img = T.ToTensor()(Image.open(img_path).convert('RGB'))
        
        _, in_h, in_w = img.shape
        # 缩放到最小比例,这样最终长和宽都能放缩到规定的尺寸
        scale1 = 600 / min(in_h, in_w)
        scale2 = 1000 / max(in_h, in_w)
        scale = min(scale1, scale2)
        img = F.interpolate(img.unsqueeze(0), size=(round(in_h * scale), round(in_w * scale)), mode="nearest").squeeze(0)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img_path, img, img.shape[1:]

    def __len__(self):
        return len(self.files)


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    这里需要说明一下,box是如何进行水平或竖直翻转的
    拿x方向水平翻转举例:根据对称性,翻转后box右边框的x坐标等于图像整体宽度减去翻转前box左边框的x坐标.
                 同理:翻转后box左边框的x坐标等于图像整体宽度减去翻转前box右边框的x坐标.
    y方向竖直翻转同理.
    """
    
    H, W = size
    bbox = bbox.copy()

    if x_flip:
        x_max = W - bbox[:, 0]
        x_min = W - bbox[:, 2]
        bbox[:, 0] = x_min
        bbox[:, 2] = x_max
    if y_flip:
        y_max = H - bbox[:, 1]
        y_min = H - bbox[:, 3]
        bbox[:, 1] = y_min
        bbox[:, 3] = y_max

    return bbox

def random_flip(img, x_random=False, y_random=False, return_param=False, copy=False):
    """
    :param img: numpy型的数组 CHW格式
    :param y_random: 是否进行竖直方向翻转
    :param x_random: 是否进行水平方向翻转
    :param return_param: 是否返回水平竖直方向翻转信息
    :param copy: 是否创建一个新的img进行翻转
    :return:取决于return_param,如果为false,则只返回翻转后的img,否则返回翻转后的img和翻转信息
    """

    x_flip, y_flip = False, False

    if x_random:
        x_flip = random.choice([True, False])
    if y_random:
        y_flip = random.choice([True, False])

    if x_flip:
        img = img[:, :, ::-1]
    if y_flip:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'x_flip': x_flip, 'y_flip': y_flip}
    else:
        return img

if __name__ == '__main__':
    from config import cfg
    trainset = ListDataset(cfg, 'ImageSets/Main/train', is_train=True)
    print(trainset[0][1:])
