import colorsys
import cv2
import numpy as np
import os
import torch
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from config import cfg
from dataset import ImageFolder
from model.faster_rcnn import FasterRCNN

device = torch.device(cfg.device)
model = FasterRCNN().to(device)
paths = os.listdir('weights')
if len(paths) != 0:
    last_epoch = max(list(map(lambda path: int(path.split('_')[0].split('epoch')[1]), paths)))
    cur_epoch = last_epoch + 1
    last_path = list(filter(lambda path: path.startswith(f'epoch{last_epoch}'), paths))[0]
    if cfg.device == 'cpu':
        model.load_state_dict(torch.load(f'weights/{last_path}', map_location='cpu')['model'])
    else:
        model.load(f'weights/{last_path}')
model.eval()

# 为每个类名配置不同的颜色
cls_name = cfg.classes
hsv_tuples = [(x / len(cls_name), 1., 1.)for x in range(len(cls_name))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

camera = cv2.VideoCapture(0)
while True:
    ret, img = camera.read()
    if ret is False:
        break

    cv2.imwrite('tmp/tmp.png', img)
    img_set = ImageFolder('tmp')
    path, img, size = img_set[0]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_boxes_, pred_labels_, pred_scores_ = model.predict(img, size)
        img_detection = [pred_boxes_, pred_labels_, pred_scores_]

    PIL_img = Image.open(path)
    w,h = PIL_img.size
    content_font = ImageFont.truetype(font='FiraMono-Medium.otf', size=16)
    thickness = (w + h) // 600
    draw = ImageDraw.Draw(PIL_img)

    # 对单张图片进行处理
    for box,label,score in zip(*img_detection):
        box, label, score = box.tolist(), label.tolist(), score.tolist()
        # 对一张图片中预测的每个box进行处理
        for (x1,y1,x2,y2),l,s in zip(box,label,score):
            if s > .5:
                # 对预测出的坐标进行缩放
                x1, y1, x2, y2 = x1*(w/size[1]), y1*(w/size[1]), x2*(w/size[1]), y2*(w/size[1])
                content = '{} {:.2f}'.format(cfg.classes[l], s)
                label_w, label_h = draw.textsize(content,content_font)
                for i in range(thickness):
                    draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=colors[l])
                    draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[l])
                    draw.text((x1, y1 - label_h), content, fill=(0, 0, 0),font=content_font)
        PIL_img = np.array(PIL_img)[...,::-1]
        if len(box) > 0:
            cv2.imshow('result',PIL_img)
    
    c = cv2.waitKey(1)
    if c == 27:
        break

camera.release()
cv2.destroyAllWindows()
    
