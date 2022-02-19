import os
import torch
import visdom
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import ListDataset
from model.faster_rcnn import FasterRCNN
from model.util.eval import Eval

if __name__ == '__main__':
    # 准备训练与验证数据
    trainset = ListDataset(cfg, split='train', is_train=True)
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testset = ListDataset(cfg, split='test', is_train=False)
    test_dataloader = DataLoader(testset, batch_size=1, num_workers=0)
    # 加载模型与权重
    model = FasterRCNN().to(cfg.device)

    cur_epoch = 0

    if cfg.load_model:
        paths = os.listdir('weights')
        if len(paths) != 0:
            last_epoch = max(list(map(lambda path: int(path.split('_')[0].split('epoch')[1]), paths)))
            cur_epoch = last_epoch + 1
            last_path = list(filter(lambda path: path.startswith(f'epoch{last_epoch}'), paths))[0]
            if cfg.device == 'cpu':
                model.load(torch.load(f'weights/{last_path}', map_location='cpu')['model'])
            else:
                model.load(torch.load(f'weights/{last_path}')['model'])

    # 创建visdom可视化端口
    vis = visdom.Visdom(env='Faster RCNN')
    vis.line([0], [0], win='Train Loss', opts=dict(title='Train Loss'))
    vis.line([0], [0], win='mAP', opts=dict(title='mAP'))

    total_loss, n_train = 0, 0

    for epoch in range(cur_epoch, cfg.epoch):
        model.train()
        for index, (img, target_box, target_label, scale) in enumerate(tqdm(dataloader)):
            cur_index = epoch * len(dataloader) + index + 1
            scale = scale.to(cfg.device)
            img, target_box, target_label = img.to(cfg.device).float(
            ), target_box.to(cfg.device), target_label.to(cfg.device)

            loss = model(img, target_box, target_label, scale)
            total_loss += loss.item()

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            n_train += 1
            vis.line(X=[cur_index], Y=[total_loss/n_train], win='Train Loss',
                     update='append', opts=dict(title='Train Loss'))

        model.eval()

        # 每个Epoch计算一次mAP
        # ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        eval_result = Eval(test_dataloader, model)
        eval_map = eval_result[2].mean()

        print("Epoch %d/%d ---- new-mAP:%.4f" % (epoch, cfg.epoch, eval_map))

        # 绘制mAP和Loss曲线
        vis.line(X=[epoch], Y=[eval_map], win='mAP',
                 update='append', opts=dict(title='mAP'))

        # 保存最佳模型
        if epoch % 1 == 0:
            save_name = f'epoch{epoch}_map{eval_map}.pth'
            best_path = model.save(save_name)
        
        # 调整学习率
        if epoch == 9:
            model.scale_lr(cfg.lr_decay)
