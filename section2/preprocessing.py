import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

from config import cfg

seed = 999
root = cfg.train_dir

def has_object(file_path):
    bbox = []
    anno = ET.parse(file_path)
    for obj in anno.findall('object'):
        if int(obj.find('difficult').text) == 1:
            continue
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
                int(float(bndbox_anno.find(tag).text)) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
    return len(bbox) != 0 


def train_test_split(train_rate=.8, seed=None):
    """
    根据所有图片随机切分训练集和测试集
    """
    
    if seed is not None:
        random.seed(seed)

    files = [x for x in os.listdir(os.path.join(root, 'Annotations'))]
    files = list(filter(lambda file: has_object(os.path.join(root, 'Annotations', file)), files))
    
    train_files = random.sample(files, int(train_rate*len(files)))
    test_files = [x for x in files if x not in train_files]

    train_files = list(map(lambda file: file.split('.')[0], train_files))
    test_files = list(map(lambda file: file.split('.')[0], test_files))

    with open(os.path.join(root, 'train.txt'), 'w') as f:
        pbar = tqdm(train_files, total=len(train_files), desc='生成训练文件')
        for file in pbar:
            f.write(file+'\n')
            
    with open(os.path.join(root, 'test.txt'), 'w') as f:
        pbar = tqdm(test_files, total=len(test_files), desc='生成测试文件')
        for file in pbar:
            f.write(file+'\n')

if __name__ == '__main__':
    train_test_split()
