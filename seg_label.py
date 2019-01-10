
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import os


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
 
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
 
# 利用下面的代码，将标注的图片转换为单通道的label图像
cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
 
 
def image2label(im):
    # 输入为标记图像的矩阵，输出为单通道映射的label图像
    data = im.astype('int32')
    idx = (data[:, :, 0]*256+data[:, :, 1])*256+data[:, :, 2]
    return np.array(cm2lbl[idx])
 
 
def change_label(label_url, label_name):
 
    label_img = load_img(label_url)
    label_img = img_to_array(label_img)
    label_img = image2label(label_img)  # 将图片映射为单通道数据
    print(np.max(label_img))
 
    label_single = Image.fromarray(label_img)
    label_single = label_single.convert('L')
 
    save_path = '/home/ispr/data/VOCdevkit/VOC2012/Label'
    save_path = os.path.join(save_path, label_name)  # 确定保存路径及名称
    label_single.save(save_path)
 
 
val_file_path = '/home/ispr/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'  # 文件名存放路径
label_file_path = '/home/ispr/data/VOCdevkit/VOC2012/SegmentationClass'  # 原label存放路径
 
with open(val_file_path, 'r') as f:
    file_names = f.readlines()
    count = 0
    for name in file_names:
        count += 1
        name = name.strip('\n')  # 去掉换行符
        label_name = name + '.png'  # label文件名
        label_url = os.path.join(label_file_path, label_name)
        print('这是第 %s 张' % count)
        print(label_url)
        change_label(label_url, label_name)

image_test = Image.open("/home/ispr/data/VOCdevkit/VOC2012/Label/2007_000123.png")
print(np.unique(image_test))
