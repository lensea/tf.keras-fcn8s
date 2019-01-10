import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import ready_data
 

data_dir = "/home/ispr/data/solder_seg/"
classes = ['background', 'solder']
 
colormap = [[192, 192, 192], [128, 128, 64]]
 
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
 
    #save_path = '/home/ispr/data/VOCdevkit/VOC2012/Label'
    #ave_path = os.path.join(save_path, label_name)  # 确定保存路径及名称
    save_path = label_name
    label_single.save(save_path)
 
 
val_file_path = '/home/ispr/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'  # 文件名存放路径
label_file_path = '/home/ispr/data/VOCdevkit/VOC2012/SegmentationClass'  # 原label存放路径



train_records,val_records = ready_data.get_imgs_path(data_dir)

for train_data in train_records:
    anno_path = train_data["annotation"]
    print(anno_path)
    change_label(anno_path,anno_path)
    

for val_data in val_records:
    anno_path = val_data["annotation"]
    print(anno_path)
    change_label(anno_path,anno_path)
