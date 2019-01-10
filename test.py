#-*- coding:utf-8 -*-
import os
import sys
import argparse 
import cv2
import numpy as np 
import math
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NUM_OF_CLASSESS = 2
IMAGE_SIZE = 256 

def get_files_name(file_dir):
    files_path = []
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] ==".jpg":
                files_path.append(os.path.join(root,file))

    return files_path
def get_batch(test_imgs_path,batch_size):
    imgs_num = len(test_imgs_path)
    for idx in range(0,imgs_num,batch_size):
        idx_end = min(imgs_num,idx+batch_size)
        imgs = []      
        for i in range(idx,idx_end):      
            img = cv2.imread(test_imgs_path[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(112,112))
            imgs.append(img) 
        yield imgs
        imgs.clear()


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
 
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./pb/model.pb", type=str, help="Frozen model file to import")
    #parser.add_argument("--txt_path", default="/home/ispr/data/solder/test.txt", type=str, help="imgs path")
    parser.add_argument("--img_path", 
                        default="/home/ispr/data/solder_seg/test/13_cut_162.bmp", 
                        type=str, help="img path")
    
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)
    for op in graph.get_operations():
        print(op.name,op.values())
    

    #'''
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img,(224,224))
    #img = np.float32(img)
    #img = (img *1.0/255 -0.5)*2
    '''
    img = misc.imread(args.img_path)
    img = misc.imresize(img,(224,224))
    image = tf.placeholder(tf.uint8,[224,224,3])
    '''
    #keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer()
        input_x = sess.graph.get_tensor_by_name("input_image:0")
        out = sess.graph.get_tensor_by_name("prediction:0")
        image = tf.convert_to_tensor(img)
        image = tf.expand_dims(image,0)
        image = sess.run(image)
        pred = sess.run(out, feed_dict={input_x: image})
        #utils.save_image(pred.astype(np.uint8), "./", name="pred_")
        print(pred.shape)
        pred = np.squeeze(pred, axis=0)
        pred = pred.astype(np.uint8)
        print(pred.shape)
        print(pred)
        #cv2.imwrite("./pred2.jpg",pred)
        gt = cv2.imread('/home/ispr/data/solder_seg/annotations/training/1_cut_13.png')#("gt_8.png")#
        inp = cv2.imread('/home/ispr/data/solder_seg/test/13_cut_162.bmp')#("inp_8.png")#
        inp = cv2.cvtColor(inp,cv2.COLOR_BGR2RGB)
        '''
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i][j]==1:
                    pred[i][j]=255
        '''
        plt.figure("images")
        plt.subplot(2,2,1)
        plt.title("org img")
        plt.axis('off')
        plt.imshow(inp)
        '''
        plt.subplot(2,2,3)
        plt.title("GT img")
        plt.axis('off')
        plt.imshow(gt, cmap=cm.Paired ,vmin=0, vmax=1)
        #plt.colorbar()
        '''
        plt.subplot(2,2,2)
        plt.title("seg img")
        plt.axis('off')
        plt.imshow(pred,cmap=cm.Paired ,vmin=0, vmax=1)
        plt.colorbar()

        plt.show() 
