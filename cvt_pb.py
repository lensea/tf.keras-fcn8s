# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import sys
import tensorflow as tf
import numpy as np


import BatchDatsetReader as dataset
from six.moves import xrange
import keras_model
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph


os.environ['CUDA_VISIBLE_DEVICES']='-1'  #设置GPU 0,1  CPU -1
'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
'''
MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 256


model_path  = "./log/model.ckpt-3200" #设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前
meta_path  = "./log/model.ckpt-3200.meta" 

def main():

    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    pred_annotation,_ = keras_model.vgg_fcn_8s(image,NUM_OF_CLASSESS = NUM_OF_CLASSESS)
    #'''
    names = [op.name for op in annotation.graph.get_operations()]
    for name in names:
        print(name)
    #'''
    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, model_path)
        tf.train.write_graph(sess.graph_def, './pb/', 'model.pbtxt')
        
        constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,["prediction"])
        with tf.gfile.FastGFile("./pb/model.pb",mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    print("done")

if __name__ == '__main__':
    main()