from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Add,Conv2DTranspose
from sklearn.preprocessing import LabelEncoder

IMAGE_SIZE = 256
img_w = 256
img_h = 256
#有一个为背景
n_label = 1+1


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5




def vgg_fcn_8s(inputs,NUM_OF_CLASSESS = 2,data_format = 'channels_last'):
    # 1 3 224 224 
    '''
    print("inputs.shape:\t",inputs.shape)
    if data_format != data_format:
        print("inputs.shape:\t",inputs.shape)
        inputs = tf.transpose(inputs,[0,3,1,2])
        print("after transpose inputs.shape:\t",inputs.shape)

    '''
    print("after transpose inputs.shape:\t",inputs.shape)
    conv1_1 = Conv2D(4,(3,3),strides=(1,1),input_shape=(img_w,img_h,3),padding='same',activation='relu')(inputs)
    batch_size = conv1_1.shape[0]
    conv1_1_imgs=conv1_1[0:batch_size,:,:,0:conv1_1.shape[3]]
    tf.summary.image("conv1_1_imgs",conv1_1_imgs,max_outputs=conv1_1.shape[3])
    bn1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(4,(3,3),strides=(1,1),padding='same',activation='relu')(bn1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2,2))(bn1_2)
    #(128,128)
    conv2_1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    bn2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2_2)
    #(64,64)
    conv3_1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    bn3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn3_2)
    bn3_3 = BatchNormalization()(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3_3)
    #(32,32)
    conv4_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    bn4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn4_1)
    bn4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn4_2)
    bn4_3 = BatchNormalization()(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4_3)
    #(16,16)
    conv5_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(pool4)
    bn5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn5_1)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(bn5_2)
    bn5_3 = BatchNormalization()(conv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5_3)
    print("pool5.shape：\t",pool5.shape)
    conv6 = Conv2D(128,(1,1),(1,1),padding="valid",activation="relu")(pool5)
    conv7 = Conv2D(128,(1,1),(1,1),padding="valid",activation="relu")(conv6)
    conv8 = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(conv7)
    print("conv8.shape：\t",conv8.shape)
    score_pool4 = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(pool4)
    print("score_pool4.shape：\t",score_pool4.shape)
    conv_t1 = Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")(conv8)
    print("conv_t1.shape：\t",conv_t1.shape)
    fuse_1 = Add()([conv_t1,score_pool4])
    print("fuse_1.shape：\t",fuse_1.shape)
    conv_t2 = Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")(fuse_1)
    print("conv_t2.shape：\t",conv_t2.shape)
    score_pool3 = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(pool3)
    print("score_pool3.shape：\t",score_pool3.shape)
    fuse_2 = Add()([conv_t2,score_pool3])
    print("fuse_2.shape：\t",fuse_2.shape)
    conv_t3 = Conv2DTranspose(NUM_OF_CLASSESS,(16,16),(8,8),padding="same")(fuse_2)
    print("conv_t3.shape：\t",conv_t3.shape)


    annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred,dim=3),conv_t3