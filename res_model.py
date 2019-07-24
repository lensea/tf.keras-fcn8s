from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

IMAGE_SIZE = 256
filters = [8, 16, 32, 64, 128]

def batch_norm_relu(x):
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        return x
def bottleneck_residual_v2(x,
                            filters,
                            strides,
                            ps=False):

    shortcut = x
    print("shortcut.shape:\t",shortcut.shape)
    print("ps 1th x.shape:\t",x.shape)
    x = batch_norm_relu(x)
    if ps == True:
        shortcut = layers.Conv2D(filters= filters*4,kernel_size=1,strides=strides,padding='same')(x)
        print("after ps shortcut.shape:\t",shortcut.shape)
        x = layers.Conv2D(filters=filters,kernel_size=1,strides=1,)(x)
        print("ps 2th x.shape:\t",x.shape)
        x = batch_norm_relu(x)
        x = layers.Conv2D(filters=filters,kernel_size=3,strides=strides,padding='same')(x)
        print("ps 3th x.shape:\t",x.shape)
        x = batch_norm_relu(x)
        x = layers.Conv2D(filters=filters*4,kernel_size=1,strides=1)(x)
    else:
        x = layers.Conv2D(filters=filters,kernel_size=1,strides=1,)(x)
        print("ps 2th x.shape:\t",x.shape)
        x = batch_norm_relu(x)
        x = layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')(x)
        print("ps 3th x.shape:\t",x.shape)
        x = batch_norm_relu(x)
        x = layers.Conv2D(filters=filters*4,kernel_size=1,strides=1)(x)
    print("final x.shape:\t",x.shape)
    x = layers.Add()([x,shortcut])
    return x 

def resnet_50(x,NUM_OF_CLASSESS = 2,data_format = 'channels_last'):

    print("org x.shape:\t",x.shape)
    org_size = tf.shape(x)[1:2]
    print("org_size:\t",org_size)
    # 128 128 
    init_conv = layers.Conv2D(8,7,2,input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),padding="same")(x)
    print("init_conv.shape:\t",init_conv.shape)
    pool1 = layers.MaxPool2D(pool_size=2)(init_conv)
    # 64 64
    print("pool1.shape:\t",pool1.shape)
    #stage 2 layer 3 strides 1
    bn1_1 = batch_norm_relu(pool1)
    print("bn1_1.shape:\t",bn1_1.shape)
    #64 64
    ps1_1 = bottleneck_residual_v2(bn1_1,filters=filters[0],strides=1,ps=True)
    ps1_2 = bottleneck_residual_v2(ps1_1,filters=filters[0],strides=1)
    ps1_3 = bottleneck_residual_v2(ps1_2,filters=filters[0],strides=1)
    print("ps1_2.shape:\t",ps1_3.shape)
    #stage 3 layer 4 strides 2
    # 32 32 
    bn2_1 = batch_norm_relu(ps1_3)
    ps2_1  = bottleneck_residual_v2(bn2_1,filters=filters[1],strides=2,ps=True)
    print("stage 3 bottleneck")
    ps2_2  = bottleneck_residual_v2(ps2_1,filters[1],strides=2)
    print("stage 3 bottleneck")
    ps2_3  = bottleneck_residual_v2(ps2_2,filters[1],strides=2)
    print("stage 3 bottleneck")
    ps2_4  = bottleneck_residual_v2(ps2_3,filters[1],strides=2)
    print("ps2_4.shape:\t",ps2_4.shape)
    #stage 4 layer 6 strides 2
    # 16 16 
    bn3_1 = batch_norm_relu(ps2_4)
    ps3_1  = bottleneck_residual_v2(bn3_1,filters=filters[2],strides=2,ps=True)
    print("stage 4 bottleneck")
    ps3_2  = bottleneck_residual_v2(ps3_1,filters=filters[2],strides=2)
    ps3_3  = bottleneck_residual_v2(ps3_2,filters=filters[2],strides=2)
    ps3_4  = bottleneck_residual_v2(ps3_3,filters=filters[2],strides=2)
    ps3_5  = bottleneck_residual_v2(ps3_4,filters=filters[2],strides=2)
    ps3_6  = bottleneck_residual_v2(ps3_5,filters=filters[2],strides=2)
    print("ps3_6.shape:\t",ps3_6.shape)
    #stage 5layer 3 strides 2
    # 8 8 
    bn4_1 = batch_norm_relu(ps3_6)
    ps4_1  = bottleneck_residual_v2(bn4_1,filters=filters[3],strides=2,ps=True)
    ps4_2  = bottleneck_residual_v2(ps4_1,filters=filters[3],strides=2)
    ps4_3  = bottleneck_residual_v2(ps4_2,filters=filters[3],strides=2)
    print("ps4_3.shape:\t",ps4_3.shape)

    conv6 = layers.Conv2D(filters[4],(1,1),(1,1),padding="valid",activation="relu")(ps4_3)
    conv7 = layers.Conv2D(filters[4],(1,1),(1,1),padding="valid",activation="relu")(conv6)
    conv8 = layers.Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(conv7)
    print("conv8.shape：\t",conv8.shape)
    score_pool4 = layers.Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(ps3_6)
    print("score_pool4.shape：\t",score_pool4.shape)
    conv_t1 = layers.Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")(conv8)
    print("conv_t1.shape：\t",conv_t1.shape)
    fuse_1 = layers.Add()([conv_t1,score_pool4])
    print("fuse_1.shape：\t",fuse_1.shape)
    conv_t2 = layers.Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")(fuse_1)
    print("conv_t2.shape：\t",conv_t2.shape)
    score_pool3 = layers.Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")(ps2_3)
    print("score_pool3.shape：\t",score_pool3.shape)
    fuse_2 = layers.Add()([conv_t2,score_pool3])
    print("fuse_2.shape：\t",fuse_2.shape)
    conv_t3 = layers.Conv2DTranspose(NUM_OF_CLASSESS,(16,16),(8,8),padding="same")(fuse_2)
    print("conv_t3.shape：\t",conv_t3.shape)


    annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred,dim=3),conv_t3
