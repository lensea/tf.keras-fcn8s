#-*- coding:utf-8 -*-
import os
import sys
import argparse 
import cv2
import numpy as np 
import math
import time
import datetime
import tensorflow as tf
import  keras_model
import ready_data
import BatchDatsetReader as dataset

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'
#tf.enable_eager_execution()

data_dir='/home/ispr/data/solder_seg/'
log_dir='./log/'
train_epochs=100
batch_size=2
learning_rate = 1e-6 
_MOMENTUM = 0.9
_WEIGHT_DECAY = 0.0005
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 256
_NUM_IMAGES = {
    'training': 2250,
    'validation': 1542,
}


def main():
    
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)

    anno_pred,conv_t = keras_model.vgg_fcn_8s(image,NUM_OF_CLASSESS=NUM_OF_CLASSESS)
    tf.summary.image("pred_annotation", tf.cast(anno_pred, tf.uint8), max_outputs=2)
    
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=conv_t,
        labels=tf.squeeze(annotation, axis=[3]),
        name="entropy"
    )
    loss = tf.reduce_mean(entropy+1e-10)
    tf.summary.scalar("loss",loss)

    
    
    global_step = tf.train.get_or_create_global_step()
    # Create a tensor named learning_rate for logging purposes.

    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate,momentum=_MOMENTUM)
    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    


    train_steps = math.ceil(_NUM_IMAGES["training"]/batch_size)
    eval_steps = math.ceil(_NUM_IMAGES['validation']/ batch_size)

    #sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    train_records, valid_records = ready_data.get_imgs_path(data_dir)
    train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        train_writer = tf.summary.FileWriter(log_dir,graph=sess.graph)
        validation_writer = tf.summary.FileWriter(log_dir + 'val/',graph=sess.graph)
        train_writer.add_graph(sess.graph)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\n******Model restored from %s******\n"%(ckpt.model_checkpoint_path))
        for iter in range(train_epochs):
            print('\n\nStarting a training cycle.\n\n')
            for step in range(train_steps):
                train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
                #print(train_images.shape)
                #print(train_annotations.shape)
                feed_dict = {image: train_images, annotation: train_annotations}
                sess.run(train_op, feed_dict=feed_dict)
                if step % batch_size == 0:
                    train_loss,summary_str = sess.run([loss,summary_op],feed_dict=feed_dict) 
                    print("Epoch:[%d/%d] Step:[%d/%d/%d] ---> Train_loss:%g"%(iter,train_epochs,step,train_steps,batch_size,train_loss))
                    train_writer.add_summary(summary_str,step)
                    
            print('\n\nStarting to evaluate after one epoch .\n\n')
            avg_eval_loss = 0.0
            for step in range(eval_steps):
                
                valid_images, valid_annotations = validation_dataset_reader.next_batch(batch_size)
                valid_loss, summary_sva = sess.run([loss, summary_op], 
                                                    feed_dict={image: valid_images,annotation: valid_annotations})
                if step % batch_size == 0:
                    print("Epoch:[%d/%d] Step:[%d/%d/%d] ---> Validation_loss: %g" % (iter,train_epochs,step,eval_steps,batch_size, valid_loss))
                avg_eval_loss += valid_loss
            # add validation loss to TensorBoard
            avg_eval_loss /=eval_steps
            print("After %d Epoch train ---> Validation avg loss :\t%f"%(iter,avg_eval_loss))
            validation_writer.add_summary(summary_sva, iter*train_epochs*batch_size)
            saver.save(sess, log_dir + "model.ckpt", iter*train_epochs*batch_size)

       
if __name__ == '__main__':
    
    main()