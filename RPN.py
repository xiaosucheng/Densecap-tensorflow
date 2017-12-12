import caffe
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import time
import io
import cPickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
from StringIO import StringIO
import json
from func import weight_variable, bias_variable, conv2d, img2feat, compute_iou, generate_anchors,\
    generate_proposals, split_proposals, centerize_ground_truth, get_gt_param, get_offset_labels

caffe.set_mode_gpu()
model_def = '/home/xiaosucheng/Data/models/VGG-16/deploy.prototxt'
model_weights = '/home/xiaosucheng/Data/models/VGG-16/VGG_ILSVRC_16_layers.caffemodel'
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

boxes = tf.Variable([
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
], dtype=tf.float32)
bbox = [
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
]

conv_height = 14
conv_width = 14
height = 600
width = 800
k = 12
gt_num = 10
anchors_num = k * conv_height * conv_width
img_num = 10
img_Epoch = 200000
train_img = np.random.permutation(img_num) + 1
description = json.load(open("/home/xiaosucheng/Data/VG/region_descriptions.json", "rb"))

feat_input = tf.placeholder(tf.float32, [None, conv_height, conv_width, 512])

with tf.variable_scope('rcnn', reuse=None):
    W_conv6 = weight_variable([3, 3, 512, 256], name="W_conv6")
    b_conv6 = bias_variable([256], name="b_conv6")
    feat = conv2d(feat_input, W_conv6) + b_conv6

    W_offset = weight_variable([1, 1, 256, k * 4], name="W_offset")
    b_offset = bias_variable([k * 4], name="b_offset")
    offset = conv2d(feat, W_offset) + b_offset
    offset = tf.reshape(offset, [k * conv_height * conv_width, 4])

    W_score = weight_variable([1, 1, 256, k], name="W_score")
    b_score = bias_variable([k], name="b_score")
    score = conv2d(feat, W_score) + b_score
    score = tf.reshape(score, [k * conv_height * conv_width])

anchors = generate_anchors(boxes, height, width, conv_height, conv_width)
anchors = tf.reshape(anchors, [-1, 4])
ground_truth_pre = tf.placeholder(tf.float32, [None, 4])
ground_truth = centerize_ground_truth(ground_truth_pre)
iou = compute_iou(ground_truth, gt_num, anchors, anchors_num)
positive, negative = split_proposals(anchors, iou, score)
positive_bbox, positive_scores, positive_labels = positive
negative_bbox, negative_scores, negative_labels = negative

predicted_scores = tf.concat([positive_scores, negative_scores], 0)
true_labels = tf.concat([positive_labels, negative_labels], 0)
score_loss = tf.reduce_sum(tf.square(predicted_scores - true_labels))

gt_param = get_gt_param(ground_truth, gt_num, anchors, anchors_num)
pos_offset, pos_offset_labels = get_offset_labels(gt_param, gt_num, offset, iou)
offset_loss = tf.reduce_sum(tf.square(pos_offset - pos_offset_labels))

total_loss = score_loss + offset_loss
learning_rate = tf.Variable(0.0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
print "start training ..."
for img_epoch in range(img_Epoch):
    sess.run(tf.assign(learning_rate, 0.0001 * (0.98 ** (img_epoch / 10000))))
    for img_id in train_img:
        img = PIL_Image.open("/home/xiaosucheng/Data/VG/VG_100K/" + str(img_id) + ".jpg")
        regions = description[img_id - 1]["regions"]
        size = img.size
        origin_width = size[0]
        origin_height = size[1]
        w_scale = width / float(origin_width)
        h_scale = height / float(origin_height)
        ground_truth_ = []
        for idx in range(gt_num):
            rgt = [int(round(regions[idx]["y"] * h_scale)),
                   int(round(regions[idx]["x"] * w_scale)),
                   int(round(regions[idx]["height"] * h_scale + (h_scale - 1))),
                   int(round(regions[idx]["width"] * w_scale + (w_scale - 1)))]
            ground_truth_.append(rgt)
        ground_truth_ = np.array((ground_truth_))
        feature = img2feat(img, net)
        feature = np.transpose(feature, [0, 2, 3, 1])
        Epoch = 7
        with tf.device("/gpu:0"):
            for epoch in range(Epoch):
                sess.run([train_step], feed_dict={feat_input: feature, ground_truth_pre: ground_truth_})
                #                 if epoch%(Epoch/10) == 0:
    if img_epoch % (img_Epoch / 100) == 0:
        print "epoch:", img_epoch, "img:", img_id, sess.run([total_loss],
                                                            feed_dict={feat_input: feature,
                                                                       ground_truth_pre: ground_truth_})
    if (img_epoch + 1) % (img_Epoch / 10) == 0:
        saver.save(sess, 'RPN_model/fasterRcnn.module', global_step=img_epoch + 1)
sess.close()
print "train end"