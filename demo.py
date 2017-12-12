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
from func import weight_variable, bias_variable, conv2d, img2feat, gen, RoI, nms

caffe.set_mode_gpu()
model_def = '/home/xiaosucheng/Data/models/VGG-16/deploy.prototxt'
model_weights = '/home/xiaosucheng/Data/models/VGG-16/VGG_ILSVRC_16_layers.caffemodel'
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

wordtoix = cPickle.load(open("word-index/wordtoix.pkl", "rb"))
ixtoword = cPickle.load(open("word-index/ixtoword.pkl", "rb"))


total_train = []
total_regions = []
total_asfmap = []
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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, 'RPN_model/fasterRcnn.module-10')

img_id = 3
img = PIL_Image.open("images/" + str(img_id) + ".jpg")
feature = img2feat(img, net)
feature = np.transpose(feature, [0, 2, 3, 1])

ofs = sess.run(offset, feed_dict={feat_input: feature})
scr = sess.run(score, feed_dict={feat_input: feature})

size = img.size
origin_width = size[0]
origin_height = size[1]
w_scale = width / float(origin_width)
h_scale = height / float(origin_height)

sco = scr.reshape(14, 14, k).transpose(2, 0, 1)
result = ofs.reshape(14, 14, 4 * k).transpose(2, 0, 1)
score_index = np.array((np.where(sco > 0.5)))

infer = []
iscore = []
for i in range(score_index.shape[1]):
    bbx_index = i
    bbx_k = score_index[0, bbx_index]
    bbx_y = score_index[1, bbx_index]
    bbx_x = score_index[2, bbx_index]
    Y = (bbx_y * float(600)) / 13
    X = (bbx_x * float(800)) / 13

    kth = bbx_k
    (h, w) = bbox[kth]
    pos_infer = result[bbx_k * 4:bbx_k * 4 + 4, bbx_y, bbx_x]
    y = Y + pos_infer[0] * h
    x = X + pos_infer[1] * w
    h = h * np.exp(pos_infer[2])
    w = w * np.exp(pos_infer[3])
    y = y - h / 2
    x = x - w / 2
    if x < 0 or y < 0 or h < 5 or w < 5 or x + w > 800 or y + h > 600:
        continue
    infer.append([y, x, h, w])
    iscore.append([sco[bbx_k, bbx_y, bbx_x]])

infer = np.array(infer).reshape(-1, 4)
iscore = np.array(iscore).reshape(1, -1)
num = infer.shape[0]
infer = tf.cast(infer, tf.float32)
iscore = tf.cast(iscore, tf.float32)
nms_infer, nms_score = nms(infer, iscore, num)
nms_infer = sess.run(nms_infer)
nms_score = sess.run(nms_score)

num = nms_infer.shape[0]
nms_infer = tf.cast(nms_infer, tf.float32)

nms_infer = sess.run(nms_infer)

nms_infer = nms_infer.reshape(-1, 4)
train_data = nms_infer
fmap = img2feat(img, net)
fmap = fmap.reshape(512, 14, 14)
fmap = fmap.transpose(1, 2, 0)

asfmap = []
for i in range(train_data.shape[0]):
    Y, X, H, W = train_data[i, :4]
    w_scale = float(14) / width
    h_scale = float(14) / height
    y = int(round(Y * h_scale))
    x = int(round(X * w_scale))
    h = int(round(H * h_scale + (h_scale - 1)))
    w = int(round(W * w_scale + (w_scale - 1)))
    sfmap = fmap[y:y + h + 1, x:x + w + 1, :]
    input_y = sfmap.shape[0]
    input_x = sfmap.shape[1]
    sfmap = sfmap.reshape(1, input_y, input_x, 512)
    sfmap = tf.cast(sfmap, tf.float32)
    sfmap = RoI(sfmap, input_y, input_x)
    sfmap = sess.run(sfmap)
    asfmap.append(sfmap)
asfmap = np.concatenate(asfmap, axis=0)
sess.close()


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.imshow(img)
for i in range(train_data.shape[0]):
    Y, X, H, W = train_data[i, :4]
    if i == 0:
        sc = gen(asfmap[i].reshape(1, 2048), None, ixtoword)
    else:
        sc = gen(asfmap[i].reshape(1, 2048), True, ixtoword)
    ax = plt.gca()
    ax.add_patch(Rectangle((X, Y),
                           W,
                           H,
                           fill=False,
                           edgecolor='red',
                           linewidth=3))
    ax.text(X, Y, sc, style='italic', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
fig = plt.gcf()
plt.tick_params(labelbottom='off', labelleft='off')
plt.show()