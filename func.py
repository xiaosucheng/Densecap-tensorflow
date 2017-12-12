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

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def sen2ix(sentence, wordtoix):
    sentence = str(sentence)
    sentence = str.lower(sentence)
    if sentence[-2] == '.':
        sentence = sentence[:-2] + ' .'
    elif sentence[-1] != '.':
        sentence = sentence + ' .'
    else:
        sentence = sentence[:-1] + ' .'
    sentence = sentence.split()
    for i in range(len(sentence)):
        if sentence[i][-1] == ',':
            sentence[i] = sentence[i][:-1]
    test = map(lambda x: wordtoix[x], sentence)
    test = map(lambda x: x + 1, test)
    train = test[:-1]
    test = np.array((test)).reshape(1, -1)
    train = np.array((train)).reshape(1, -1)
    return train, test


def img2feat(img, net):
    img = np.array(img.resize([224, 224]))
    net.blobs['data'].data[...] = img.transpose([2, 0, 1])
    net.forward()
    feat = net.blobs['conv5_3'].data
    return feat


def neural_network(input_image_feature, input_data, keep_prob, model='lstm', rnn_size=256, num_layers=1, reuse=None):
    len_words = 3000
    image_feat_size = 2048

    if model == 'rnn':
        cell_fun = rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(1, tf.float32)

    with tf.variable_scope('rnnlm', reuse=reuse):
        with tf.device("/gpu:0"):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, len_words])
            softmax_b = tf.get_variable("softmax_b", [len_words])
            Wi = tf.get_variable("Wi", [image_feat_size, rnn_size])
            bi = tf.get_variable("bi", [rnn_size])

            embedding = tf.get_variable("embedding", [len_words, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

            inputs = tf.nn.dropout(inputs, keep_prob)

            image_inp = tf.matmul(input_image_feature, Wi) + bi

            image_inp = tf.nn.dropout(image_inp, keep_prob)

            inputs = tf.reshape(inputs, [-1, rnn_size])
            input_stack = tf.concat([image_inp, inputs], 0)
            input_stack = tf.reshape(input_stack, [1, -1, rnn_size])
            image_inp = tf.reshape(image_inp, [1, -1, rnn_size])
            inputs = tf.reshape(inputs, [1, -1, rnn_size])

            outputs, last_state = tf.nn.dynamic_rnn(cell, input_stack, initial_state=initial_state, scope='rnnlm')
            output = tf.reshape(outputs, [-1, rnn_size])
            img_outs, img_last_state = tf.nn.dynamic_rnn(cell, image_inp, initial_state=initial_state, scope='rnnlm')
            img_out = tf.reshape(img_outs, [-1, rnn_size])
            word_outs, word_last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
            word_out = tf.reshape(word_outs, [-1, rnn_size])

            output = tf.nn.dropout(output, keep_prob)
            img_out = tf.nn.dropout(img_out, keep_prob)
            word_out = tf.nn.dropout(word_out, keep_prob)

            logits = tf.matmul(output, softmax_w) + softmax_b
            probs = tf.nn.softmax(logits)
            img_logits = tf.matmul(img_out, softmax_w) + softmax_b
            img_probs = tf.nn.softmax(img_logits)
            word_logits = tf.matmul(word_out, softmax_w) + softmax_b
            word_probs = tf.nn.softmax(word_logits)
    return logits, last_state, probs, cell, initial_state, img_probs, img_last_state, word_probs, word_last_state


def gen(feature, reuse, ixtoword):
    keep = 0.5
    image_feat_size = 2048
    input_image_feature = tf.placeholder(tf.float32, [1, image_feat_size])
    input_data = tf.placeholder(tf.int64, [1, None])
    keep_prob = tf.placeholder(tf.float32)

    _, last_state, probs, cell, initial_state, img_probs, img_last_state, word_probs, word_last_state = neural_network(
        input_image_feature, input_data, keep_prob, reuse=reuse)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, 'RNN_model/test.module')
        state_ = sess.run(cell.zero_state(1, tf.float32))
        x = feature
        [probs_, state_] = sess.run([img_probs, img_last_state],
                                    feed_dict={input_image_feature: x, initial_state: state_, keep_prob: keep})
        num = np.argmax(probs_)
        word = ixtoword[num - 1]
        sentence = []
        count = 0
        while num != 1:
            sentence.append(word)
            x = num.reshape(1, 1)
            [probs_, state_] = sess.run([word_probs, word_last_state],
                                        feed_dict={input_data: x, initial_state: state_, keep_prob: keep})
            num = np.argmax(probs_)
            word = ixtoword[num - 1]
            count += 1
            if count > 20:
                break
    sentence = map(lambda x: str(x), sentence)
    sentence = ' '.join(sentence)
    return sentence


def nms(infer, score, num):
    '''nms

    infer: N x 4 tensor
    score: N x 1 tensor
    num: N

    return:
    nms_infer: n x 4 tensor
    nms_score: n x 1 tensor
    '''
    inferY = tf.expand_dims(infer, axis=1)
    inferY = tf.tile(inferY, [1, num, 1])

    inferX = tf.expand_dims(infer, axis=0)
    inferX = tf.tile(inferX, [num, 1, 1])

    yc11, xc11, height1, width1 = tf.unstack(inferY, axis=2)
    yc21, xc21, height2, width2 = tf.unstack(inferX, axis=2)

    x11, y11 = xc11, yc11
    x21, y21 = xc21, yc21
    x12, y12 = x11 + width1, y11 + height1
    x22, y22 = x21 + width2, y21 + height2

    intersection = (
        tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21)) *
        tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))
    )
    # N x N tensor
    iou_metric = intersection / (
        width1 * height1 + width2 * height2 - intersection
    )

    mask = tf.greater(iou_metric, 0.5)
    mask = tf.cast(mask, tf.float32)
    score1 = tf.tile(score, [num, 1])
    score_mask = score1 * mask
    max_pos = tf.reshape(tf.argmax(score_mask, axis=1), [-1, 1])
    true_labels = tf.cast(np.array([i for i in range(num)]).reshape(-1, 1), tf.int64)
    infer_mask = tf.equal(max_pos, true_labels)
    infer_mask = tf.reshape(infer_mask, [-1])

    nms_infer = tf.boolean_mask(infer, infer_mask)
    score = tf.reshape(score, [-1, 1])
    nms_score = tf.boolean_mask(score, infer_mask)

    return nms_infer, nms_score


def RoI(feature, input_y, input_x, output_y=2, output_x=2):
    if input_y >= 2 and input_x >= 2:
        feature = tf.nn.max_pool(feature, ksize=[1, int(round(float(input_y) / 2)), int(round(float(input_x) / 2)), 1],
                                 strides=[1, input_y / 2, input_x / 2, 1], padding='VALID')
    elif input_y >= 2:
        feature = tf.nn.max_pool(feature, ksize=[1, int(round(float(input_y) / 2)), 1, 1],
                                 strides=[1, input_y / 2, 1, 1], padding='VALID')
        feature = tf.tile(feature, [1, 1, 2, 1])
    elif input_x >= 2:
        feature = tf.nn.max_pool(feature, ksize=[1, 1, int(round(float(input_x) / 2)), 1],
                                 strides=[1, 1, input_x / 2, 1], padding='VALID')
        feature = tf.tile(feature, [1, 2, 1, 1])
    else:
        feature = tf.tile(feature, [1, 2, 2, 1])
    return tf.reshape(feature, [1, 2048])


def train_neural_network(total_train, total_regions, total_asfmap, wordtoix):
    keep = 0.5
    image_feat_size = 2048
    len_words = 3000
    input_image_feature = tf.placeholder(tf.float32, [1, image_feat_size])
    input_data = tf.placeholder(tf.int64, [1, None])
    keep_prob = tf.placeholder(tf.float32)
    output_targets = tf.placeholder(tf.int64, [1, None])
    # feat = tf.placeholder(tf.float32, [])
    logits, last_state, _, _, _, _, _, _, _ = neural_network(input_image_feature, input_data, keep_prob)
    targets = tf.reshape(output_targets, [-1])
    loss = seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len_words)
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(500):
            sess.run(tf.assign(learning_rate, 0.001 * (0.9 ** (epoch / 100))))
            for k in range(len(total_train)):
                train_data = total_train[k]
                regions = total_regions[k]
                asfmap = total_asfmap[k]
                for i in range(train_data.shape[0]):
                    train, test = sen2ix(regions[train_data[i, 4].astype('int32')]['phrase'], wordtoix)
                    train_loss, _, _ = sess.run([cost, last_state, train_op],
                                                feed_dict={input_data: train,
                                                           input_image_feature: asfmap[i].reshape(1, 2048),
                                                           output_targets: test, keep_prob: keep})

                if (epoch + 1) % 50 == 0:
                    print(epoch, train_loss)
        saver.save(sess, 'RNN_model/test.module')
    print "train end!"



def compute_iou(ground_truth, ground_truth_count, proposals, proposals_count):
    '''Caclulate IoU for given ground truth and proposal boxes

    ground_truth: M x 4 ground truth boxes tensor
    proposals: N x 4 ground truth boxes tensor

    returns:
    N x M IoU tensor
    '''
    proposals = tf.expand_dims(proposals, axis=1)
    proposals = tf.tile(proposals, [1, ground_truth_count, 1])

    ground_truth = tf.expand_dims(ground_truth, axis=0)
    ground_truth = tf.tile(ground_truth, [proposals_count, 1, 1])

    yc11, xc11, height1, width1 = tf.unstack(proposals, axis=2)
    yc21, xc21, height2, width2 = tf.unstack(ground_truth, axis=2)

    x11, y11 = xc11 - width1 // 2, yc11 - height1 // 2
    x21, y21 = xc21 - width2 // 2, yc21 - height2 // 2
    x12, y12 = x11 + width1, y11 + height1
    x22, y22 = x21 + width2, y21 + height2

    intersection = (
        tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21)) *
        tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))
    )

    iou_metric = intersection / (
        width1 * height1 + width2 * height2 - intersection
    )
    return iou_metric


def generate_anchors(boxes, height, width, conv_height, conv_width):
    '''Generate anchors for given geometry

    boxes: K x 2 tensor for anchor geometries, K different sizes
    height: source image height
    width: source image width
    conv_height: convolution layer height
    conv_width: convolution layer width

    returns:
    conv_height x conv_width x K x 4 tensor with boxes for all
    positions. Last dimension 4 numbers are (y, x, h, w)
    '''
    k, _ = boxes.get_shape().as_list()

    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    grid = tf.transpose(tf.stack(tf.meshgrid(
        tf.linspace(0.0, height - 0.0, conv_height),
        tf.linspace(0.0, width - 0.0, conv_width)), axis=2), [1, 0, 2])

    # convert boxes from K x 2 to 1 x 1 x K x 2
    boxes = tf.expand_dims(tf.expand_dims(boxes, 0), 0)
    # convert grid from H' x W' x 2 to H' x W' x 1 x 2
    grid = tf.expand_dims(grid, 2)

    # combine them into single H' x W' x K x 4 tensor
    return tf.concat([tf.tile(grid, [1, 1, k, 1]),
                      tf.tile(boxes, [conv_height, conv_width, 1, 1])], 3)


def generate_proposals(coefficients, anchors):
    '''Generate proposals from static anchors and normalizing coefficients

    coefficients: N x 4 tensor: N x (ty, tx, th, tw)
    anchors: N x 4 tensor with boxes N x (y, x, h, w)

    anchors contains x,y of box _center_ while returned tensor x,y coordinates
    are top-left corner.

    returns:
    N x 4 tensor with bounding box proposals
    '''

    y_coef, x_coef, h_coef, w_coef = tf.unstack(coefficients, axis=1)
    y_anchor, x_anchor, h_anchor, w_anchor = tf.unstack(anchors, axis=1)

    w = w_anchor * tf.exp(w_coef)
    h = h_anchor * tf.exp(h_coef)
    x = x_anchor + x_coef * w_anchor
    y = y_anchor + y_coef * h_anchor

    proposals = tf.stack([y, x, h, w], axis=1)
    return proposals


def split_proposals(proposals, iou, scores):
    '''Generate batches from proposals and ground truth boxes

    Idea is to drastically reduce number of proposals to evaluate. So, we find those
    proposals that have IoU > 0.5 with _any_ ground truth and mark them as positive samples.
    Proposals with IoU < 0.3 with _all_ ground truth boxes are considered negative. All
    other proposals are discarded.

    We generate batch with at most half of examples being positive. We also pad them with negative
    have we not enough positive proposals.

    proposals: N x 4 tensor
    iou: N x M tensor of IoU between every proposal and ground truth
    scores: N-dimension vector with scores
    '''
    # now let's get rid of non-positive and non-negative samples
    # Sample is considered positive if it has IoU > 0.7 with _any_ ground truth box
    # XXX: maximal IoU ground truth proposal should be treated as positive
    positive_mask = tf.reduce_any(tf.greater_equal(iou, 0.5), axis=1)

    # Sample would be considered negative if _all_ ground truch box
    # have iou less than 0.3
    negative_mask = tf.reduce_all(tf.less(iou, 0.3), axis=1)

    # Select only positive boxes and their corresponding predicted scores
    positive_boxes = tf.boolean_mask(proposals, positive_mask)
    positive_scores = tf.boolean_mask(scores, positive_mask)
    positive_labels = tf.ones_like(positive_scores)

    # Same for negative
    negative_boxes = tf.boolean_mask(proposals, negative_mask)
    negative_scores = tf.boolean_mask(scores, negative_mask)
    negative_labels = tf.zeros_like(negative_scores)


    return (
        (positive_boxes, positive_scores, positive_labels),
        (negative_boxes, negative_scores, negative_labels)
    )


def centerize_ground_truth(ground_truth_pre):
    y, x, height, width = tf.unstack(ground_truth_pre, axis=1)
    yc, xc = y + height // 2, x + width // 2
    return tf.stack([yc, xc, height, width], axis=1)


def get_gt_param(ground_truth, ground_truth_num, anchor_centers, proposals_num):
    # ground_truth shape is M x 4, where M is count and 4 are y,x,h,w
    ground_truth = tf.expand_dims(ground_truth, axis=0)
    ground_truth = tf.tile(ground_truth, [proposals_num, 1, 1])
    # anchor_centers shape is N x 4 where N is count and 4 are ya,xa,ha,wa
    anchor_centers = tf.expand_dims(anchor_centers, axis=1)
    anchor_centers = tf.tile(anchor_centers, [1, ground_truth_num, 1])

    y_anchor, x_anchor, height_anchor, width_anchor = tf.unstack(anchor_centers, axis=2)
    y_ground_truth, x_ground_truth, height_ground_truth, width_ground_truth = tf.unstack(
        ground_truth, axis=2)

    tx_ground_truth = (x_ground_truth - x_anchor) / width_anchor
    ty_ground_truth = (y_ground_truth - y_anchor) / height_anchor
    tw_ground_truth = tf.log(width_ground_truth / width_anchor)
    th_ground_truth = tf.log(height_ground_truth / height_anchor)

    gt_params = tf.stack(
        [ty_ground_truth, tx_ground_truth, th_ground_truth, tw_ground_truth], axis=2)
    return gt_params


def get_offset_labels(gt_params, gt_num, offset, iou):
    '''get the positive anchors' offset and labels

    gt_params: 4 x N x M tensor
    offset: N x 4 tensor
    iou: N x M tensor

    returns:
    pos_offset: pos_num x 4 tensor
    pos_offset_labels: pos_num x 4 tensor
    '''
    pos_mask = tf.reduce_any(tf.greater_equal(iou, 0.5), axis=1)
    pos_gt_params = tf.boolean_mask(gt_params, pos_mask)
    pos_offset = tf.boolean_mask(offset, pos_mask)
    pos_iou = tf.boolean_mask(iou, pos_mask)
    max_mask = tf.equal(tf.one_hot(tf.argmax(pos_iou, axis=1), gt_num, 1, 0, -1), 1)
    pos_offset_labels = tf.reshape(tf.boolean_mask(pos_gt_params, max_mask), [-1, 4])
    return pos_offset, pos_offset_labels