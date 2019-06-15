import numpy as np
import tensorflow as tf

import Configuration as cfg

def iou(box1, box2):
    box1 = tf.stack([box1[:, :, :, :, 0] - box1[:, :, :, :, 2] / 2.0,
                    box1[:, :, :, :, 1] - box1[:, :, :, :, 3] / 2.0,
                    box1[:, :, :, :, 0] + box1[:, :, :, :, 2] / 2.0,
                    box1[:, :, :, :, 1] + box1[:, :, :, :, 3] / 2.0])
    box1 = tf.transpose(box1, [1, 2, 3, 4, 0])

    box2 = tf.stack([box2[:, :, :, :, 0] - box2[:, :, :, :, 2] / 2.0,
                    box2[:, :, :, :, 1] - box2[:, :, :, :, 3] / 2.0,
                    box2[:, :, :, :, 0] + box2[:, :, :, :, 2] / 2.0,
                    box2[:, :, :, :, 1] + box2[:, :, :, :, 3] / 2.0])
    box2 = tf.transpose(box2, [1, 2, 3, 4, 0])

    left_up = tf.maximum(box1[:, :, :, :, :2], box2[:, :, :, :, :2])
    right_down = tf.minimum(box1[:, :, :, :, 2:], box2[:, :, :, :, 2:])

    inter_box = tf.maximum(0.0, right_down - left_up)
    inter_area = inter_box[:, :, :, :, 0] * inter_box[:, :, :, :, 1]

    box1_area = (box1[:, :, :, :, 2] - box1[:, :, :, :, 0]) * (box1[:, :, :, :, 3] - box1[:, :, :, :, 1])
    box2_area = (box2[:, :, :, :, 2] - box2[:, :, :, :, 0]) * (box2[:, :, :, :, 3] - box2[:, :, :, :, 1])

    union_area = tf.maximum(box1_area + box2_area - inter_area, 1e-10)

    return tf.clip_by_value(inter_area / union_area, 0.0, 1.0)

def box_loss(true_box, pred_box, object_mask):
    mask = tf.expand_dims(object_mask, 4)
    box_diff = mask * (true_box - pred_box)
    box_loss_mean = cfg.coord_lambda * tf.reduce_mean(tf.reduce_sum(tf.square(box_diff), axis=[1, 2, 3, 4]), name='box_loss')

    return box_loss_mean

def confidence_loss(pred_confidence, object_mask):
    no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

    object_diff = object_mask * (1.0 - pred_confidence)
    object_loss_mean = tf.reduce_mean(tf.reduce_sum(tf.square(object_diff), axis=[1, 2, 3]),  name='object_loss')

    no_object_diff = no_object_mask * pred_confidence
    no_object_loss_mean = cfg.no_object_lambda * tf.reduce_mean(tf.reduce_sum(tf.square(no_object_diff), axis=[1, 2, 3]), name='no_object_loss')

    return object_loss_mean, no_object_loss_mean

def class_loss(true_class, pred_class, true_object):
    class_diff = true_object * (true_class - pred_class)
    class_loss_mean = tf.reduce_mean(tf.reduce_sum(tf.square(class_diff), axis=[1, 2, 3]), name='class_loss')

    return class_loss_mean

def loss(y_true, y_pred):
    index_confidence = cfg.box_per_grid
    index_class = cfg.box_per_grid * (4 + 1)

    pred_confidence = tf.reshape(y_pred[:, :, :, :index_confidence], [-1, cfg.grid_size, cfg.grid_size, cfg.box_per_grid])
    pred_box = tf.reshape(y_pred[:, :, :, index_confidence:index_class], [-1, cfg.grid_size, cfg.grid_size, cfg.box_per_grid, 4])
    pred_class = tf.reshape(y_pred[:, :, :, index_class:], [-1, cfg.grid_size, cfg.grid_size, cfg.object_class_num])

    true_object = tf.reshape(y_true[:, :, :, 0], [-1, cfg.grid_size, cfg.grid_size, 1])
    true_box_image = tf.reshape(y_true[:, :, :, 1:5], [-1, cfg.grid_size, cfg.grid_size, 1, 4])
    true_box_image = tf.tile(true_box_image, [1, 1, 1, cfg.box_per_grid, 1])
    true_class = y_true[:, :, :, 5:]

    offset = np.transpose(np.reshape(np.array([np.arange(cfg.grid_size)] * cfg.grid_size * cfg.box_per_grid), (cfg.box_per_grid, cfg.grid_size, cfg.grid_size)), (1, 2, 0))
    offset = tf.constant(offset, dtype=tf.float32)
    offset = tf.reshape(offset, [1, cfg.grid_size, cfg.grid_size, cfg.box_per_grid])

    pred_box_grid = tf.stack([pred_box[:, :, :, :, 0],
                                pred_box[:, :, :, :, 1],
                                tf.sqrt(pred_box[:, :, :, :, 2]),
                                tf.sqrt(pred_box[:, :, :, :, 3])])
    pred_box_grid = tf.transpose(pred_box_grid, [1, 2, 3, 4, 0])

    pred_box_image = tf.stack([(pred_box_grid[:, :, :, :, 0] + offset) / cfg.grid_size,
                                (pred_box_grid[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cfg.grid_size,
                                tf.square(pred_box_grid[:, :, :, :, 2]),
                                tf.square(pred_box_grid[:, :, :, :, 3])])
    pred_box_image = tf.transpose(pred_box_image, [1, 2, 3, 4, 0])

    true_box_grid = tf.stack([true_box_image[:, :, :, :, 0] * cfg.grid_size - offset,
                                true_box_image[:, :, :, :, 1] * cfg.grid_size - tf.transpose(offset, (0, 2, 1, 3)),
                                tf.sqrt(true_box_image[:, :, :, :, 2]),
                                tf.sqrt(true_box_image[:, :, :, :, 3])])
    true_box_grid = tf.transpose(true_box_grid, [1, 2, 3, 4, 0])

    true_pred_iou = iou(true_box_image, pred_box_image)

    object_mask = tf.reduce_max(true_pred_iou, 3, keepdims=True)
    object_mask = true_object * tf.cast((true_pred_iou >= object_mask), dtype=tf.float32)

    box_loss_mean = box_loss(true_box_grid, pred_box_grid, object_mask)
    object_loss_mean, no_object_loss_mean = confidence_loss(pred_confidence, object_mask)
    class_loss_mean = class_loss(true_class, pred_class, true_object)

    return (box_loss_mean + object_loss_mean + no_object_loss_mean + class_loss_mean)