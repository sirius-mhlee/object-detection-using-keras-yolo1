import sys
import cv2
import random as rand

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataGenerator as dg
import BoxOperator as bo

import YOLONetwork as yn
import YOLOLoss as yl

def generate_image(label_file_path, image, nms_bbox_list):
    label_file = open(label_file_path, 'r')
    label_list = [line.strip() for line in label_file.readlines()]
    label_file.close()

    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(len(label_list))]

    save_image = image.copy()
    height, width, channel = save_image.shape

    for bbox in nms_bbox_list:
        left = int(max(bbox.left, 0))
        top = int(max(bbox.top, 0))
        right = int(min(bbox.right, width))
        bottom = int(min(bbox.bottom, height))

        cv2.rectangle(save_image, (left, top), (right, bottom), color[bbox.get_class()], 2)

        text_size, baseline = cv2.getTextSize(' ' + label_list[bbox.get_class()] + ' ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(save_image, (left - 1, top - text_size[1] - (baseline * 2)), (left + text_size[0], top), color[bbox.get_class()], -1)
        cv2.putText(save_image, ' ' + label_list[bbox.get_class()] + ' ', (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return save_image

def main():
    input_model_path = sys.argv[1]
    label_file_path = sys.argv[2]
    input_image_path = sys.argv[3]
    output_image_path = sys.argv[4]

    yolo_model = yn.create_model(input_model_path)

    image, _, expand_np_image, width, height = dg.load_image(input_image_path)
    pred_output = yolo_model.predict(expand_np_image)[0]

    pred_confidence = np.reshape(pred_output[:, :, :3], [cfg.grid_size, cfg.grid_size, cfg.box_per_grid])
    pred_box = np.reshape(pred_output[:, :, 3:15], [cfg.grid_size, cfg.grid_size, cfg.box_per_grid, 4])
    pred_class = np.reshape(pred_output[:, :, 15:], [cfg.grid_size, cfg.grid_size, cfg.object_class_num])

    offset = np.transpose(np.reshape(np.array([np.arange(cfg.grid_size)] * cfg.grid_size * cfg.box_per_grid), (cfg.box_per_grid, cfg.grid_size, cfg.grid_size)), (1, 2, 0))
    offset = np.reshape(offset, [cfg.grid_size, cfg.grid_size, cfg.box_per_grid])

    pred_box_grid = np.stack([pred_box[:, :, :, 0],
                                pred_box[:, :, :, 1],
                                np.sqrt(pred_box[:, :, :, 2]),
                                np.sqrt(pred_box[:, :, :, 3])])
    pred_box_grid = np.transpose(pred_box_grid, [1, 2, 3, 0])

    pred_box_image = np.stack([(pred_box_grid[:, :, :, 0] + offset) / cfg.grid_size,
                                (pred_box_grid[:, :, :, 1] + np.transpose(offset, (1, 0, 2))) / cfg.grid_size,
                                np.square(pred_box_grid[:, :, :, 2]),
                                np.square(pred_box_grid[:, :, :, 3])])
    pred_box_image = np.transpose(pred_box_image, [1, 2, 3, 0])

    bbox_list = []
    for row in range(cfg.grid_size):
        for col in range(cfg.grid_size):
            for box_idx in range(cfg.box_per_grid):
                prob = pred_confidence[row, col, box_idx] * pred_class[row, col, :]
                if np.sum(prob) > 0:
                    box_center_x = pred_box_image[row, col, box_idx, 0] * width
                    box_center_y = pred_box_image[row, col, box_idx, 1] * height
                    box_width = pred_box_image[row, col, box_idx, 2] * width
                    box_height = pred_box_image[row, col, box_idx, 3] * height
                    
                    left = box_center_x - box_width / 2
                    top = box_center_y - box_height / 2
                    right = box_center_x + box_width / 2
                    bottom = box_center_y + box_height / 2

                    bbox = bo.BBox(left, top, right, bottom, prob)
                    bbox_list.append(bbox)

    nms_bbox_list = bo.nms_bbox(bbox_list, cfg.nms_threshold, cfg.obj_threshold)

    save_image = generate_image(label_file_path, image, nms_bbox_list)
    cv2.imwrite(output_image_path, save_image)

if __name__ == '__main__':
    main()