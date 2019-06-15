import cv2

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def load_image(image_path):
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    reshape_image = cv2.resize(image, dsize=(cfg.image_size_width, cfg.image_size_height), interpolation=cv2.INTER_CUBIC)
    np_image = np.array(reshape_image, dtype=float)
    expand_np_image = np.expand_dims(np_image, axis=0)
    return image, np_image, expand_np_image, width, height

class DataGenerator(kr.utils.Sequence):
    def __init__(self, data_path, batch_size, transform_param=None):
        self.data_image_info = []
        self.data_label_info = {}

        data_file = open(data_path, 'r')
        while True:
            line = data_file.readline()
            if not line:
                break

            image_name_box_cnt = line.split(' ')
            self.data_image_info.append(image_name_box_cnt[0])

            class_box_info_list = []
            for idx in range(int(image_name_box_cnt[1])):
                line = data_file.readline()

                class_box_info = line.split(' ')
                class_box_info_list.append((int(class_box_info[0]), int(class_box_info[1]), int(class_box_info[2]), int(class_box_info[3]), int(class_box_info[4])))

            self.data_label_info[image_name_box_cnt[0]] = class_box_info_list
        data_file.close()

        self.batch_size = batch_size
        self.image_data_generator = kr.preprocessing.image.ImageDataGenerator()
        self.transform_param = transform_param
        self.shuffle = True

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_image_info) / self.batch_size))

    def __getitem__(self, index):
        batch_data_image_info = self.data_image_info[index * self.batch_size:(index + 1) * self.batch_size]
        image, label = self.data_generation(batch_data_image_info)

        return image, label

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.data_image_info)

    def data_generation(self, batch_data_image_info):
        batch_image = np.empty((self.batch_size, cfg.image_size_width, cfg.image_size_height, 3))
        batch_label = np.empty((self.batch_size, cfg.grid_size, cfg.grid_size, ((4 + 1) + cfg.object_class_num)))

        for batch_idx, data_image_info in enumerate(batch_data_image_info):
            _, np_image, _, width, height = load_image(data_image_info)

            np_image -= cfg.image_mean_value
            np_image /= 255.0

            if self.transform_param is not None:
                batch_image[batch_idx] = self.image_data_generator.apply_transform(np_image, self.transform_param)
            else:
                batch_image[batch_idx] = np_image

            data_label_info = np.array(self.data_label_info[data_image_info], dtype=float)

            class_info = np.array(data_label_info[:, 0], dtype=int)

            data_label_info[:, 1:5:2] /= width
            data_label_info[:, 2:5:2] /= height

            box_info = np.stack([
                (data_label_info[:, 1] + data_label_info[:, 3]) / 2,
                (data_label_info[:, 2] + data_label_info[:, 4]) / 2,
                (data_label_info[:, 3] - data_label_info[:, 1]),
                (data_label_info[:, 4] - data_label_info[:, 2])
            ], axis=1)

            x_grid_idx = np.array(box_info[:, 0] * cfg.grid_size, dtype=int)
            y_grid_idx = np.array(box_info[:, 1] * cfg.grid_size, dtype=int)

            label = np.zeros((cfg.grid_size, cfg.grid_size, ((4 + 1) + cfg.object_class_num)))
            label[y_grid_idx[:], x_grid_idx[:], 0] = 1
            label[y_grid_idx[:], x_grid_idx[:], 1:5] = box_info[:, ...]
            label[y_grid_idx[:], x_grid_idx[:], 5 + class_info[:]] = 1

            batch_label[batch_idx] = label

        return batch_image, batch_label