import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def create_model(pre_train_file=None):
    model = kr.models.Sequential()

    model.add(kr.layers.InputLayer(input_shape=(cfg.image_size_width, cfg.image_size_height, 3), name='input'))

    model.add(kr.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1'))
    model.add(kr.layers.BatchNormalization(name='norm1'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu1'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1'))

    model.add(kr.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2'))
    model.add(kr.layers.BatchNormalization(name='norm2'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu2'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2'))

    model.add(kr.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv3'))
    model.add(kr.layers.BatchNormalization(name='norm3'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu3'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv4'))
    model.add(kr.layers.BatchNormalization(name='norm4'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu4'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv5'))
    model.add(kr.layers.BatchNormalization(name='norm5'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu5'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv6'))
    model.add(kr.layers.BatchNormalization(name='norm6'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu6'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool6'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv7'))
    model.add(kr.layers.BatchNormalization(name='norm7'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu7'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv8'))
    model.add(kr.layers.BatchNormalization(name='norm8'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu8'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv9'))
    model.add(kr.layers.BatchNormalization(name='norm9'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu9'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv10'))
    model.add(kr.layers.BatchNormalization(name='norm10'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu10'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv11'))
    model.add(kr.layers.BatchNormalization(name='norm11'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu11'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv12'))
    model.add(kr.layers.BatchNormalization(name='norm12'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu12'))

    model.add(kr.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv13'))
    model.add(kr.layers.BatchNormalization(name='norm13'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu13'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv14'))
    model.add(kr.layers.BatchNormalization(name='norm14'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu14'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv15'))
    model.add(kr.layers.BatchNormalization(name='norm15'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu15'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv16'))
    model.add(kr.layers.BatchNormalization(name='norm16'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu16'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool16'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv17'))
    model.add(kr.layers.BatchNormalization(name='norm17'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu17'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv18'))
    model.add(kr.layers.BatchNormalization(name='norm18'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu18'))

    model.add(kr.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv19'))
    model.add(kr.layers.BatchNormalization(name='norm19'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu19'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv20'))
    model.add(kr.layers.BatchNormalization(name='norm20'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu20'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv21'))
    model.add(kr.layers.BatchNormalization(name='norm21'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu21'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False, name='conv22'))
    model.add(kr.layers.BatchNormalization(name='norm22'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu22'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv23'))
    model.add(kr.layers.BatchNormalization(name='norm23'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu23'))

    model.add(kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv24'))
    model.add(kr.layers.BatchNormalization(name='norm24'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu24'))

    #model.add(kr.layers.LocallyConnected2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False, name='local25'))
    #model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu25'))

    model.add(kr.layers.Flatten())

    model.add(kr.layers.Dense(units=4096, name='fc26'))
    model.add(kr.layers.LeakyReLU(alpha=0.1, name='relu26'))
    model.add(kr.layers.Dropout(rate=0.5, name='drop26'))
    
    model.add(kr.layers.Dense(units=(cfg.grid_size * cfg.grid_size * (cfg.box_per_grid * (4 + 1) + cfg.object_class_num)), activation='sigmoid', name='fc27'))
    model.add(kr.layers.Reshape((cfg.grid_size, cfg.grid_size, (cfg.box_per_grid * (4 + 1) + cfg.object_class_num)), name='reshape27'))

    if pre_train_file is not None:
        model.load_weights(pre_train_file, by_name=True)

    return model