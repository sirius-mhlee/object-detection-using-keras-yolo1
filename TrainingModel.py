import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataGenerator as dg

import YOLONetwork as yn
import YOLOLoss as yl

def main():
    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    output_model_path = sys.argv[3]

    yolo_model = yn.create_model()
    adam_optimizer = kr.optimizers.Adam(lr=cfg.learning_rate, clipnorm=0.001)

    yolo_model.compile(loss=yl.loss, optimizer=adam_optimizer)
    yolo_model.summary()

    train_transform_param = {
        'tx':0.1,
        'ty':0.1,
        'zx':0.1,
        'zy':0.1
    }

    train_generator = dg.DataGenerator(train_data_path, cfg.batch_size, train_transform_param)
    validation_generator = dg.DataGenerator(validation_data_path, cfg.batch_size)

    earlystop = kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min')
    reducelr = kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    tensorboard = kr.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, batch_size=cfg.batch_size, write_graph=True, write_images=False)

    yolo_model.fit_generator(
        epochs=cfg.epochs,
        generator=train_generator,
        validation_data=validation_generator,
        callbacks=[earlystop, reducelr, tensorboard]
    )

    yolo_model.save_weights(output_model_path)

if __name__ == '__main__':
    main()