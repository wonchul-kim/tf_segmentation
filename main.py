import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

import segmentation_models as sm

from datasets import get_datasets
from dist_datasets import get_dist_dataset
from modeling import get_model
from losses import get_loss_fn
from train_single_gpu import train_fit, train_ctl
from train_multiple_gpus import train_ctl_multigpus

# train_mode = 'fit'
train_mode = 'ctl'
# train_mode = 'ctl_multigpus'
output_dir = './results'

### test dataloader
# DATA_DIR = '/DeepLearning/_uinttest/public/camvid'
DATA_DIR = "/HDD/datasets/public/SegNet-Tutorial-master/camvid"
input_height, input_width, input_channel = 256, 256, 3
CLASSES = ['car', 'sky']

MODEL_NAME = 'unet'
BACKBONE = 'efficientnetb0'
BATCH_SIZE = 2
EPOCHS = 10
OPT = 'adam' # SGD
LR = 0.0001

# OPT = 'sgd' # danet
# LR = 0.2


# OPT = 'adam' # swinunet
# LR = 0.0001

### configurate GPUs settings
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_device, True)
    except:
        pass

output_dir = osp.join(output_dir, train_mode)
if not osp.exists(output_dir):
    os.makedirs(output_dir)
debug_dir = osp.join(output_dir, 'debug')
if not osp.exists(debug_dir):
    os.mkdir(debug_dir)
val_dir = osp.join(output_dir, 'val')
if not osp.exists(val_dir):
    os.mkdir(val_dir)
weights_dir = osp.join(output_dir, 'weights')
if not osp.exists(weights_dir):
    os.mkdir(weights_dir)


preprocess_input = sm.get_preprocessing(BACKBONE)
num_classes = len(CLASSES) + 1

print(f">>> Start {train_mode} training: ")
if train_mode == 'fit' or train_mode == 'ctl':
    train_dataset, val_dataset, train_dataloader, val_dataloader = \
            get_datasets(DATA_DIR, input_height, input_width, input_channel, CLASSES, BATCH_SIZE, 1, debug_dir, preprocess_input)

    model = get_model(MODEL_NAME, input_height, input_width, input_channel, BACKBONE, len(CLASSES) + 1)
    if OPT == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipvalue=0.5)
    elif OPT == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9) # LR = 0.2
    loss_fn = get_loss_fn(len(CLASSES) + 1)

    if train_mode == 'fit':
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(osp.join(weights_dir, '{}_best_model.h5'.format(train_mode)), save_weights_only=True, save_best_only=True, mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau(),
        ]

        history = train_fit(model, EPOCHS, optimizer, loss_fn, train_dataloader, val_dataloader, metrics=metrics, callbacks=callbacks)
        
        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(osp.join(output_dir, "fit_res_{}.png".format(EPOCHS)))

    elif train_mode == 'ctl':
        callbacks = None
        metrics = sm.metrics.IOUScore(threshold=0.5)

        TRAIN_IOU_SCORES, VAL_IOU_SCORES, TRAIN_LOSSES, VAL_LOSSES = train_ctl(model, EPOCHS, optimizer, loss_fn, train_dataloader, val_dataloader, val_dir, weights_dir, metrics=metrics, callbacks=callbacks)

        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(TRAIN_IOU_SCORES)
        plt.plot(VAL_IOU_SCORES)
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(TRAIN_LOSSES)
        plt.plot(VAL_LOSSES)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(osp.join(output_dir, "fit_res_{}.png".format(EPOCHS)))

elif train_mode == 'ctl_multigpus':
    callbacks = None
    strategy = tf.distribute.MirroredStrategy()
    print('* Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    train_dataset, val_dataset, train_dataloader, val_dataloader = \
            get_datasets(DATA_DIR, input_height, input_width, input_channel, CLASSES, BATCH_SIZE, strategy.num_replicas_in_sync, debug_dir, preprocess_input)
    
    train_dist_dataset = get_dist_dataset(strategy, train_dataloader)
    val_dist_dataset = get_dist_dataset(strategy, val_dataloader)

    with strategy.scope():
        model = get_model(MODEL_NAME, input_height, input_width, input_channel, BACKBONE, len(CLASSES) + 1)
        if OPT == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipvalue=0.5)
        elif OPT == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9) # LR = 0.2

        loss_fn = get_loss_fn(len(CLASSES) + 1)
        metrics = sm.metrics.IOUScore(threshold=0.5)
        # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        def compute_loss(labels, preds):
            per_example_loss = loss_fn(labels, preds)
            loss = tf.nn.compute_average_loss(per_example_loss,
                                            global_batch_size=BATCH_SIZE*strategy.num_replicas_in_sync)
            return loss


    TRAIN_IOU_SCORES, VAL_IOU_SCORES, TRAIN_LOSSES, VAL_LOSSES = train_ctl_multigpus(strategy, model, EPOCHS, BATCH_SIZE, optimizer, loss_fn, train_dist_dataset, val_dist_dataset, val_dir, weights_dir, metrics=metrics, callbacks=callbacks)

    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(TRAIN_IOU_SCORES)
    plt.plot(VAL_IOU_SCORES)
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(TRAIN_LOSSES)
    plt.plot(VAL_LOSSES)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(osp.join(output_dir, "fit_res_{}.png".format(EPOCHS)))