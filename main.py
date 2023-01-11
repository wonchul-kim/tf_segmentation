import os
import os.path as osp
import numpy as np

import tensorflow as tf 

import segmentation_models as sm
from utils.helpers import visualize, denormalize
from src.datasets import CamvidDataset, Dataloader
from tensorflow.keras.backend import max

from datasets import get_datasets
from modeling import get_model
from losses import get_loss_fn
from train import train_fit, train_ctl

# train_mode = 'fit'
# train_mode = 'ctl'
train_mode = 'ctl_multigpus'
output_dir = './results'

### test dataloader
DATA_DIR = '/DeepLearning/_uinttest/public/camvid'
input_height, input_width, input_channel = 256, 256, 3
CLASSES = ['car', 'sky']

MODEL_NAME = 'unet'
BACKBONE = 'efficientnetb0'
BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 50

### configurate GPUs settings
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_device, True)
    except:
        pass

if not osp.exists(output_dir):
    output_dir = osp.join(output_dir, train_mode)
    os.makedirs(output_dir, train_mode)
debug_dir = osp.join(output_dir, 'figs')
if not osp.exists(debug_dir):
    os.mkdir(debug_dir)
val_dir = osp.join(output_dir, 'figs')
if not osp.exists(val_dir):
    os.mkdir(val_dir)
weights_dir = osp.join(output_dir, 'checkpoints')
if not osp.exists(weights_dir):
    os.mkdir(weights_dir)


preprocess_input = sm.get_preprocessing(BACKBONE)
train_dataset, val_dataset, train_dataloader, val_dataloader = \
            get_datasets(DATA_DIR, input_height, input_width, input_channel, CLASSES, BATCH_SIZE, debug_dir, preprocess_input)

num_classes = len(CLASSES) + 1

model = get_model(MODEL_NAME, BACKBONE, len(CLASSES) + 1)
optimizer = tf.keras.optimizers.Adam(LR)
loss_fn = get_loss_fn(len(CLASSES) + 1)


if train_mode == 'fit':
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(osp.join(weights_dir, '{}_best_model.h5'.format(train_mode)), save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
    ]


    train_fit(model, EPOCHS, optimizer, loss_fn, train_dataloader, val_dataloader, metrics=metrics, callbacks=callbacks)
elif train_mode == 'ctl':
    train_steps = len(train_dataset)//BATCH_SIZE
    val_steps = len(val_dataset)//BATCH_SIZE

    if len(train_dataset) % BATCH_SIZE != 0:
        train_steps += 1
    if len(val_dataset) % BATCH_SIZE != 0:
        val_steps += 1

    callbacks = None
    metrics = sm.metrics.IOUScore(threshold=0.5)

    train_ctl(model, EPOCHS, optimizer, loss_fn, train_dataloader, train_steps, val_dataloader, val_steps, val_dir, weights_dir, metrics=metrics, callbacks=callbacks)
elif train_mode == 'ctl_multigpus'