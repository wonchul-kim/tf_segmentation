import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

import segmentation_models as sm
from utils.helpers import visualize
from src.datasets import CamvidDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation

### configurate GPUs settings
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_device, True)
    except:
        pass

output_dir = './results'
if not osp.exists(output_dir):
    os.mkdir(output_dir)
vis_results = osp.join(output_dir, 'figs')
if not osp.exists(vis_results):
    os.mkdir(vis_results)
ckpt_results = osp.join(output_dir, 'checkpoints')
if not osp.exists(ckpt_results):
    os.mkdir(ckpt_results)

### test dataloader
DATA_DIR = '/home/wonchul/HDD/datasets/SegNet-Tutorial-master/CamVid'
CLASSES = ['car', 'pedestrian']

x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_dir = os.path.join(DATA_DIR, 'train/masks')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

tmp_dataset = CamvidDataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])

image, mask = tmp_dataset[5] # get some sample
visualize({"image" :image, "cars_mask": mask[..., 0].squeeze(), "sky_mask": mask[..., 1].squeeze(), "background_mask": mask[..., 2].squeeze()}, 
            fp=osp.join(vis_results, 'raw.png'))

tmp_dataset = CamvidDataset(x_train_dir, y_train_dir, classes=['car', 'sky'], augmentation=get_training_augmentation())

image, mask = tmp_dataset[5] # get some sample
visualize({"image" :image, "cars_mask": mask[..., 0].squeeze(), "sky_mask": mask[..., 1].squeeze(), "background_mask": mask[..., 2].squeeze()}, 
            fp=osp.join(vis_results, 'aug.png'))

### define training parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'


### Define model && strategy for multi-gpu
strategy = tf.distribute.MirroredStrategy()
print('* Number of devices: {}'.format(strategy.num_replicas_in_sync))

BACKBONE = 'efficientnetb1'
BATCH_SIZE = 8
CLASSES = ['car', 'pedestrian']
LR = 0.0001
EPOCHS = 300

preprocess_input = sm.get_preprocessing(BACKBONE)

with strategy.scope():
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    optim = tf.keras.optimizers.Adam(LR)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    metrics = dice_loss + (1 * focal_loss)
    cal_iou = sm.metrics.IOUScore(threshold=0.5)

### define datalader for train images
train_dataset = CamvidDataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

valid_dataset = CamvidDataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

def train_generator():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_dataloader, use_multiprocessing=False)
    multi_enqueuer.start(workers=4, max_queue_size=10)
    for _ in range(len(train_dataloader)):
        batch_xs, batch_ys = next(multi_enqueuer.get())
        yield batch_xs, batch_ys

_train_dataset = tf.data.Dataset.from_generator(train_generator,
                                         output_types=(tf.float64, tf.float64),
                                         output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None, None, None]))
                                        )

### To check the generator is working well
# for idx, batch in enumerate(_train_dataset):
#     print(idx, batch[0].shape, batch[1].shape) 
#     for jdx in range(BATCH_SIZE):
#         fig = plt.figure()
#         plt.subplot(211)
#         plt.imshow(batch[0][jdx])
        
#         plt.subplot(212)
#         plt.imshow(batch[1][jdx])
#         plt.savefig("figs/debug/batch_{}_{}".format(idx, jdx))

#         plt.close()

valid_dataloader = Dataloader(valid_dataset, batch_size=2, shuffle=False)

def valid_generator():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_dataloader, use_multiprocessing=False)
    multi_enqueuer.start(workers=4, max_queue_size=10)
    for _ in range(len(valid_dataloader)):
        batch_xs, batch_ys = next(multi_enqueuer.get()) # I have three outputs
        yield batch_xs, batch_ys

_valid_dataset = tf.data.Dataset.from_generator(valid_generator,
                                         output_types=(tf.float64, tf.float64),
                                         output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        tf.TensorShape([None, None, None, None]))
                                        )

# ## To check the generator is working well
# for idx, batch in enumerate(_valid_dataset):
#     print(idx, batch[0].shape, batch[1].shape, len(batch)) 
#     for jdx in range(1):
#         fig = plt.figure()
#         plt.subplot(211)
#         plt.imshow(batch[0][jdx])
        
#         plt.subplot(212)
#         plt.imshow(batch[1][jdx])
#         plt.savefig("figs/debug/batch_{}_{}".format(idx, jdx))

#         plt.close()

train_dist_dataset = strategy.experimental_distribute_dataset(_train_dataset)
valid_dist_dataset = strategy.experimental_distribute_dataset(_valid_dataset)

val_unscaled_loss = tf.keras.metrics.Sum(name='val_loss')
val_unscaled_iou = tf.keras.metrics.Sum(name='val_iou')

@tf.function
def train_step(batch):
    image, label = batch 
    label = tf.cast(label, tf.float32)
    with tf.GradientTape() as tape:
        preds = model(image)
        loss = metrics(label, preds)
        iou = cal_iou(label, preds)

    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))


    # print("TRAINTRAIN -- unscaled_loss: ", loss, type(loss))
    # print(loss.numpy())
    return loss, iou

@tf.function
def distributed_train_epoch(ds):
    total_loss = 0.
    total_iou = 0.
    num_train_batches = 0.
    for batch in ds:
        per_replica_loss, per_replica_iou = strategy.run(train_step, args=(batch,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        iou = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_iou, axis=None)
        total_loss += loss
        total_iou += iou

        num_train_batches += 1
    
    return total_loss, total_iou, num_train_batches

@tf.function
def val_step(batch):
    image, label = batch 
    label = tf.cast(label, tf.float32)
    preds = model(image)
    unscaled_loss = metrics(label, preds)
    unscaled_iou = cal_iou(label, preds)
    # print("VALVAL -- unscaled_loss: ", unscaled_loss, type(unscaled_loss))
    # print(unscaled_loss.numpy())
    val_unscaled_loss(unscaled_loss)
    val_unscaled_iou(unscaled_iou)

@tf.function
def distributed_val_epoch(ds):
    num_val_batches = 0.
    for batch in ds:
        # print("------------------------------------------------------------idx: ", idx)
        strategy.run(val_step, args=(batch,))
        # print("----------------------------------------------------------------")
        num_val_batches += 1

    return val_unscaled_loss.result(), val_unscaled_iou.result(), num_val_batches
    

TRAIN_LOSSES = []
TRAIN_IOU_SCORES = []
VAL_LOSSES = []
VAL_IOU_SCORES = []
best_iou_score = 0.0
best_val_loss = 999
for epoch in range(300):
    train_total_loss, train_total_iou, num_train_batches = distributed_train_epoch(train_dist_dataset)
    val_total_loss, val_total_iou, num_val_batches = distributed_val_epoch(valid_dist_dataset)
    train_avg_loss = train_total_loss / num_train_batches / strategy.num_replicas_in_sync
    train_avg_iou = train_total_iou / num_train_batches / strategy.num_replicas_in_sync

    val_avg_loss = val_total_loss / num_val_batches / strategy.num_replicas_in_sync
    val_avg_iou = val_total_iou / num_val_batches / strategy.num_replicas_in_sync

    print('>> Epoch: {}, Train Loss: {}, Train IoU: {}, Val Loss: {}, Val IoU: {}'.format(epoch, train_avg_loss, \
                                                                                train_avg_iou, val_avg_loss, val_avg_iou))

    if best_val_loss > val_avg_loss:
        best_val_loss = val_avg_loss
        model.save_weights(osp.join(ckpt_results, 'best_model.h5'))

    val_unscaled_iou.reset_states()
    val_unscaled_loss.reset_states()


    # if sum(val_iou_scores) / len(val_iou_scores) > best_iou_score:
    #     best_iou_score = sum(val_iou_scores) / len(val_iou_scores)
    #     model.save_weights("files/model.h5")
