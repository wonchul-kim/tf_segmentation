import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

import segmentation_models as sm
from utils.helpers import visualize, denormalize
from src.datasets import CamvidDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation

# train_mode = 'fit'
train_mode = 'ctl'

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
DATA_DIR = '/HDD/datasets/public/camvid'
CLASSES = ['car', 'sky']

x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_dir = os.path.join(DATA_DIR, 'train/masks')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

input_height, input_width, input_channel = 128, 128, 3
tmp_dataset = CamvidDataset(x_train_dir, y_train_dir, classes=CLASSES, \
                augmentation=get_training_augmentation(input_height, input_width))

image, mask = tmp_dataset[5] # get some sample
print("Dataset shape: ", image.shape)

visualize({"image" :image, "cars_mask": mask[..., 0].squeeze(), \
            "sky_mask": mask[..., 1].squeeze(), "background_mask": mask[..., 2].squeeze()}, 
            fp=osp.join(vis_results, 'aug.png'))

### define training parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

### Define model
BACKBONE = 'resnet50'
BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 10

preprocess_input = sm.get_preprocessing(BACKBONE)
print(preprocess_input)

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
optim = tf.keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


### define datalader for train images
train_dataset = CamvidDataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
    augmentation=get_training_augmentation(input_height, input_width),
    preprocessing=get_preprocessing(preprocess_input),
)

valid_dataset = CamvidDataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(input_height, input_width),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, input_height, input_width, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, input_height, input_width, n_classes)



if train_mode == 'fit':
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim, total_loss, metrics)
    
    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(osp.join(ckpt_results, 'best_model.h5'), save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
    ]
    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
    )
elif train_mode == 'ctl':
    metrics = sm.metrics.IOUScore(threshold=0.5)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x)
            # preds = tf.cast(preds, tf.float32)
            y = tf.cast(y, tf.float32)
            loss = total_loss(y, preds)
            iou = metrics(y, preds)

        grads = tape.gradient(loss, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))

        return loss, iou

    @tf.function
    def validation_step(x, y):
        preds = model(x)
        y = tf.cast(y, tf.float32)
        test_loss = total_loss(y, preds)
        test_iou = metrics(y, preds)

        return test_loss, test_iou


    train_steps = len(train_dataset)//BATCH_SIZE
    valid_steps = len(valid_dataset)//BATCH_SIZE

    if len(train_dataset) % BATCH_SIZE != 0:
        train_steps += 1
    if len(valid_dataset) % BATCH_SIZE != 0:
        valid_steps += 1


    TRAIN_LOSSES = []
    TRAIN_IOU_SCORES = []
    VAL_LOSSES = []
    VAL_IOU_SCORES = []
    best_iou_score = 0.0
    for epoch in range(300):
        # initializes training losses and iou_scores lists for the epoch
        losses = []
        iou_scores = []

        for step, (batch) in enumerate(train_dataloader):

            x, y = batch[0], batch[1]
            # run one training step for the current batch
            loss, iou = train_step(x, y)

            # Save current batch loss and iou-score
            losses.append(float(loss))
            iou_scores.append(float(iou))

            print("\r Epoch: {} >> step: {}/{} >> train-loss: {} >> IOU: {}".format(epoch, step, train_steps, \
                                np.round(sum(losses) / len(losses), 4), np.round(sum(iou_scores) / len(iou_scores), 4)), end="")

        # Save the train and validation losses and iou scores for each epoch.
        TRAIN_LOSSES.append(sum(losses) / len(losses))
        TRAIN_IOU_SCORES.append(sum(iou_scores) / len(iou_scores))
            
        if epoch % 1 == 0 and epoch != 0:
            val_losses = []
            val_iou_scores = []
            for val_step, val_batch in enumerate(valid_dataloader):
                x_val, y_val = val_batch[0], val_batch[1]

                # val_loss, val_iou_score = distributed_val_step(x_val, y_val)
                val_loss, val_iou_score = validation_step(x_val, y_val)
                
                val_losses.append(val_loss)
                val_iou_scores.append(val_iou_score)

                print("** \rEpoch: {} >> Val_Loss: {} >> Val_IOU-Score: {} ".format(epoch, np.round(sum(val_losses) / len(val_losses), 4), \
                                    np.round(sum(val_iou_scores) / len(val_iou_scores), 4)), end="")
                
                    
            if sum(val_iou_scores) / len(val_iou_scores) > best_iou_score:
                best_iou_score = sum(val_iou_scores) / len(val_iou_scores)
                model.save_weights(osp.join(ckpt_results, 'best_model_{}.h5'.format(epoch)))
                
                preds = model(x_val)
                visualize({"image" : denormalize(image.squeeze()), "gt_mask": y_val.squeeze(), \
                    "pr_mask": preds.numpy().squeeze()}, fp=osp.join(vis_results, 'val_{}.png'.format(epoch)))

                print("------------------------------------------")
            VAL_LOSSES.append(sum(val_losses) / len(val_losses))
            VAL_IOU_SCORES.append(sum(val_iou_scores) / len(val_iou_scores))
            
        print()

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("./results/figs/res_{}.png".format(EPOCHS))

