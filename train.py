import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

from pidnet.pidnet import PIDNet

import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
from keras_unet_collection._model_swin_unet_2d import swin_transformer_stack, swin_unet_2d_base
import keras_unet_collection.utils as utils
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d
from keras_unet_collection import models, base
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply

from utils.helpers import visualize, denormalize
from src.datasets import CamvidDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation
from tensorflow.keras.backend import max

# train_mode = 'fit'
# train_mode = 'ctl'
train_mode = 'ctl_multigpus'

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
DATA_DIR = '/HDD/datasets/public/SegNet-Tutorial-master/camvid/'
CLASSES = ['car', 'sky']

x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_dir = os.path.join(DATA_DIR, 'train/masks')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

input_height, input_width, input_channel = 256, 256, 3

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
BACKBONE = 'efficientnetb0'
BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 50

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
        tf.keras.callbacks.ModelCheckpoint(osp.join(ckpt_results, '{}_best_model.h5'.format(train_mode)), save_weights_only=True, save_best_only=True, mode='min'),
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
    plt.savefig("./results/figs/{}_res_{}.png".format(train_mode, EPOCHS))

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
    for epoch in range(EPOCHS):
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
                model.save_weights(osp.join(ckpt_results, '{}_best_model.h5'.format(train_mode)))
                
                
                # preds = model(x_val)
                # visualize({"image" : denormalize(image.squeeze()), "gt_mask": y_val.squeeze(), \
                #     "pr_mask": preds.numpy().squeeze()}, fp=osp.join(vis_results, 'val_{}.png'.format(epoch)))

                for _val_step, _val_batch in enumerate(valid_dataloader):
                    _x_val, _y_val = _val_batch[0], _val_batch[1]
                    _preds = model(_x_val)
                    visualize({"image" : denormalize(_x_val.squeeze()), "gt_mask": _y_val.squeeze(), \
                        "pr_mask": _preds.numpy().squeeze()}, fp=osp.join(vis_results, 'val_{}_{}.png'.format(epoch, _val_step)))
                print("------------------------------------------")
            VAL_LOSSES.append(sum(val_losses) / len(val_losses))
            VAL_IOU_SCORES.append(sum(val_iou_scores) / len(val_iou_scores))
            
        print()
        
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(TRAIN_IOU_SCORES)
    plt.plot(VAL_IOU_SCORES)
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(TRAIN_LOSSES)
    plt.plot(VAL_LOSSES)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./results/figs/{}_res_{}.png".format(train_mode, EPOCHS))

elif train_mode == 'ctl_multigpus':
    strategy = tf.distribute.MirroredStrategy()
    print('* Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        ######### sm
        # model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)   
        # optim = tf.keras.optimizers.Adam(LR)

        # ######### tasm    
        # base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE, weights="imagenet", \
        #             height=input_height, width=input_width, include_top=False, pooling=None)

        # BACKBONE_TRAINABLE = False
        # MODEL_PATH = "danet"
        # model = tasm.DANet(n_classes=n_classes, base_model=base_model, output_layers=layers, \
        #     backbone_trainable=BACKBONE_TRAINABLE)

        # optim = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)
        # # iou_score = tasm.metrics.IOUScore(threshold=0.5)
        # # categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss(class_weights=class_weights)
    
        # ########## unet-series
        # filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
        # depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
        # stack_num_down = 2         # number of Swin Transformers per downsampling level
        # stack_num_up = 2           # number of Swin Transformers per upsampling level
        # patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
        # num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
        # window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
        # num_mlp = 512              # number of MLP nodes within the Transformer
        # shift_window=True          # Apply window shifting, i.e., Swin-MSA

        # n_labels = n_classes

        # # Input section
        # input_size = (input_height, input_width, input_channel)
        # # IN = Input(input_size)

        # # # Base architecture
        # # X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
        # #                       patch_size, num_heads, window_size, num_mlp, 
        # #                       shift_window=shift_window, name='swin_unet')

        # # # Output section
        # # OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X)

        # # # Model configuration
        # # model = Model(inputs=[IN,], outputs=[OUT,])

        # model = swin_unet_2d(input_size=input_size, filter_num_begin=filter_num_begin, n_labels=n_labels, depth=depth, \
        #                     stack_num_down=stack_num_down, stack_num_up=stack_num_up, patch_size=patch_size, num_heads=num_heads, \
        #                     window_size=window_size, num_mlp=num_mlp, shift_window=shift_window, name='swin_unet')

        # optim = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)

        # ########## pidnet


        ### -------------------------------------------------------------------------------------------------
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        metrics = dice_loss + (1 * focal_loss)
        cal_iou = sm.metrics.IOUScore(threshold=0.5)


        
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
    for epoch in range(EPOCHS):
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
            model.save_weights(osp.join(ckpt_results, '{}_best_model.h5'.format(train_mode)))

        val_unscaled_iou.reset_states()
        val_unscaled_loss.reset_states()


        # if sum(val_iou_scores) / len(val_iou_scores) > best_iou_score:
        #     best_iou_score = sum(val_iou_scores) / len(val_iou_scores)
        #     model.save_weights("files/model.h5")
