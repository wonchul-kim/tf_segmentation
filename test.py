import os 
from src.datasets import CamvidDataset, Dataloader
from utils.augment import  get_preprocessing, get_validation_augmentation
from utils.helpers import visualize, denormalize

import segmentation_models as sm 
import tensorflow_advanced_segmentation_models as tasm
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d
from keras_unet_collection import models, base
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
import tensorflow as tf 
from tensorflow.keras.backend import max

import keras 
import numpy as np 

# train_mode = 'fit'
# train_mode = 'ctl'
train_mode = 'ctl_multigpus'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

DATA_DIR = '/HDD/datasets/public/camvid'
x_test_dir = os.path.join(DATA_DIR, 'val/images')
y_test_dir = os.path.join(DATA_DIR, 'val/masks')

BACKBONE = 'efficientnetb0'
CLASSES = ['car', 'sky']
LR = 0.0001

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
preprocess_input = sm.get_preprocessing(BACKBONE)

input_height, input_width, input_channel = 256, 256, 3
test_dataset = CamvidDataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(input_height, input_width),
    preprocessing=get_preprocessing(preprocess_input),
)
test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE, weights="imagenet", \
#             height=input_height, width=input_width, include_top=False, pooling=None)

# BACKBONE_TRAINABLE = False
# MODEL_PATH = "danet"
# model = tasm.DANet(n_classes=n_classes, base_model=base_model, output_layers=layers, \
#     backbone_trainable=BACKBONE_TRAINABLE)


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


# ########## unet-series
name = 'unet3plus'
activation = 'ReLU'
filter_num_down = [32, 64, 128, 256, 512]
filter_num_skip = [32, 32, 32, 32]
filter_num_aggregate = 160

stack_num_down = 2
stack_num_up = 1
n_labels = n_classes

# `unet_3plus_2d_base` accepts an input tensor 
# and produces output tensors from different upsampling levels
# ---------------------------------------- #
input_tensor = tf.keras.layers.Input((input_height, input_width, input_channel))
# base architecture
X_decoder = base.unet_3plus_2d_base(
    input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
    stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation, 
    batch_norm=True, pool=True, unpool=True, backbone=None, name=name)

OUT_stack = []
# reverse indexing `X_decoder`, so smaller tensors have larger list indices 
X_decoder = X_decoder[::-1]

# deep supervision outputs
for i in range(1, len(X_decoder)):
    # 3-by-3 conv2d --> upsampling --> sigmoid output activation
    pool_size = 2**(i)
    X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv1_{}'.format(name, i-1))(X_decoder[i])
    
    X = UpSampling2D((pool_size, pool_size), interpolation='bilinear', 
                    name='{}_output_sup{}'.format(name, i-1))(X)
    
    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
    # collecting deep supervision tensors
    OUT_stack.append(X)

# the final output (without extra upsampling)
# 3-by-3 conv2d --> sigmoid output activation
X = Conv2D(n_labels, 3, padding='same', name='{}_output_final'.format(name))(X_decoder[0])
X = Activation('sigmoid', name='{}_output_final_activation'.format(name))(X)
# collecting final output tensors
OUT_stack.append(X)

# Classification-guided Module (CGM)
# ---------------------------------------- #
# dropout --> 1-by-1 conv2d --> global-maxpooling --> sigmoid
X_CGM = X_decoder[-1]
X_CGM = Dropout(rate=0.1)(X_CGM)
X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
X_CGM = GlobalMaxPooling2D()(X_CGM)
X_CGM = Activation('sigmoid')(X_CGM)

CGM_mask = max(X_CGM, axis=-1) # <----- This value could be trained with "none-organ image"

for i in range(len(OUT_stack)):
    if i < len(OUT_stack)-1:
        # deep-supervision
        OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_sup{}_CGM'.format(name, i))
    else:
        # final output
        OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_final_CGM'.format(name))


model = tf.keras.models.Model([input_tensor,], OUT_stack)
optim = tf.keras.optimizers.Adam(learning_rate=1e-4)


optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)
model.build(input_shape=(None, input_height, input_width, input_channel))

# load best weights
model.load_weights('./results/checkpoints/{}_best_model.h5'.format(train_mode)) 

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)
for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    
    visualize({"image": denormalize(image.squeeze()), \
            "gt_mask": gt_mask, "pr_mask": pr_mask.squeeze()}, \
            fp='./results/figs/{}_res_{}.png'.format(train_mode, i))
