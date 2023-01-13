import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
from keras_unet_collection._model_swin_unet_2d import swin_transformer_stack, swin_unet_2d_base
import keras_unet_collection.utils as utils
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d
from keras_unet_collection import models, base
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
import tensorflow as tf 

def get_model(model_name, input_height, input_width, input_channel, backbone, num_classes):
    ######### sm
    if model_name == 'unet':
        model = sm.Unet(backbone, classes=num_classes, activation="softmax")

    ######### tasm  
    elif model_name == 'danet':  
        base_model, layers, layer_names = tasm.create_base_model(name=backbone, weights="imagenet", \
                    height=input_height, width=input_width, include_top=False, pooling=None)

        BACKBONE_TRAINABLE = False
        model = tasm.DANet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
            backbone_trainable=BACKBONE_TRAINABLE)

    # iou_score = tasm.metrics.IOUScore(threshold=0.5)
    # categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss(class_weights=class_weights)
    elif model_name == 'deeplabv3plus':  
        base_model, layers, layer_names = tasm.create_base_model(name=backbone, weights="imagenet", \
                    height=input_height, width=input_width, include_top=False, pooling=None)

        BACKBONE_TRAINABLE = False
        model = tasm.DeepLabV3plus(n_classes=num_classes, base_model=base_model, output_layers=layers, \
            backbone_trainable=BACKBONE_TRAINABLE)


    ########## unet-series
    elif model_name == 'swinunet':
        filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
        depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
        stack_num_down = 2         # number of Swin Transformers per downsampling level
        stack_num_up = 2           # number of Swin Transformers per upsampling level
        patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
        num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
        window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
        num_mlp = 512              # number of MLP nodes within the Transformer
        shift_window=True          # Apply window shifting, i.e., Swin-MSA

        n_labels = num_classes

        # Input section
        input_size = (input_height, input_width, input_channel)
        # IN = Input(input_size)

        # # Base architecture
        # X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
        #                       patch_size, num_heads, window_size, num_mlp, 
        #                       shift_window=shift_window, name='swin_unet')

        # # Output section
        # OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X)

        # # Model configuration
        # model = Model(inputs=[IN,], outputs=[OUT,])

        model = swin_unet_2d(input_size=input_size, filter_num_begin=filter_num_begin, n_labels=n_labels, depth=depth, \
                            stack_num_down=stack_num_down, stack_num_up=stack_num_up, patch_size=patch_size, num_heads=num_heads, \
                            window_size=window_size, num_mlp=num_mlp, shift_window=shift_window, name='swin_unet')


    ########## pidnet


    return model
