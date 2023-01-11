import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
from keras_unet_collection._model_swin_unet_2d import swin_transformer_stack, swin_unet_2d_base
import keras_unet_collection.utils as utils
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d
from keras_unet_collection import models, base
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply

def get_model(model_name, backbone, num_classes):
    model = sm.Unet(backbone, classes=num_classes, activation="softmax")

    return model
