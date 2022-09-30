import os 
from src.datasets import CamvidDataset, Dataloader
from utils.augment import  get_preprocessing, get_validation_augmentation
from utils.helpers import visualize, denormalize

import segmentation_models as sm 
import tensorflow as tf 
import keras 
import numpy as np 
import tf2onnx

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

CLASSES = ['car', 'pedestrian']
NUM_CLASSES = len(CLASSES) + 1 # add 1 for background class
BACKBONE = 'efficientnetb1'

activation = 'sigmoid' if NUM_CLASSES == 1 else 'softmax'
preprocess_input = sm.get_preprocessing(BACKBONE)


model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation=activation)
model.load_weights('./results/checkpoints/best_model.h5') 

in_shape = model.inputs[0].shape.as_list()
in_shape[0] = 1
in_shape[1] = 320
in_shape[2] = 320
spec = (tf.TensorSpec(in_shape, tf.float32, name="data"),)        
output_path = './results/best_model.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path=output_path)      
