import os 
from src.datasets import CamvidDataset, Dataloader
from utils.augment import  get_preprocessing, get_validation_augmentation
from utils.helpers import visualize, denormalize

import segmentation_models as sm 
import tensorflow as tf 
import keras 
import numpy as np 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

DATA_DIR = '/home/wonchul/HDD/datasets/SegNet-Tutorial-master/CamVid'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

BACKBONE = 'efficientnetb1'
CLASSES = ['car', 'pedestrian']
LR = 0.0001

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
preprocess_input = sm.get_preprocessing(BACKBONE)

test_dataset = CamvidDataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)
test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)

# load best weights
model.load_weights('./results/checkpoints/best_model.h5') 

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
    
    visualize({"image": denormalize(image.squeeze()), "gt_mask": gt_mask.squeeze(), "pr_mask": pr_mask.squeeze()}, fp='./results/figs/res_{}.png'.format(i))
