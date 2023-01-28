import matplotlib.pyplot as plt 
import numpy as np 
import os.path as osp 

def vis_history(history, epoch, train_mode, output_dir):
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
    plt.savefig(osp.join(output_dir, "{}_res_{}.png".format(train_mode, epoch)))
    plt.close()

def vis_res(TRAIN_IOU_SCORES, VAL_IOU_SCORES, TRAIN_LOSSES, VAL_LOSSES, output_dir, train_mode, epoch):
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
    plt.savefig(osp.join(output_dir, "{}_res_{}.png".format(train_mode, epoch)))
    plt.close()

# helper function for data visualization
def visualize(images, fp=None):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(fp)
    plt.close()

def normalize_255(image, **Kwargs):
    transformed_image = image/255.

    return transformed_image

def denormalize_255(image, dtype='uint8'):
    denormed_image = image*255

    if dtype == 'unint8':
        return denormed_image.astype(np.uint8)
    else:
        return denormed_image


# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
