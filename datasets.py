import os.path as osp 
from src.datasets import CamvidDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.helpers import visualize
import numpy as np 

def get_datasets(input_dir, input_height, input_width, input_channel, classes, train_batch_size, val_batch_size, debug_dir, preprocess_input=None):

    x_train_dir = osp.join(input_dir, 'train/images')
    y_train_dir = osp.join(input_dir, 'train/masks')

    x_valid_dir = osp.join(input_dir, 'val/images')
    y_valid_dir = osp.join(input_dir, 'val/masks')

    tmp_dataset = CamvidDataset(x_train_dir, y_train_dir, classes=classes, \
                    augmentation=get_training_augmentation(input_height, input_width))

    ### define datalader for train images
    train_dataset = CamvidDataset(
        x_train_dir, 
        y_train_dir, 
        classes=classes, 
        augmentation=get_training_augmentation(input_height, input_width),
        preprocessing=get_preprocessing(preprocess_input),
    )

    val_dataset = CamvidDataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=classes, 
        augmentation=get_validation_augmentation(input_height, input_width),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=val_batch_size, shuffle=False)

    idxes = [1, 10, 30, 40]
    for idx in idxes:
        __val_batch = val_dataloader[idx] # get some sample
        for _val_step, (_val_image, _val_mask) in enumerate(zip(__val_batch[0], __val_batch[1])):

            print("* Dataset shape: ", _val_image.shape, _val_mask.shape)
            print("* Image Max: ", np.max(_val_image))
            print("* Image Min: ", np.min(_val_image))

            visualize({"image" :_val_image, "cars_mask": _val_mask[..., 0].squeeze(), \
                        "sky_mask": _val_mask[..., 1].squeeze(), \
                        "pedestrian_mask": _val_mask[..., 2].squeeze(), \
                        "background_mask": _val_mask[..., 3].squeeze()}, 
                        fp=osp.join(debug_dir, 'aug_{}_{}.png'.format(idx, _val_step)))
    # for __val_step, __val_batch in enumerate(val_dataloader):
    #     for _val_step, (_val_image, _val_mask) in enumerate(zip(__val_batch[0], __val_batch[1])):
    #         _preds = model(tf.expand_dims(_val_image, 0))
    #         visualize({"image" : denormalize(_val_image), "gt_mask": _val_mask, \
    #             "pr_mask": _preds.numpy().squeeze()}, fp=osp.join(val_dir, 'val_{}_{}_{}.png'.format(epoch, __val_step, _val_step)))


    # check shapes for errors
    assert train_dataloader[0][0].shape == (train_batch_size, input_height, input_width, 3)
    assert train_dataloader[0][1].shape == (train_batch_size, input_height, input_width, len(classes) + 1)

    return train_dataset, val_dataset, train_dataloader, val_dataloader