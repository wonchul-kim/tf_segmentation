import os.path as osp 
from src.datasets import CamvidDataset, Dataloader
from utils.augment import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.helpers import visualize


def get_datasets(input_dir, input_height, input_width, input_channel, classes, batch_size, debug_dir, preprocess_input=None):

    x_train_dir = osp.join(input_dir, 'train/images')
    y_train_dir = osp.join(input_dir, 'train/masks')

    x_valid_dir = osp.join(input_dir, 'val/images')
    y_valid_dir = osp.join(input_dir, 'val/masks')

    tmp_dataset = CamvidDataset(x_train_dir, y_train_dir, classes=classes, \
                    augmentation=get_training_augmentation(input_height, input_width))

    image, mask = tmp_dataset[5] # get some sample
    print("Dataset shape: ", image.shape)

    visualize({"image" :image, "cars_mask": mask[..., 0].squeeze(), \
                "sky_mask": mask[..., 1].squeeze(), "background_mask": mask[..., 2].squeeze()}, 
                fp=osp.join(debug_dir, 'aug.png'))

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

    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (batch_size, input_height, input_width, 3)
    assert train_dataloader[0][1].shape == (batch_size, input_height, input_width, len(classes) + 1)

    return train_dataset, val_dataset, train_dataloader, val_dataloader