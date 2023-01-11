import segmentation_models as sm 
import numpy as np

def get_loss_fn(num_classes):
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
    focal_loss = sm.losses.BinaryFocalLoss() if num_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    return total_loss 