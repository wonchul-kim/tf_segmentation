from glob import glob
import os.path as osp 
import json 
from tqdm import tqdm 
import cv2 
import numpy as np 

import albumentations as A
# from aivdata.src.slicer.slice import Image2Patches

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(height, width):
    train_transform = [
        A.Resize(height, width),
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # A.RandomCrop(height=320, width=320, always_apply=True),

        # A.GaussNoise(p=0.2),
        # A.Perspective(p=0.5),

        # A.OneOf(
        #     [
        #         A.CLAHE(p=1),
        #         A.RandomBrightness(p=1),
        #         A.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.Sharpen(p=1),
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.RandomBrightnessContrast(p=1),
        #         A.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height, width)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)



# def get_patch_datasets(patch_info, dataset_dir, class2idx, roi=None, img_w=None, img_h=None, input_formats=['bmp', 'png']):
#     engine = Image2Patches()

#     img_files = []
#     for input_format in input_formats:
#         img_files += glob(osp.join(dataset_dir, '*.{}'.format(input_format)))

#     assert len(img_files) != 0, "There are no image in dataset_dir({dataset_dir})"

#     per_patch_info = []

#     for img_file in tqdm(img_files, desc="making patches: "):
#         if img_w == None or img_h == None:
#             img = cv2.imread(img_file)
#             if len(img.shape) == 3:
#                 img_h, img_w, img_c = img.shape
#             elif len(img.shape) == 2:
#                 img_h, img_w = img.shape

#         if roi != None:
#             img_h = roi['br_y'] - roi['tl_y']
#             img_w = roi['br_x'] - roi['tl_x']

#         fn = osp.split(osp.splitext(img_file)[0])[-1]

#         points = []
#         cxs, cys = [], []
#         patch_centric = patch_info['patch_centric']
#         slice_height, slice_width = patch_info['patch_height'], patch_info['patch_width']
#         overlap_ratio = patch_info['patch_overlap_ratio']

#         with open(osp.join(dataset_dir, fn + '.json')) as f:
#             anns = json.load(f)

#         # mask = np.zeros((img_h, img_w))
#         for shape in anns['shapes']:
#             shape_type = str(shape['shape_type'])
#             _label = shape['label']
#             if _label.lower() in class2idx.keys() or _label.upper() in class2idx.keys(): ##################################
#                 if shape_type == 'polygon' or shape_type == 'Watershed':
#                     _points = shape['points']
#                 elif shape_type == 'point': 
#                     _points = shape['points']

#                     if len(_points) == 1:
#                         if len(_points[0]) == 2:
#                             points.append(_points)
#                             continue
                    
#                 try: 
#                     if roi != None:
#                         for _point in _points:
#                             _point[0] -= roi['tl_x']
#                             _point[1] -= roi['tl_y']

#                             if _point[0] < 0 or _point[1] < 0:
#                                 raise ValueError(f"There are defects/objects to detect out of RoI range: {img_file}")

#                     if patch_centric:
#                         for _point in _points:
#                             cxs.append(_point[0])
#                             cys.append(_point[1])
#                         avg_cx = np.mean(cxs)
#                         avg_cy = np.mean(cys)

#                         if avg_cx - slice_width/2 > 0 and avg_cy - slice_height/2 > 0:
#                             per_patch_info.append([img_file, [avg_cx - slice_width/2, avg_cy - slice_height/2, \
#                                                             avg_cx + slice_width/2, avg_cy + slice_height/2]])
#                         elif avg_cx - slice_width/2 < 0 and avg_cy - slice_height/2 > 0:
#                             per_patch_info.append([img_file, [0, avg_cy - slice_height/2, slice_width, avg_cy + slice_height/2]])
#                         elif avg_cx - slice_width/2 > 0 and avg_cy - slice_height/2 < 0:
#                             per_patch_info.append([img_file, [avg_cx - slice_width/2, 0, avg_cx + slice_width/2, slice_height]])
#                         elif avg_cx - slice_width/2 < 0 and avg_cy - slice_height/2 < 0:
#                             per_patch_info.append([img_file, [0, 0, slice_width, slice_height]])

#                     # points_arr = np.array(_points, dtype=np.int32)
#                 except:
#                     # print("Not found points: ", _points)
#                     continue 

#                 assert class2idx[_label] in class2idx.values()
#                 # cv2.fillPoly(mask, [points_arr], color=(class2idx[_label]))
#                 points.append(_points)

#         patch_coords = engine.get_segmentation_pil_patches_info(img_h=img_h, img_w=img_w, slice_height=slice_height, slice_width=slice_width, \
#                                                         points=points, overlap_ratio=overlap_ratio)

#         for patch_coord in patch_coords:
#             per_patch_info.append([img_file, patch_coord])

#     return per_patch_info


# if __name__ == '__main__':
#     TL_X = 220
#     TL_Y = 80
#     BR_X = 1628
#     BR_Y = 1488
#     roi = {"tl_x": TL_X, "tl_y": TL_Y, "br_x": BR_X, "br_y": BR_Y}

#     PATCHES = True 
#     PATCH_CENTRIC = True 
#     PATCH_WIDTH = 320
#     PATCH_HEIGHT = 320
#     PATCH_OVERLAP_RATIO = 0.2
#     patch_info = {"flag": PATCHES, "patch_centric": PATCH_CENTRIC, "patch_width": PATCH_WIDTH, \
#                  "patch_height": PATCH_HEIGHT, "patch_overlap_ratio": PATCH_OVERLAP_RATIO}

#     CLASSES = ["_background_", "FLANGE_S"]
#     class2idx = {}
#     for idx, _class in enumerate(CLASSES):
#             class2idx[_class] = int(idx)

#     img_w = None
#     img_h = None
#     engine = Image2Patches()

#     dataset_dir = '/home/wonchul/mnt/NAS/DeepLearning/_projects/test/ctr/flange/val'

#     get_patch_datasets(patch_info, dataset_dir, class2idx, roi=roi)