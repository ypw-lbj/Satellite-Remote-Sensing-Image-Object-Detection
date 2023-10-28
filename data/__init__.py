from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        _,h,w= sample[0].shape
        boxes = sample[1]
        targets.append(torch.FloatTensor(boxes))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    # print (size)
    # print (image.shape)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x
class preproc(object):

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), p=0.2):
        self.p = p

    def __call__(self, image, targets):
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o))
        # image_t, boxes = _mirror(image, boxes)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width_o
        boxes[:, 1::2] /= height_o
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return targets_t
# 4 corner chips
class Test_Crop(object):
    def __call__(self,img,thresh = 0.6):
        h,w = img.shape[0:2]
        crop_h = int(thresh*h)
        crop_w = int(thresh*w)
        img_0 = img[:crop_h,:crop_w,:]
        img_1 = img[:crop_h,w-crop_w:,:]
        img_2 = img[h-crop_h:,:crop_w,:]
        img_3 = img[h-crop_h:,w-crop_w:,:]
        return img_0,img_1,img_2,img_3

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
