#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-09-21 10:03
# * Last modified : 2018-09-21 10:03
# * Filename      : augmentations.py
# * Description   : 
# **********************************************************
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
# import random
from random import uniform
class preproc(object):

    def __init__(self,p=0.2):
        self.p = p

    def __call__(self, image, targets,labels):
        boxes = targets.copy()
        height_o, width_o, _ = image.shape
        # image_t, boxes = _mirror(image, boxes)
        boxes[:, 0::2] /= width_o
        boxes[:, 1::2] /= height_o
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        return image,boxes_t,labels_t


def intersect(box_a, box_b):
    #return (num_bboxs,1)
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertToInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.uint8), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

class VerticalFlip(object):

    def __init__(self, prob=0.5):
        # super().__init__()
        self.prob = prob

    def __call__(self, image,boxes = None, labels = None):
        if uniform(0, 1) >= self.prob:
            # print("verticalFlip")
            image = cv2.flip(image, 1)
            width = image.shape[1]
            Boxes = boxes.copy()
            boxes[:,0] = width - 1 - Boxes[:,2]
            boxes[:,1] = boxes[:,1]
            boxes[:,2] = width - 1 - Boxes[:,0]
            boxes[:,3] = boxes[:,3]
        return image, boxes, labels

class HorizonalFlip(object):
    def __init__(self, prob = 0.5):
        # super().__init__()
        self.prob = prob
    def __call__(self, image,boxes = None, labels = None):
        if uniform(0,1) >= self.prob:
            # print("HorizonalFlip")

            image = cv2.flip(image, 0)
            height = image.shape[0]
            # Boxes = [[lb[0], height - 1 -lb[3], lb[2], height - 1 - lb[1],lb[4]]  for lb in boxes]
            Boxes = boxes.copy()
            boxes[:,1] = height - 1 - Boxes[:,3]
            boxes[:,3] = height - 1 - Boxes[:,1]
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """ 
    '''
    To make the model more robust to various input object sizes and
    shapes, each training image is randomly sampled by one of the following options:
    –
    Use the entire original input image.
    –
    Sample a patch so that the
    minimum
    jaccard overlap with the objects is 0.1, 0.3,
    0.5, 0.7, or 0.9.
    –
    Randomly sample a patch.
    '''
    
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
	    #特别注意在这里这两个一个指min iou 另外一个指maxiou
            (0.1,None),
            (0.3,None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
               
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
		#计算所有的iou 和 这张图里头的bbox 的重叠度，如果存在最小值小于对应的值则丢弃
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if not (min_iou <= overlap.min() and overlap.max() <= max_iou):
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                
                # keep overlap with gt box IF center in sampled patch
		# 计算所有的中间值对于所有的bbox
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
		#确保中间值在crop size 左边界 的右边
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
		# 确保中间值在crop size 右边界的左边
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                # 取一个或操作如果存在一个为真
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
		#对于左上角两点我们应当取的为最大的值
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
		#新的位置所以要减去新的坐标系
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
		#对于右下角的坐标系我们需要选取较小的坐标位置
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
		#任然对左上角的参考位置进行调整
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        height, width, depth = image.shape
        for _ in range(50):
            scale = random.uniform(1, 4)

            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = np.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws < 1 or hs < 1:
                continue
            w = int(ws * width)
            h = int(hs * height)

            left = random.randint(0, w - width)
            top = random.randint(0, h - height)

            boxes_t = boxes.copy()
            boxes_t[:, :2] += (left, top)
            boxes_t[:, 2:] += (left, top)

            expand_image = np.empty(
                (h, w, depth),
                dtype=image.dtype)
            expand_image[:, :] = self.mean
            expand_image[top:top + height, left:left + width] = image
            image = expand_image

            return image, boxes_t,labels
        #
        # if random.randint(2):
        #     return image, boxes, labels
        #
        # height, width, depth = image.shape
        # ratio = random.uniform(1, 1.5)
        # left = random.uniform(0, width*ratio - width)
        # top = random.uniform(0, height*ratio - height)
        #
        # expand_image = np.zeros(
        #     (int(height*ratio), int(width*ratio), depth),
        #     dtype=image.dtype)
        # expand_image[:, :, :] = self.mean
        # expand_image[int(top):int(top + height),
        #              int(left):int(left + width)] = image
        # image = expand_image
        #
        # boxes = boxes.copy()
        # boxes[:, :2] += (int(left), int(top))
        # boxes[:, 2:] += (int(left), int(top))
        #
        # return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class Crop(object):
    """Crop
    做这个的目的是对我们的任意输入的图片裁剪到一定范围的值，注意这个操作非常重要因为对于非常大的图片输入往往存在着不可逆问题
    所以这里用于解决不可逆问题的步骤，you must carefully understand
    """
    '''
    To make the model more robust to various input object sizes and
    shapes, each training image is randomly sampled by one of the following options:
    –
    Use the entire original input image.
    –
    Sample a patch so that the
    minimum
    jaccard overlap with the objects is 0.1, 0.3,
    0.5, 0.7, or 0.9.
    –
    Randomly sample a patch.
    '''

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            0.7,0.5,0.7,0.5,0.9
            # randomly sample a patch
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode == None:
                return image,boxes,labels
            h,w = image.shape[0:2]
            if h<1000 and w<1000:
                return image,boxes,labels
            min_iou=mode
            max_iou =None
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.4 * width, 0.6*width)
                h = random.uniform(0.4 * height, 0.6*height)

                # aspect ratio constraint b/t .5 & 2

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                # 计算所有的iou 和 这张图里头的bbox 的重叠度，如果存在最小值小于对应的值则丢弃
                overlap = self.jaccard_numpy(boxes, rect)
                mask = overlap>min_iou
                if (np.sum(mask))==0:
                    continue
                boxes =boxes[mask]
                labels = labels[mask]
                # is min and max overlap constraint satisfied? if not try again

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                # 计算所有的中间值对于所有的bbox
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                # 确保中间值在crop size 左边界 的右边
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                # 确保中间值在crop size 右边界的左边
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                # 取一个或操作如果存在一个为真
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                # 对于左上角两点我们应当取的为最大的值
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # 新的位置所以要减去新的坐标系
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                # 对于右下角的坐标系我们需要选取较小的坐标位置
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # 任然对左上角的参考位置进行调整
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

    def intersect(self,box_a, box_b):
        # return (num_bboxs,1)
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    def jaccard_numpy(self,box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))  # [A,B]
        union = area_a
        return inter / union  # [A,B]
class PhotometricDistort(object):
    def __init__(self):
	#RandomContrast: 隨機處理對比度 對比度值在這裏爲0.5-1.5
	#ConvertColor:將其轉化爲HSV 空間
	#RandomSaturation 隨機的飽和度色調處理
	#其實就是一種隨機的圖像處理增強的機制在這裏特別需要注意的是Hue 在這裏的變化範圍爲18 ---360,9---180
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
	#隨機的明暗處理
        self.rand_brightness = RandomBrightness()
	#通道的改變

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        return distort(im, boxes, labels)

###################################################################################

def isLarge(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    # print('w,h:',width,height)
    flag = False
    if max(width, height) >= 0.05:
        flag = True
    return flag

def filterBox(image,label, width, height):
    width_ratio = 1. / float(width)
    height_ratio = 1. / float(height)
    newlab=[]
    for lb in label:
        lb = lb.astype(np.float32)
        lb[0],lb[2] = lb[0] * width_ratio, lb[2] * width_ratio
        lb[1],lb[3] = lb[1] * height_ratio, lb[3] * height_ratio
        newlab.append(lb)

    newLabel = filter(isLarge, newlab)
    newLabel = list(newLabel)

    for newlab in newLabel:
        newlab[0],newlab[2] = newlab[0]*(width), newlab[2]*(width)
        newlab[1],newlab[3] = newlab[1]*(height), newlab[3]*(height)
    # print("pre contrast after: ",len(label),len(newLabel))
    newLabel = np.asarray(newLabel, np.float32)
    # print("newLabel.shape: ", newLabel.shape)
    return image, newLabel[:,:-1], newLabel[:,-1]

def iou(data):
    xmin, ymin, xmax, ymax = data[0], data[1], data[2], data[3]
    Xmin, Ymin, Xmax, Ymax = data[-4], data[-3], data[-2], data[-1]
    w = min(xmax, Xmax) - max(xmin, Xmin)
    h = min(ymax, Ymax) - max(ymin, Ymin)
    iou = 0
    if w>0 and h>0:
        area = w*h
        boxArea = (data[2] - data[0]) * (data[3] - data[1])
        iou = area / boxArea
    flag = False
    if iou >= 0.95:
        flag = True
    return flag


def patchFilter(label,point0,point1,point2,point3):
    label = np.array(label)
    point = np.array([point0, point1, point2, point3])

    pointArray = np.ones((label.shape[0],4))*point
    allData = list(np.concatenate((label, pointArray),1))
    patchBox = filter(iou,allData)
    boxes = list(patchBox)
    boxes = np.array(boxes)

    # print("boxes.shape: ",boxes.shape)

    boxes[:, 0] = boxes[:, 0] - point0
    boxes[:, 1] = boxes[:, 1] - point1
    boxes[:, 2] = boxes[:, 2] - point0
    boxes[:, 3] = boxes[:, 3] - point1

    boxes = boxes[:,:5]

    return(list(boxes))
def outBoxFilter(data):
    width ,height = data[-2], data[-1]
    flag = False
    if ((data[0]>=0) and (data[1]>=0) and (data[2]>=0) and (data[3]>=0) and (data[0] <width ) and (data[2] < width ) and (data[1] < height) and (data[3] < height )):
        flag = True
    return flag

class abnormalFilter(object):
    def __call__(self,image,boxes,labels):
        labels = labels.reshape((labels.shape[0],1))
        label = np.concatenate((boxes,labels),1)
        height, width = image.shape[:2]
        point = np.array([width, height])
        pointArray = np.ones((labels.shape[0], 2))*point
        # print("pointArray.shape: ",pointArray.shape)
        allData = list(np.concatenate((label, pointArray), 1))
        result = filter(outBoxFilter, allData)
        Result = np.array(result)
        Result = Result.reshape((Result.shape[0],7))
        return image, Result[:,:4], Result[:,4]
class SelectRegion(object):
    def __init__(self, prob = 0.2) :
        # super().__init__()
        self.prob = prob

    def __call__(self, image, boxes, labels):
        labels = labels.reshape((labels.shape[0],1))


        label = np.concatenate((boxes,labels),1)
        height, width = image.shape[:2]
        label = np.asarray(label)
        # label[:,2] = np.clip(label[:,2], 0, width-1)
        # label[:,3] = np.clip(label[:,3], 0, height-1)
        label = list(label)
        index = random.randint(0,len(label))
        selectBox = label[index]
        if max(1.0*(selectBox[2]-selectBox[0])/width, 1.0*(selectBox[3]-selectBox[1])/height) >= 0.08:
            # print("filter all picture")
            return filterBox(image,label, width, height)

        else:
###start
            point0, point1, point2, point3 = 0., 0., 0., 0.
            subsize = 0
            if max(((selectBox[2]-selectBox[0]) ) , (selectBox[3]-selectBox[1])) < 50:
                ratio = uniform(10,20)
                # print('ratio: ',ratio)
                subsize = max((selectBox[2]-selectBox[0]), (selectBox[3]-selectBox[1])) * ratio

            elif max(((selectBox[2]-selectBox[0]) ) , (selectBox[3]-selectBox[1])) < 100:

                ratio = uniform(5,15)
                # print('ratio: ',ratio)
                subsize = max((selectBox[2]-selectBox[0]), (selectBox[3]-selectBox[1])) * ratio


            elif max(((selectBox[2]-selectBox[0]) ) , (selectBox[3]-selectBox[1])) < 300:
                ratio = uniform(3,10)
                # print('ratio: ',ratio)
                subsize = max((selectBox[2]-selectBox[0]), (selectBox[3]-selectBox[1])) * ratio

            else:
                ratio = uniform(3,8)
                # print('ratio: ',ratio)
                subsize = max((selectBox[2]-selectBox[0]), (selectBox[3]-selectBox[1])) * ratio
###start
            # else:
            #     ratio = uniform(10,30)
            #     print('ratio: ',ratio)
            #     subsize = max((selectBox[2]-selectBox[0]), (selectBox[3]-selectBox[1])) * ratio
            # print("subsize",subsize)

            point0 = max(int(selectBox[2] - subsize * uniform(0.4,0.6)), 0)
            point1 = max(int(selectBox[3] - subsize * uniform(0.4,0.6)), 0)
            point2 = min(int(point0 + subsize-1), width-1)
            point3 = min(int(point1 + subsize-1), height-1)

            # print("width,height: ", width, height)
            # print("select box:",selectBox[0],selectBox[1],selectBox[2],selectBox[3])
            # print("patch point: ",point0, point1, point2, point3)
            label = patchFilter(label, point0, point1, point2, point3)
            image = image[point1:point3+1,point0:point2+1,:]
            label = np.array(label)
            # print("image.shape: ", image.shape)
            # print("shape: ",label.shape)
            return image,label[:,:-1],label[:,-1]

###################################################################################


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
	#ConvertFromInts:將其變爲float類型
	#ToAbsoluteCoords:將bbox 變爲原來的非相對值
        #进行HSV空间的图形处理
	#水平翻转
        #在将bbox 变回相对位置
	#减去平均值
        self.augment = Compose([
            ToAbsoluteCoords(),
            abnormalFilter(),
            SelectRegion(prob=0.2),
            HorizonalFlip(prob = 0.5),
            VerticalFlip(prob = 0.5),
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            # PhotometricDistort(),
            # Expand(self.mean),

            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            # ConvertToInts()
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
