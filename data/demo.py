#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-09-26 10:56
# * Last modified : 2018-09-26 10:56
# * Filename      : demo.py
# * Description   : 
# **********************************************************
"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
'''*******************************************类名填写×××××××××××××××××××××××××××××××××××××××××××××××××××××'''
#特别注意这一步是非常美丽的一步切结这里是填写我们要分类的类名字需要特别注意的是类的名字和分类要按特定类别分好，其中背景在这里不添加使用
TRAIN_CLASSES=['defect']
'''
VOC_CLASSES = (  # always index 0
    'aeroplane')
'''
'''*******************************************类名填写×××××××××××××××××××××××××××××××××××××××××××××××××××××'''
# note: if you used our download scripts, this should be right
abs_path = osp.abspath('../')
#将我们的类别放到我们的train_dir中 traindir是可修改文件按照个人喜好
#其根目录如下
'''
train_dir --->Detections--->|--->JPEGImages
                            |
                            |--->Annotations
          
'''
TRAIN_DIR = osp.join(abs_path, "data/train_dir")
print(TRAIN_DIR)
TYPE ="bmp"
'''
标定名字的特别注意环节
<annotation>
	<folder>target</folder>
	<filename>20180122-180656338-BW.png</filename>
	<path>/home/qlt/qiulingteng/detect/target/20180122-180656338-BW.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1600</width>
		<height>1200</height>
		<depth>1</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>defect</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>484</xmin>
			<ymin>766</ymin>
			<xmax>996</xmax>
			<ymax>1138</ymax>
		</bndbox>
	</object>
</annotation>
这里我们的名字标定规则如下所示
'''
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(TRAIN_CLASSES, range(len(TRAIN_CLASSES))))
	print (self.class_to_ind)
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
	'''
	bbox coords,class name
	'''
	'''
	<object><name>person</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
		<bndbox><xmin>174</xmin><ymin>101</ymin><xmax>349</xmax><ymax>351</ymax></bndbox>
	'''
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = 1.0*cur_pt / width if i % 2 == 0 else 1.0*cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    '''
    train_dir --->Detections--->|--->JPEGImages
                                |
                                |--->Annotations
                                |
                                |--->img_idx.txt:用于标记是否
    '''
    '''
    root 表示的是我们的根目录---->train_dir
    image_sets--------------->Detections
    trainsform--------------->用于data augmention 具体参见augment.py
    target_transform--------->获取图片的相对位置
    dataset_name------------->存放dataset 文件
    '''
    def __init__(self, root='train_dir',
                 image_sets='Detections',
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='img_idx'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
	    #VOCAnnotationTransform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.{}'.format(TYPE))
        self.ids = list()
        rootpath = osp.join(self.root,image_sets )
        for line in open(osp.join(rootpath,dataset_name + '.txt')):
            self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        print img_id
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        if self.target_transform is not None:
	    #a list containing lists of bounding boxes  [bbox coords, class name]
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width
	# target in here is very easily [xmin,ymin,xmax,ymax,class_idx]

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
if __name__ =="__main__":
    voc_data = VOCDetection()
    img, gts = voc_data[5]
    img = img.numpy().transpose((1,2,0))
    h,w,_ = img.shape
    for gt in gts:
        print (int(gt[0]*w),int(gt[2]*w))
        print (int(gt[1]*h),int(gt[3]*h))
        img = cv2.rectangle(img,(int(gt[0]*w),int(gt[1]*h)),(int(gt[2]*w),int(gt[3]*h)),(0,0,255))
    cv2.imshow('fuck', img)
    cv2.waitKey()
    print gt



