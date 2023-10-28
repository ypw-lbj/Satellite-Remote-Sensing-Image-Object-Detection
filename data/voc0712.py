"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
# from config import HOME
import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
from random import uniform
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
# VOC_CLASSES = (  # always index 0
#
#     'airport', 'bridge', 'harbor', 'airplane',
#     'oilcan', 'boat')
VOC_CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
			  'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',
			  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
classToId = {'plane':0, 'baseball-diamond':1, 'bridge':2, 'ground-track-field':3, 'small-vehicle':4,
              'large-vehicle':5, 'ship':6, 'tennis-court':7,'basketball-court':8, 'storage-tank':9,
              'soccer-ball-field':10, 'roundabout':11, 'harbor':12, 'swimming-pool':13, 'helicopter':14}
# note: if you used our download scripts, this should be right
abs_path = osp.abspath('./')
VOC_ROOT = osp.join(abs_path, './dataVOCdevkit')
print (VOC_ROOT)
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
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        # print (self.class_to_ind)
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
                # cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = int(bbox.find(pt).text)
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

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
	    #VOCAnnotationTransform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._txtpath = osp.join('%s', 'labelTxt', '%s.txt')

        self._imgpath = osp.join('%s', 'JPEGImages', '%s.png')
        self.class_to_ind = dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)
    # def readInformation(self, fileName):
    def readLabel(self,txtFile):
        locations = []
        classification = []
        with open(txtFile,'r') as file:
            lines = file.readlines()
            for line in lines:
                information = line.strip().split(' ')
                if len(information) > 8:
                    coordinate = np.ones((3,4))
                    coordinate[0][0] = int(information[0])
                    coordinate[0][1] = int(information[2])
                    coordinate[0][2] = int(information[4])
                    coordinate[0][3] = int(information[6])

                    coordinate[1][0] = int(information[1])
                    coordinate[1][1] = int(information[3])
                    coordinate[1][2] = int(information[5])
                    coordinate[1][3] = int(information[7])
                    locations.append(coordinate)
                    classification.append(int(classToId[information[8]]))
        # print(len(locations))
        # print(len(classification))

        return locations, classification

    def rotation(self,image,labels):

        angle = uniform(-45, 45)
        # print('angle',angle)
        # angle = 0 /180.0 * PI
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)

        # matrixRotation = np.array([[math.cos(angle),-1*math.sin(angle),0],[math.sin(angle),math.cos(angle),0]])
        matrixRotation = cv2.getRotationMatrix2D(center, angle, 1)
        # print(matrixRotation)
        points = np.dot(matrixRotation , np.array([[0,width-1,0,width-1],[0,0,height-1,height-1],[1,1,1,1]]))
        # print(points)
        W = np.max(points[0]) - np.min(points[0])
        H = np.max(points[1]) - np.min(points[1])
        W_translate = np.min(points[0])
        H_translate = np.min(points[1])
        matrixRotation[0,2] -= W_translate
        matrixRotation[1,2] -= H_translate

        # print(W,H)
        image_warp = cv2.warpAffine(image, matrixRotation,(int(W-1),int(H-1)))
        coordS = []
        for label in labels:
            label = np.dot(matrixRotation,label)
            # print(label)
            top_left = (int(np.min(label[0,:])), int(np.min(label[1,:])))
            low_right = (int(np.max(label[0,:])),int(np.max(label[1,:])))
            coordinate = np.array([int(np.min(label[0,:])), int(np.min(label[1,:])),
                                   int(np.max(label[0,:])),int(np.max(label[1,:]))])
            coordS.append(coordinate)

        coordS = np.asarray(coordS, dtype=np.uint32)
        return image_warp, coordS

    def pull_file_name(self,index):
        img_id = self.ids[index]
        return osp.abspath(self._imgpath%img_id)
    def pull_item(self, index):
        img_id = self.ids[index]
        # print("img_id: ",img_id)
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        # print("img.shape: ",img.shape)
        if self.target_transform is not None:
	    #a list containing lists of bounding boxes  [bbox coords, class name]
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            target[:,0] = np.clip(target[:,0], 0, width-1)
            target[:,1] = np.clip(target[:,1], 0, height-1)
            target[:,2] = np.clip(target[:,2], 0, width-1)
            target[:,3] = np.clip(target[:,3], 0, height-1)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

#######################################################################################
            if 0:
                tempBox = list(boxes)
                with open("anchor.txt",'a') as file:
                    for box in tempBox:
                        print(str(box[2]-box[0]) + ' ' + str(box[3]-box[1]), box[4])
                        # line = ""
                        # line = str(box[2]-box[0]) + ' ' + str(box[3]-box[1]) + '\n'
                        # file.write(line)
                    file.close()
#######################################################################################

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
#######################################################################################
        if 0:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # image = img.copy()
            target = list(target)
            # print(type(target))
            print(len(target))
            # print(target.shape)
            for index,_ in enumerate(target):
                # print((int(target[index][0]*width),int(target[index][1]*height)), (int(target[index][2]*width),int(target[index][3]*height)))
                cv2.rectangle(img,(int(target[index][0]*512),int(target[index][1]*512)), (int(target[index][2]*512),int(target[index][3]*512)), (0,2,255), thickness=3)
                # cv2.rectangle(img,(int(target[index][0]),int(target[index][1])), (int(target[index][2]),int(target[index][3])), (0,2,255), thickness=3)

            cv2.namedWindow("image",0)
            cv2.resizeWindow("image",800,800)
            cv2.imshow('image',img)
            cv2.waitKey()

#######################################################################################

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
	voc_data = VOCDetection("./VOCdevkit",image_sets=[('2012','trainval')])
	voc_data[3]	
