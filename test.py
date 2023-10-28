#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-09-26 17:17
# * Last modified : 2018-09-26 17:17
# * Filename      : test.py
# * Description   : 
# **********************************************************
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
import tqdm
import cv2
import detection
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/FSSD119686.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='result/', type=str,
                    help='Dir to save results')
parser.add_argument('--test_file_index', default='./test.txt', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='./data/VOCdevkit', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--model', default="SSD300", type=str, help="which model will be used")
parser.add_argument('--test', default="None", type=str, help="choice test which img")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    pass
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if args.model == "SSD300":
    from ssd import build_net
elif args.model == "FSSD":
    from FSSD_vgg import build_net

def test_net(save_folder, net, cuda, testset, transform, thresh):
    '''

    :param save_folder: 将结果存在哪儿
    :param net: ssd-300
    :param cuda:
    :param testset: test_loader_set
    :param transform: 我们只要减去均值就可以了没有必要做一些其他变化
    :param thresh: 0.6 Final confidence threshold
    :return:
    '''
    # dump predictions and assoc. ground truth to text file for now
    filename = args.test_file_index
    #5832 test_files

    num_images = len(testset)
    if args.test =="None":
        for i in tqdm.trange(num_images):
            #获取图片
            img = testset.pull_image(i)
            #下面一步是获取coco 的an 的但是我们没有必要使用
            #img_id, annotation = testset.pull_anno(i)
            #transform --->BaseTransform ---->resize and substr the meanval
            x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))
            file_name = testset.pull_file_name(i)
            img = cv2.imread(file_name)
            #input here is 300*300
            if cuda:
                x = x.cuda()
            y = net(x)      # forward pass
            y = net.detect(*y)
            #输出的y 是根据极大值阈值之后的y
            # output[0]----->(8732,4)   bbox
            # output[1]----->(8732,21)  cls_scores
            # output[2]----->(8732,4)   pirors
            detections = y.data
            # scale each detection back up to the image

            scale = torch.Tensor([img.shape[1], img.shape[0],
                                 img.shape[1], img.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.5:
                    score = detections[0, i, j, 0]
                    label_name = labelmap[i-1]
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    img = cv2.rectangle(img,(pt[0],pt[1]),(pt[2],pt[3]),(0,0,255))
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1
                    with open(filename, mode='a') as f:
                        f.write(str(file_name)+' '+label_name+' ' +
                                str(score.item()) +' '+' '.join(str(c) for c in coords) + '\n')
                    j += 1
            cv2.imshow("fuck",img)
            cv2.waitKey()
    else:
        img = cv2.imread(args.test)
        # 下面一步是获取coco 的an 的但是我们没有必要使用
        # img_id, annotation = testset.pull_anno(i)
        # transform --->BaseTransform ---->resize and substr the meanval
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        file_name = args.test
        img = cv2.imread(file_name)
        # input here is 300*300
        if cuda:
            x = x.cuda()
        y = net(x)  # forward pass
        y = net.detect(*y)
        # 输出的y 是根据极大值阈值之后的y
        # output[0]----->(8732,4)   bbox
        # output[1]----->(8732,21)  cls_scores
        # output[2]----->(8732,4)   pirors
        detections = y.data
        # scale each detection back up to the image

        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (0, 0, 255))
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(file_name) + ' ' + label_name + ' ' +
                            str(score.item()) + ' ' + ' '.join(str(c) for c in coords) + '\n')
                j += 1
        cv2.imshow("detect", img)
        cv2.waitKey()


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_net('test', 300, num_classes) # initialize SSD
    static = torch.load(args.trained_model)
    print (type(static))
    for key in  static.keys():
        if 'vgg' in key:
            new_key = key.replace('vgg','base')
            static[new_key] = static[key]
            static.pop(key)
        else:
            temp = static[key]
            static.pop(key)
            static[key] = temp
    net.load_state_dict(static)
    net.eval()
    print('Finished loading model!')
    testset = VOCDetection(args.voc_root, [('2012', 'val')], None, VOCAnnotationTransform())
    img,_ = testset[0]
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
