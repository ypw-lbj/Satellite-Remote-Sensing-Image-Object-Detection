#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Pengwei Yin
# * Email         : 1043773117@qq.com
# * Create time   : 2018-09-22 11:43
# * Last modified : 2018-09-22 11:43
# * Filename      : train.py
# * Description   : 
# **********************************************************
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import utils.visualizer as vis_tools
viz = vis_tools.Visualizer(save_where="logdir", env="main")
#NET_INIT:
'''
data_set = voc
data_root = ./data/
basenet = vgg16_reducedfc
batch_size = 32
resume = None
start_iter = 0
lr = 1e-3
--momentum = 0.9
weight_decay 5e-4

'''
'''
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
'''

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='./data/VOCdevkit',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='pretrain/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--models', default='SSD300',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_frequency',default=10,type = int)
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
	#这里有个bug bug 主要在于.3版本和.4版本的区别
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
	pass
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    #weight
    os.mkdir(args.save_folder)
if args.models == 'SSD300':
    from ssd import build_net
elif args.models =="FSSD":
    from FSSD_vgg import build_net
def train():
    print args.dataset 
    if args.dataset == 'VOC':
        cfg = voc
	#men_dim = 300
	#MEANS = (104, 117, 123)
        dataset = VOCDetection(root=args.dataset_root,image_sets=[('2012','trainval')],
                               transform= SSDAugmentation(cfg['min_dim'],MEANS = (104, 117, 123))
                               )
    #dataset ---> img(torch vision) ground_truth,h,w
    ssd_net = build_net('train', cfg['min_dim'], cfg['num_classes'])
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        print args.save_folder + args.basenet
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.base.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        if args.models =="FSSD":
            ssd_net.ft_module.apply(weights_init)
            ssd_net.pyramid_ext.apply(weights_init)
    #lr->1e-3,momentum--->0.9,weight_decay---> 5e-4
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    #num_classes, overlap_thresh, prior_for_matching,
    #bkg_label, neg_mining, neg_pos, neg_overlap, encode_target
    criterion = MultiBoxLoss(num_classes = cfg['num_classes'], overlap_thresh=0.5,prior_for_matching= True,bkg_label= 0, neg_mining=True, neg_pos = 3,
                             neg_overlap = 0.5,encode_target = False, use_gpu = args.cuda)
    sche = torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['lr_steps'],gamma= args.gamma)
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')
    print(len(dataset))
    #360
    epoch_size = len(dataset) // args.batch_size
    print epoch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch_on ' + dataset.name
        vis_legend = ['train/loc_loss', 'train/conf_loss', 'train/total_loss','train/epoch']
    print "batch size for here is {}".format(args.batch_size)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=4,shuffle=True,collate_fn=detection_collate)

    # create batch iterator
    # output[0]----->(8732,4)   bbox
    # output[1]----->(8732,21)  cls_scores
    # output[2]----->(8732,4)   pirors
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            #updata_step
            viz.line(tag="train/train_epoch",scalar_value=epoch,ite = iteration)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1
            batch_iterator = iter(data_loader)
            torch.save(ssd_net.state_dict(), 'weights/' +str(args.models)+
                       repr(iteration) + '.pth')
        #学习
        sche.step()
        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        # output[0]----->(8732,4)   bbox
        # output[1]----->(8732,21)  cls_scores
        # output[2]----->(8732,4)   pirors
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))

        if args.visdom:
            #draw loss_function
            viz.line(tag = "train/loc_loss",scalar_value=loss_l.item(),ite = iteration)
            viz.line(tag = "train/conf_loss",scalar_value=loss_c.item(),ite = iteration)
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/' +str(args.models)+
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    print m
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0




if __name__ == '__main__':
    train()
