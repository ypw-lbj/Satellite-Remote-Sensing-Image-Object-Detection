#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOC_1536, VOCDetection, detection_collate, BaseTransform,VOC_512,Test_Crop
from utils.augmentations import SSDAugmentation
from utils.voc_eval import voc_eval
from layers.modules import RefineMultiBoxLoss
from layers.functions import PriorBox
import detection
import time
from utils.nms_wrapper import nms
from utils.timer import Timer
from nms_util.nms_wrapper import soft_nms
import tqdm
import cv2
import shutil
from tensorboardX import SummaryWriter
import pickle

MEANS = (101, 101, 101)
colorList = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(128,128,0),(128,0,128),(0,128,128),
             (128,0,255),(0,128,255),(255,128,0),(255,0,128),(255,128,128),(25,100,255),(255,20,100)]
classnames = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle',
			  5:'large-vehicle', 6:'ship', 7:'tennis-court',8:'basketball-court', 9:'storage-tank',
			  10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool', 14:'helicopter'}


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')

parser.add_argument('--dataset_root', default='./data/VOCdevkit',
                    help='Dataset root directory path')

parser.add_argument('--basenet', default='resnet152.pth',
                    help='Pretrained base model')

parser.add_argument('-v', '--version', default='Refine_res',
                    help='Refine_vgg'or'Refine_res')
parser.add_argument('-s', '--size', default='1536',
                    help='320 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=5,
                    type=int, help='Batch size for training')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-7, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max','--max_epoch', default=400,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--result_folder', default='result',
                    help='save the detect result')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pre_train', default='pretrain/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_frequency',default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency',default=10)
parser.add_argument('--visdom',default=True,type=bool,help="see or not")
parser.add_argument('--fire',default="test",type=str,help="train or test")
parser.add_argument('--checkpoint',default="Refine_res_130.pth",type=str,help="train or test")
parser.add_argument('--save_thresh',default=-0.65,type=float,help="thresh to save")
parser.add_argument('--subsize', default = 400, type = int, help = 'split interval')
args = parser.parse_args()

if args.dataset == 'VOC':
    train_sets = [('2007', 'train'), ('2012', 'train')]
    cfg = (VOC_1536)
# print(cfg)
img_dim = 1536
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = 16
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
# if args.visdom:
#     from utils import visualizer
#     viz = visualizer.Visualizer("log_dirs_"+str(args.size))


from RefineSSD_res152 import build_net
#这边特别注意下一为什么我们不需要maxsize 因为我们的ratio 只取3个也就说我们的ratio 只取（2,1,0.5） 并不需要两个ratio为1.0
cfg = VOC_1536
net = build_net(args.size, num_classes,use_refine=True)
# print(net.size)

if not args.resume_net:
    print(args.basenet)
    state_dict = torch.load(os.path.join(args.pre_train,args.basenet))

    print('Loading base network...')

    new_state_dict = net.base.state_dict()
    for index,(name, v_new) in enumerate(new_state_dict.items()):
        v_new.copy_(state_dict[name])
    net.base.load_state_dict(new_state_dict)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.trans_layers.apply(weights_init)
    net.latent_layrs.apply(weights_init)
    net.up_layers.apply(weights_init)
    net.arm_loc.apply(weights_init)
    net.arm_conf.apply(weights_init)
    net.odm_loc.apply(weights_init)
    net.odm_conf.apply(weights_init)
else:
    resume_path = os.path.join(args.save_folder,'Refine_res_{}.pth'.format(args.resume_epoch))
    resume_dics = torch.load(resume_path)
    print('loading model：',resume_path)
    net.load_state_dict(resume_dics)
    # net.inter_layerr.apply(weights_init)

# state_dict = torch.load(os.path.join(args.pre_train,"Refine_res_220.pth"))
#
# print('Loading previous network...')
#
# new_state_dict = net.state_dict()
# for index,(name, v_new) in enumerate(new_state_dict.items()):
#     v_new.copy_(state_dict[name])
# net.load_state_dict(new_state_dict)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=0.9)
#lr decay
stone = [int(i*args.max_epoch)for i in cfg["lr_epochs"]]
lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer,stone,gamma=0.2)
#(num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,object_score = 0):
arm_criterion = RefineMultiBoxLoss(num_classes=2, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0,neg_mining= True,neg_pos= 3, neg_overlap=0.5,encode_target= False)
odm_criterion = RefineMultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3,neg_overlap= 0.5, encode_target= False,object_score =0.01)
priorbox = PriorBox(cfg)
# detector = detection.Detect(num_classes,0,top_k= 200 ,conf_thresh = 0.01,nms_thresh=0.45)
detector = detection.Refine_Detect(num_classes,0,cfg,object_score=0.01)
#他在这一步来实现detach from
with torch.set_grad_enabled(False):
    priors = Variable(priorbox.forward())
    if torch.cuda.is_available():
        priors = priors.cuda()
#dataset
print('Loading Dataset...')
if args.dataset == 'VOC':
    train_dataset = VOCDetection(root=args.dataset_root,image_sets=[('2012','train')],
                               transform= SSDAugmentation(1536,MEANS)
                               )
    test_dataset = VOCDetection(root=args.dataset_root,image_sets=[('2012','visual')],transform = None
                               )
    #
    # for i in range(5):
    #     for iter, data in enumerate(train_dataset):
    #       print(iter)
    # xx
    # for iter, data in enumerate(train_dataset):
    #     print(iter)
    #     xx


else:
    print('Only VOC and COCO are supported now!')
    exit()

def train():
    #output here is
    '''
    anchor_loc------->(1, 6375, 4)
    anchor_cls------->(1, 6375, 2)
    class_loc-------->(1, 6375, 4)
    class_cls------->(1, 6375, 21)
    '''
    writer = SummaryWriter("./logdir")
    # if torch.cuda.is_available():
    #     writer.add_graph(net.cpu(), torch.rand(1, 3, 512, 512))
    #     net.cuda()
    net.train()
    # loss counters
    loc_loss = 0    # epoch
    conf_loss = 0
    epoch = 0
    if args.resume_net:
        epoch = 0 + args.resume_epoch

    epoch_size = len(train_dataset) // args.batch_size
    print(epoch_size)
    #最大迭代部分
    #here we set the max_epoch for here is 200
    max_iter = args.max_epoch * epoch_size
    print(max_iter)
    #144200 iteration
    print('Training',args.version, 'on', train_dataset.name)
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    batch_iterator = None
    mean_odm_loss_c = 0
    mean_odm_loss_l = 0
    mean_arm_loss_c = 0
    mean_arm_loss_l = 0
    net.train()
    print(start_iter)
    for iteration in range(start_iter, max_iter+10):
        if (iteration % epoch_size == 0):
            net.train()
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=1, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(),os.path.join(args.save_folder,"{}_{}.pth".format(args.version,epoch)))
            epoch += 1
            lr_sche.step()
            net.train()
        load_t0 = time.time()



        # load train data
        images, targets = next(batch_iterator)
        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))
        images = Variable(images.cuda())
        with torch.set_grad_enabled(False):
            targets = [Variable(anno.cuda()) for anno in targets]
        # forward
        '''
        anchor_loc------->(1, 6375, 4)
        anchor_cls------->(1, 6375, 2)
        class_loc-------->(1, 6375, 4)
        class_cls------->(1, 6375, 21)
        '''
        out = net(images)
        arm_loc, arm_conf, odm_loc, odm_conf = out
        # backprop
        optimizer.zero_grad()
        #arm branch loss
        print(priors.type())
        arm_loss_l,arm_loss_c = arm_criterion((arm_loc,arm_conf),priors,targets)
        #odm branch loss
        odm_loss_l, odm_loss_c = odm_criterion((odm_loc,odm_conf),priors,targets,(arm_loc,arm_conf),False)

        mean_arm_loss_c += arm_loss_c.item()
        mean_arm_loss_l += arm_loss_l.item()
        mean_odm_loss_c += odm_loss_c.item()
        mean_odm_loss_l += odm_loss_l.item()

        loss = arm_loss_l+arm_loss_c+odm_loss_l+odm_loss_c
        loss.backward()
        optimizer.step()
        print("iteration: ",iteration)
        print('ARM:|| ARM_Loss_c:{:.4f} || ARM_Loss_l:{:.4f}'.format(arm_loss_c.item(), arm_loss_l.item()))
        print('ODM:|| ODM_Loss_c:{:.4f} || ODM__Loss_l:{:.4f}'.format(odm_loss_c.item(), odm_loss_l.item()))

        writer.add_scalar("ARM_Loss_c/iteration", arm_loss_c.item(), iteration)
        writer.add_scalar("ARM_Loss_l/iteration", arm_loss_l.item(), iteration)
        writer.add_scalar("ODM_Loss_c/iteration", odm_loss_c.item(), iteration)
        writer.add_scalar("ODM_Loss_l/iteration", odm_loss_l.item(), iteration)


        load_t1 = time.time()
        if iteration % 100 == 0 and iteration>0:
            print('timer: %.4f sec.' % (load_t1 - load_t0))
            print('iter ' + repr(iteration) + ' || Loss:{:.4f} || '.format(loss.item()))
            print('MEAN_ARM:|| ARM_Loss_c:{:.4f} || ARM_Loss_l:{:.4f}'.format(mean_arm_loss_c/100.0, mean_arm_loss_l/100.0))
            print('MEAN_ODM:|| ODM_Loss_c:{:.4f} || ODM__Loss_l:{:.4f}'.format(mean_odm_loss_c/100.0, mean_odm_loss_l/100.0))

            writer.add_scalar("MEAN_ARM_Loss_c/iteration", mean_arm_loss_c, iteration)
            writer.add_scalar("MEAN_ARM_Loss_l/iteration", mean_arm_loss_l, iteration)
            writer.add_scalar("MEAN_ODM_Loss_c/iteration", mean_odm_loss_c, iteration)
            writer.add_scalar("MEAN_ODM_Loss_l/iteration", mean_odm_loss_l, iteration)

            mean_odm_loss_c = 0
            mean_odm_loss_l = 0
            mean_arm_loss_c = 0
            mean_arm_loss_l = 0
    # torch.save(net.state_dict(), os.path.join(args.save_folder ,
    #            'Final_' + args.version +'_' + args.dataset+ '.pth'))
    # writer.export_scalars_to_json( './loddir'+ os.sep + "all_logs.json")
    # writer.close()
def splitImage(image, subsize, lap):
	height,width = image.shape[:2]
    # picture = image.copy()
	left, up = 0, 0
	points = []
	while up < (height ):
		if (up + subsize) >= height:
			up = max(height - subsize, 0)
		left = 0
		while left < (width - 1):
			if (left + subsize) >= width:
				left = max(width - subsize, 0)
			points.append((left,up,left,up))
			if left + subsize >= width:
				break
			else:
				left = left + subsize -lap
		if up + subsize >=height:
			break
		else:
			up = up + subsize - lap

	images = []
	for point in points:
		images.append((image[point[1]:point[1]+subsize, point[0]:point[0]+subsize, :]).copy())

	return images, points

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * args.warm_epoch)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def test():
    path = os.path.join(args.save_folder,args.checkpoint)
    print('loading model:',path)
    dics = torch.load(path)
    net.load_state_dict(dics)
    net.eval()
    net.cuda()
    print(net.size)
    test_net(args.result_folder, net, detector,args.cuda, test_dataset,
             BaseTransform(1536, MEANS),
             thresh=0.0000001)
#######################################################################################################################
    if 0:
        file = open('/home/cv/ypw/ObjectDetection/experienmentRecord.txt','a')
        file.write('\n')
        file.write(10*'####'+'\n')
        print(time.asctime( time.localtime(time.time())),file = file)
        print(os.getcwd(),file=file)
        print('soft-nms test',file=file)

        print('subsize:',args.subsize,file=file)
        # print('initial learing rate:',args.lr,file=file)
        # print("lr_epochs:", cfg["lr_epochs"],file = file)
        # print("min_sizes:", cfg["min_sizes"],file = file)
        print('checkpoint:',args.checkpoint,file=file)
        # with open('refinedet_train_test.py','r') as main:
        #     lines = main.readlines()
        #     Flag = False
        #     for line in lines:
        #         if line.strip() == '###start':
        #             Flag = True
        #         if Flag:
        #             file.write(line)
        #         if line.strip() == '###end':
        #             Flag = False
        # main.close()
        results = []
        testfilepath = os.path.join(args.dataset_root,args.dataset+'2012','ImageSets', 'Main', "val" + '.txt')
        annopath = os.path.join(args.dataset_root,args.dataset+'2012','Annotations/'+'{}.xml')
        for classname in (classnames.values()):
            detpath = os.path.join(os.path.join('./save',str(args.subsize)),('Task2_'+classname+'.txt'))
            print(classname, voc_eval(detpath, annopath, testfilepath, classname)[2])
            print(classname, voc_eval(detpath, annopath, testfilepath, classname)[2],file=file)
            results.append(voc_eval(detpath, annopath, testfilepath, classname)[2])
        for score in results:
            print(score)
        results = np.array(results)
        print('mAP : ', results.mean())
        print('mAP : ', results.mean(),file = file)
        file.close()

def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=2000, thresh=0.005,test_tresh = 0.6):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    t_c = Test_Crop()
    num_images = len(testset)
    ind_to_class ={}
    for i in (testset.class_to_ind):
        ind_to_class[testset.class_to_ind[i]] = i
    for i in ind_to_class:
        if os.path.exists(os.path.join(args.result_folder,ind_to_class[i])):
            shutil.rmtree(os.path.join(args.result_folder,ind_to_class[i]))
        os.mkdir(os.path.join(args.result_folder,ind_to_class[i]))
    print(ind_to_class)
    num_classes = (16, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]         # ！！！
                 for _ in range(num_classes)]
    _t = {'im_detect': Timer(), 'misc': Timer()}
    net.eval()

    startTest = 0
    if startTest == 0:
        if os.access(os.path.join('./save',str(args.subsize)), os.F_OK):
            os.system('rm -rf ' + os.path.join('./save',str(args.subsize)))
            os.system('mkdir ' + os.path.join('./save',str(args.subsize)))
        else:
            os.system('mkdir ' + os.path.join('./save',str(args.subsize)))
    with torch.set_grad_enabled(False):
        # for i in tqdm.trange(0,num_images):
        for i in tqdm.trange(startTest,num_images):
            subsize = 0
            img = testset.pull_image(i)                  # pull image from index
            picture = img.copy()
            # img = cv2.imread('test.jpg')
            # img = img[:, :, (2,1,0)]

            h,w = img.shape[0:2]

###start
            lap = 0
            subsize = args.subsize
            # if h*w <= 3000*3000:
            #     subsize =500
            # elif h*w <= 4000*4000:
            #     subsize = 600
            # elif h*w <= 5000*5000:
            #     subsize = 800
            # elif h*w <= 6000*6000:
            #     subsize = 1000
            # elif h*w <= 7000*7000:
            #     subsize = 1200
            # else:
            #     subsize = 1500
            # lap = int(subsize * 0.4)
###end

            print("img.shape: ",img.shape)
            images, pts = splitImage(img, subsize, lap)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)             # ? ? ?
            images.append(img)
            pts.append((0,0,0,0))
            points = []
            for point in pts:
                point = torch.Tensor(point)
                points.append(point)
            print("len(points): ",len(points))

            images = images[-1:]
            points = points[-1:]
            imgs = []
            for img in images:
                x1 = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).contiguous()
                x1 = Variable(x1.unsqueeze(0))
                imgs.append(x1)

            total_patch = len(imgs)
            test_batch = 1
            times = total_patch // test_batch
            last_patch = total_patch % test_batch
            if last_patch > 0:
                times += 1
            boxesList = []
            scoresList = []
            for time in range(0,times):
                if time == times-1 and last_patch > 0:

                    images_temp = imgs[test_batch*time:test_batch*time+last_patch]
                    x = torch.cat(images_temp,dim = 0)
                    if cuda:
                        x = x.cuda()
                    out = net(x=x, test=True)  # forward pass
                    arm_loc,arm_conf,odm_loc,odm_conf = out
                    boxes, scores = detector.forward((odm_loc,odm_conf), priors,(arm_loc,arm_conf))
                    detect_time = _t['im_detect'].toc()
                    boxes = boxes
                    scores=scores

                    boxes = boxes.cpu().numpy()
                    boxesList.append(boxes)
                    scores = scores.cpu().numpy()
                    scoresList.append(scores)
                    # print('scores.shape:',scores.shape)
                    # print('boxes.shape:',boxes.shape)
                else:
                    images_temp = []
                    images_temp = imgs[test_batch * time:test_batch * (time+1)]
                    x = torch.cat(images_temp,dim = 0)
                    if cuda:
                        x = x.cuda()
                    out = net(x=x, test=True)  # forward pass
                    arm_loc,arm_conf,odm_loc,odm_conf = out
                    boxes, scores = detector.forward((odm_loc,odm_conf), priors,(arm_loc,arm_conf))
                    detect_time = _t['im_detect'].toc()
                    boxes = boxes
                    scores=scores

                    boxes = boxes.cpu().numpy()
                    boxesList.append(boxes)
                    scores = scores.cpu().numpy()
                    scoresList.append(scores)

            boxes = np.concatenate(tuple(boxesList),0)
            scores = np.concatenate(tuple(scoresList),0)

            scale = torch.Tensor([int(subsize), int(subsize),
                                  int(subsize), int(subsize)]).cpu().numpy()
            scale_orign = torch.Tensor([w, h, w, h]).cpu().numpy()
            # decode
            boxes[:-1]= boxes[:-1]*scale
            boxes[-1]=boxes[-1]*scale_orign

            # encode
            for p in range(boxes.shape[0]):
                box = boxes[p]
                point = points[p]
                # print(point)
                point = point.unsqueeze(0).expand_as(torch.from_numpy(box)).cpu().numpy()
                boxes[p]+= point

            # print(type(boxes[0]))
            print((boxes[0].shape))

            boxes = np.concatenate((tuple(boxes)), axis=0)
            scores = np.concatenate(tuple(scores),axis = 0 )
            print("boxes.shape",boxes.shape)

            # nms postProcessing
            for j in range(1, num_classes):
                '''
                特别注意这里我们的输入只有一个num 所以这一步我们可以看成没有循环
                
                '''
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)         # shadow copy
                '''
                
                '''
                if args.dataset == 'VOC':           # using gpu
                    cpu = True
                else:
                    cpu = False
# ###start
#                 nms_thresh = 0.3
#                 if(j==4 or j==9 or j==11 or j==12 or j==15):
#                     nms_thresh = 0.3
#                 elif(j==1):
#                     nms_thresh = 0.35
#                 elif(j==2 or j==5 or j==10):
#                     nms_thresh = 0.4
#                 elif(j==13 or j==14):
#                     nms_thresh = 0.45
#                 elif(j==6 or j==8):
#                     nms_thresh = 0.5
#                 elif(j==7):
#                     nms_thresh = 0.55
#                 else:
#                     nms_thresh = 0.6
#                 keep = soft_nms(c_dets,Nt=nms_thresh)
#                 # keep = nms(c_dets, nms_thresh, force_cpu=cpu)
#                 keep = keep[:2000]
#################################################################################################3
#                 nms_thresh = 0.3
#                 if(j==4 or j==8 or j==9 or j==14):
#                     nms_thresh = 0.3
#                 elif(j==1 or j==5):
#                     nms_thresh = 0.35
#                 elif(j==2 or j==3 or j==10 or j==13):
#                     nms_thresh = 0.4
#                 elif(j==7 or j==15):
#                     nms_thresh = 0.45
#                 elif(j==6 or j==11):
#                     nms_thresh = 0.5
#                 else:
#                     nms_thresh = 0.6
#
#                 keep = soft_nms(c_dets,Nt=nms_thresh)
#                 keep = keep[:2000]
# ###end
###start
                nms_thresh = 0.3
                if(j==4 or j==9 or j==11 or j==12 or j==15):
                    nms_thresh = 0.3
                elif(j==1):
                    nms_thresh = 0.35
                elif(j==2 or j==5 or j==10):
                    nms_thresh = 0.4
                elif(j==13 or j==14):
                    nms_thresh = 0.45
                elif(j==6 or j==8):
                    nms_thresh = 0.5
                elif(j==7):
                    nms_thresh = 0.55
                else:
                    nms_thresh = 0.6
                keep = nms(c_dets, nms_thresh, force_cpu=cpu)
                keep = keep[:200]
###end
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
################### write result ##################################################################################################
#             for j in range(1,num_classes):
#                  boxes = np.asarray(all_boxes[j][i])
#                  if(boxes.shape[0] !=0):
#                      for z in range(boxes.shape[0]):
# #                          xmin, ymin, xmax, ymax = boxes[z][0], boxes[z][1],boxes[z][2],boxes[z][3]
# #                          width = xmax - xmin
# #                          height = ymax - ymin
# # ###start
# #                          if max(width/w,height/h) > 0.005:
# # ###end
#                          line = ''
#                          with open(os.path.join(os.path.join('./save',str(args.subsize)),'Task2_'+classnames[j-1]+".txt"),'a') as writer:
#                             file_name = testset.ids[i][1]
#                             writer.write("{} {:.6f} {} {} {} {}\n".format(file_name,boxes[z][4],max(int(boxes[z][0]),0),max(int(boxes[z][1]),0),min(int(boxes[z][2]),w-1),min(int(boxes[z][3]),h-1)))
#                          writer.close()
# ################### visulize result ##################################################################################################
            for j in range(1,num_classes):
                boxes = np.asarray(all_boxes[j][i])
                boxes= boxes[boxes[:,4]>0.2]
                boxes[:,:4] = boxes[:, :4]/3
                if(boxes.shape[0] !=0):
                    for z in range(boxes.shape[0]):
                        # print(ind_to_class[j-1])
                        xmin, ymin, xmax, ymax = boxes[z][0], boxes[z][1],boxes[z][2],boxes[z][3]
                        width = xmax - xmin
                        height = ymax - ymin
                        if max(width/w,height/h) > 0.001:
                            # print(max(width/w,height/h))
                            cv2.rectangle(picture,(int(boxes[z][0]),int(boxes[z][1])),(int(boxes[z][2]),int(boxes[z][3])),colorList[j],3)
                        times = _t["misc"].toc()-detect_time
            file_name = testset.ids[i][1]
            # save_path = os.path.join('picsave',file_name+'.png')
            # cv2.imwrite(save_path, picture)
            # print(times)
            cv2.namedWindow("enhanced", 0);
            cv2.resizeWindow("enhanced", 800, 800);
            cv2.imshow("enhanced", picture)
            cv2.waitKey()
#####################################################################################################################

if __name__ == '__main__':
    if args.fire == "train":
        train()
    else:
        test()
