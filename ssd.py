import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import detection
#return for here we know is 
#output[0]----->(8732,4)   bbox
#output[1]----->(8732,21)  cls_scores
#output[2]----->(8732,4)   gt_bbox
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
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
        self.cfg = (coco, voc)[num_classes == 21]
	#priorbox
        self.priorbox = PriorBox(self.cfg)
        #volatile ---->means this part we needn't autograd.
        with torch.set_grad_enabled(False):
                self.priors = self.priorbox.forward()
                self.priors = self.priors.to("cuda")
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        #in channels 512, gamma 20 for weight norm
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = detection.Detect(num_classes=num_classes,bkg_label= 0,top_k =  10,conf_thresh =  0.01, nms_thresh = 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        #special need to know for here the return is\
        sources = list()
        loc = list()
        conf = list()

        # apply base up to conv4_3 relu
        # conv4_3 relu max_pooling end
        for k in range(23):
            x = self.base[k](x)
	
	    #L2Norm for dim = 1
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x)
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        #compute all loc-->(4*cfg[k])
        #compute all conf --->(21*cfg[k])
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        #in here we do it only for view because last dim we plan to save the loc
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        #here we must be careful
        #the softmax  --------------->self.softmax = nn.Softmax(dim=-1)last dim we use softmax
        #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type_as(x)                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        #special need to know for here the return is 
	    #output[0]----->(8732,4)   bbox
        #output[1]----->(8732,21)  cls_scores
        #output[2]----->(8732,4)   pirors
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
#this part we dispute the softmax layer.
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    #vgg: conv---->fc7(1024 channels)
    #extra_layers(512,256,256,256)
    #cfg---->(4, 6, 6, 6, 4, 4)
    #in here the paper is wrong we means the conv4_3
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    print(len(vgg))
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
	print(k,v)
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

#basenet
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
#extractnet
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
#number of boxes per feature map location
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
if __name__ =="__main__":
    net = build_net('train', size=300, num_classes=21)
    x = torch.randn(1,3,300,300)
    out = net(x)
    for i in out:
	print (i.shape)
    
