#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.augmentations import SSDAugmentation
from layers import *
import pickle

class ConvBnReluLayer(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ExtraLayers(nn.Module):

    def __init__(self, inplanes):
        super(ExtraLayers, self).__init__()
        self.convbnrelu1_1 = ConvBnReluLayer(inplanes, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)


    def forward(self, x):
        out1_1 = self.convbnrelu1_1(x)
        out1_2 = self.convbnrelu1_2(out1_1)
        return out1_2
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,padding =1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

def make_layer(block, inplanes,planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    if dilation!=1:
        pd = int((3 + (dilation - 1) * 2) - 1) / 2
        layers.append(block(inplanes, planes, stride, dilation, downsample,pd))
    else:
        layers.append(block(inplanes, planes, stride, dilation, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)
class ResNetBase(nn.Module):

    def __init__(self, block, layers, width=1, usd_dialtion = False):
        self.inplanes = 64
        widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]
        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        if usd_dialtion == True:
        # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=1, dilation=2)
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2, dilation=1)
        # remove the final avgpool and fc layers
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(widths[3] * block.expansion, num_classes)
        # add extra layers


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if dilation!=1:
            pd = int((3 + (dilation - 1) * 2) - 1) / 2
            layers.append(block(self.inplanes, planes, stride, dilation, downsample,pd))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # count = None
        # file = None
        # if os.access('/home/gxd/ypw/CODE/featureVulization/count.pkl', os.F_OK):
        #     file = open('/home/gxd/ypw/CODE/featureVulization/count.pkl','rb')
        #     count = pickle.load(file)
        #     file.close()
        #     if count >=26:
        #         count = 0
        # else:
        #     # file = open("result.pkl",'wb')
        #     count = 0
        # file = open('/home/gxd/ypw/CODE/featureVulization/count.pkl', 'wb')
        # count += 1
        # print(count)
        # pickle.dump(count,file)
        # file.close()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # saveName3 = '/home/gxd/ypw/CODE/featureVulization/layers1_' + str(count) + '.pkl'
        # file = open(saveName3, 'w')
        # pickle.dump(x, file)
        # file.close()

        file = open('/home/gxd/ypw/CODE/featureVulization/layers1.pkl', 'w')
        pickle.dump(x, file)
        file.close()
        print(x.size())
        x = self.layer2(x)

        # saveName3 = '/home/gxd/ypw/CODE/featureVulization/layers2_' + str(count) + '.pkl'
        # file = open(saveName3, 'w')
        # pickle.dump(x, file)
        # file.close()

        file = open('/home/gxd/ypw/CODE/featureVulization/layers2.pkl', 'w')
        pickle.dump(x, file)
        file.close()
        print(x.size())

        conv3_7 = x
        x = self.layer3(x)
        #
        # saveName3 = '/home/gxd/ypw/CODE/featureVulization/layers3_' + str(count) + '.pkl'
        # file = open(saveName3, 'w')
        # pickle.dump(x, file)
        # file.close()

        file = open('/home/gxd/ypw/CODE/featureVulization/layers3.pkl', 'w')
        pickle.dump(x, file)
        file.close()
        print(x.size())

        conv4_35 =x
        x = self.layer4(x)

        # saveName3 = '/home/gxd/ypw/CODE/featureVulization/layers4_' + str(count) + '.pkl'
        # file = open(saveName3, 'w')
        # pickle.dump(x, file)
        # file.close()

        file = open('/home/gxd/ypw/CODE/featureVulization/layers4.pkl', 'w')
        pickle.dump(x, file)
        file.close()
        print(x.size())

        conv5_2 = x
        return conv3_7,conv4_35,conv5_2

        # return out38x38, out19x19, out10x10, out5x5, out3x3, out1x1
class RefineSSD_Res(nn.Module):
    def __init__(self, size, num_classes, use_refine=False):
        super(RefineSSD_Res, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        # this part means refine our object
        self.size = size
        self.use_refine = use_refine

        # SSD network
        self.base = resnet_base(152)
        # this step we have this 操作
        # vgg 我们可以的得到的是 conv4_3 conv5_3 fc6和fc7 特别注意fc7最后的输出是1024维
        # Layer learns to scale the l2 normalized features from conv4_3
        # 我们先正则化一样在norm 一下并且尺度做一些修改，文章中所论述的那样尺度取得是scale ：10 和 scale 为 8
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True))
        self.extras = nn.Sequential(ConvBnReluLayer(2048, 256, kernel_size=1, padding=0, stride=1),
                                    ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2))

        if use_refine:
            # 为什么这里是12解释一下因为我们有三个ratio 所以这里设置为12 是有意义的3*4=12 回归 三个ratio
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(2048, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1), \
                                          ])
            # 为什么这里设置成6 和上头如初一则
            self.arm_conf = nn.ModuleList([nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(1024, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(2048, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1), \
                                           ])
        # 这里的操作和上头也是一模一样的因为我们都有3个ratio
        self.odm_loc = nn.ModuleList([nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                      ])
        self.odm_conf = nn.ModuleList([nn.Conv2d(1024, 3 * num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(1024, 3 * num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(1024, 3 * num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(1024, 3 * num_classes, kernel_size=3, stride=1, padding=1), \
                                       ])
        '''
        中间层
        '''
        self.inter_layerr = nn.ModuleList([make_layer(Bottleneck,256,256,1),make_layer(Bottleneck,256,256,1),make_layer(Bottleneck,256,256,1),make_layer(Bottleneck,256,256,1)
        ])
        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.BatchNorm2d(256),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.BatchNorm2d(256)),
                                           nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.BatchNorm2d(256),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.BatchNorm2d(256),),
                                           nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.BatchNorm2d(256),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.BatchNorm2d(256))
                                           ])
        self.up_layers = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),nn.BatchNorm2d(256)),
                                        nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),nn.BatchNorm2d(256)),
                                        nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),nn.BatchNorm2d(256)), ])
        self.latent_layrs = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256)),
                                           nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256)),
                                           nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256)),])

        self.softmax = nn.Softmax()

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

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
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()
        obm_iter=list()

        # apply vgg up to conv4_3 relu
        conv3_7,conv4_35,conv5_2 = self.base(x)
        arm_sources.append(conv3_7)
        arm_sources.append(conv4_35)
        arm_sources.append(conv5_2)
        x = conv5_2
        '''
        conv4_3 ---->(1,512,40,40)
        conv5_3-------->(1,1024,20,20)
        fc7--------->(1,2048,10,10)
        '''
        # conv6_2
        x = self.extras(x)
        arm_sources.append(x)
        '''
        上面那些是从ssd 的前两部提了出来这里只做提取特征用
        最后我们得到的是这样的一个feature map
        conv4_3 ---->(1,512,40,40)
        conv5_3----->(1,512,20,20)
        fc7--------->(1,1024,10,10)
        conv6_2----->(1,512, 5 ,5)
        '''

        # apply multibox head to arm branch
        if self.use_refine:
            for (x, l, c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                arm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
        # (1,25500),(1,12750)

        x = self.last_layer_trans(x)
        obm_sources.append(x)

        # get transformed layers
        trans_layer_list = list()
        '''
        对前三次进行计算
        512/512/1024 256 3
        relu
        256 256 3


        '''

        for (x_t, t) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t(x_t))
        # fpn module
        trans_layer_list.reverse()
        # 为何倒过来是有理由自底向下的加回去
        arm_sources.reverse()
        for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layrs):
            x = F.relu(l(F.relu(u(x) + t, inplace=True)), inplace=True)
            obm_sources.append(x)
        obm_sources.reverse()
        '''
        resnet101 we need add iterlayers
        '''
        for s,l in zip(obm_sources,self.inter_layerr):
            obm_iter.append(l(s))
        '''
        obm ----->256 40 40
                  256 20 20
                  256 10 10
                  256 5  5
        '''
        for (x, l, c) in zip(obm_iter, self.odm_loc, self.odm_conf):
            obm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())

        # file = open('/home/gxd/ypw/CODE/featureVulization/obm_loc_list.pkl', 'w')
        # pickle.dump(obm_loc_list, file)
        # file.close()
        #
        # file = open('/home/gxd/ypw/CODE/featureVulization/obm_conf_list.pkl', 'w')
        # pickle.dump(obm_conf_list, file)
        # file.close()

        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)


        # apply multibox head to source layers
        '''
                obm ----->256 40 40
                          256 20 20
                          256 10 10
                          256 5  5
                arm ------512 5 5
                          1024 10 10
                          512 20 20
                          512 40 40
                arm_conf ------->1,12750 ----->(fore? background)
                arm_loc  ------->1,25500
                obm_loc -------->1,25500
                obm_conf ------->1,133875
        '''
        if test:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(arm_conf.view(-1, 2)),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
        else:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    arm_conf.view(arm_conf.size(0), -1, 2),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )
        '''
        anchor_loc------->(1, 6375, 4)
        anchor_cls------->(1, 6375, 2)
        class_loc-------->(1, 6375, 4)
        class_cls------->(1, 6375, 21)
        '''
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_net(size=320, num_classes=21, use_refine=False):

    return RefineSSD_Res(size, num_classes=num_classes, use_refine=use_refine)

def resnet_base(depth, width=1, pretrained=False, **kwargs):
    """Constructs a ResNet base network model for SSD.
    Args:
        depth (int): choose 18, 34, 50, 101, 152
        width (float): widen factor for intermediate layers of resnet
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (depth not in [18, 34, 50, 101, 152]):
        raise ValueError('Choose 18, 34, 50, 101 or 152 for depth')
    if ((width != 1) and pretrained):
        raise ValueError('Does not support pretrained models with width>1')

    name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
    layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                   101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    block_dict = {50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
    model = ResNetBase(block_dict[depth], layers_dict[depth], width, **kwargs)

    return model
    #return RefineSSD_Res_net_101(size, num_classes=num_classes, use_refine=use_refine)
if __name__ == "__main__":
    static = torch.load("./resnet_reducedfc.pth")
    net = build_net(320,use_refine= True)
    net.base.load_state_dict(static)
    '''
    x = torch.randn(1,3,320,320)
    out = net(x)
    for i in out:
        print i.shape
    '''
