#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
from torch.autograd import Function
from layers import box_utils
from data import voc as cfg
from  utils import nms_wrapper
class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    '''
    #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    '''
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        #loc_data: bs,8732,4
        #conf_data:bs,8732,21
        #prior_data:8732,4
        #(8732,4)
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        #(1,21,200,5)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        #conf_preds----->(1,21,8732)
        # Decode predictions into bboxes.
        for i in range(num):
            #decode_boxes
            #这里返回的是(x1,y1,x2,y2)
            decoded_boxes = box_utils.decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            #conf_scores ---->(21,8732)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                #将分值太小的剃掉不参与分类
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.shape[0] == 0:
                    continue
                #这里来获取我们的回归值
                #l_mask ------->to(8732,4) ----->if all 1 we must get the bbox
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                scores = scores.unsqueeze(1)
                det = torch.cat([boxes,scores],1)
                #det for here is (x1,y1,x2,y2,scores) 特别注意这里的 x1,y1,x2,y2 对应的是相对位置
                #nms_thresh ----->equal to 0.45
                det = det.detach().cpu().float().numpy()
                #返回的是留下来的bbox
                keep = nms_wrapper.nms(det,self.nms_thresh, False)
                count = len(keep)

                output[i, cl, :count] = \
                    torch.cat((scores[keep[:count]],
                               boxes[keep[:count]]), 1)
        #(1,4200,5)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        mask = (rank>self.top_k).unsqueeze(-1).expand_as(flt)
        flt[mask] = 0
        return output
#detector = detection.Refine_Detect(num_classes,0,cfg,object_score=0.01)
class Refine_Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, cfg, object_score=0):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.object_score = object_score
        # self.thresh = thresh

        # Parameters used in nms.
        self.variance = cfg['variance']

    def forward(self, predictions, prior, arm_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf = predictions
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        '''
        loc_data ---->predict_loc
        conf_data --->predict_conf
        '''
        if arm_data:
            arm_loc, arm_conf = arm_data
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            #根据论文里说的那样如果其阈值小于0.01 则可丢弃
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= self.object_score
            conf_data[no_object_index.expand_as(conf_data)] = 0

        self.num_priors = prior_data.size(0)
        #用于最后的预测使用
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, self.num_priors,
                                        self.num_classes)
            self.boxes.expand(num, self.num_priors, 4)
            self.scores.expand(num, self.num_priors, self.num_classes)
        # Decode predictions into bboxes.

        for i in range(num):
            #将arm 位置loc 解码
            if arm_data:
                default = box_utils.decode(arm_loc_data[i], prior_data, self.variance)
                default = box_utils.center_size(default)
            else:
                default = prior_data
            decoded_boxes = box_utils.decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores
        '''
        返回的是decode 后的boxes----->(6375,4)
                        scores----->(6375,21)
        '''
        return self.boxes, self.scores
