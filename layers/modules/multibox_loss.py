# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    #   criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,False, args.cuda)
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
	#21
        self.num_classes = num_classes
	#0.5
        self.threshold = overlap_thresh
	#0
        self.background_label = bkg_label
	#false
        self.encode_target = encode_target
	#True
        self.use_prior_for_matching = prior_for_matching
	#true
        self.do_neg_mining = neg_mining
	#3
        self.negpos_ratio = neg_pos
	#0.5
        self.neg_overlap = neg_overlap
	#0.1, 0.2
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
	    #(8172,4),(8172,21),(8172,4)---->(ground_truth,label)


        '''
        这里所使用到的变量做一个梳理
        loc_data ------>(bs,8732,4 ) --------->(各个prior窗的坐标值)
        conf_data------>(bs,8732,21) --------->(各个prior的分数)
        loc_t ------->(bs,8732,4) ------>也就是说我们的prior 和 真值的 差别 的差别(各个prior的分数值dx,dy,dw,dh)，
        在后头需要对我们回归的值和prior 的差别做了 L1 这样就可以使得操作OK
        conf_t ------>(bs,8732) -------->（使我们通过计算prior 和 gt 所得到的可以用来回归的prior）
        通过conf_t  中的信息我们可以得到非常美丽的 pos
        pos（bs，8732） ---->标记着（） 是否取值
        neg（bs，8732）  ----->通过取出1:3的 三倍于正样本的个数并且取损失函数较大的来训练所得到的
        conf_p ------>(bs*(pos+neg),21) ----->通过pos和neg 标记了使得可以合理的取出对应的个数 
        target_labels ---->(pos+neg)---->用于计算最美的值
        '''
        loc_data, conf_data, priors = predictions
	    #get_batch_size
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
	    #8732
        num_priors = (priors.size(0))
	    #21
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
	    #(boxes,4)
            truths = targets[idx][:, :-1].data
	    #(boxes,1)
            labels = targets[idx][:, -1].data
	    #(8732,4)
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        # loc_t 只是encode了一下
        # con_t
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        #计算每个batch的 正样本的个数 也即非背景个数
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        #smooth
        # Shape: [batch,num_priors,4]
        #pos ---->(32,8732)   , loc_data ------>(32,8732,4)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        #获取所得到的点的正信息，的坐标将其取出来 意味着loc_我们只取正样本
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        #here conf_data --->(bs,8732,21),conf_t--->(32,8732)
        #这里也就是意味着，batch——conf 是我们所预测的，而 conf_t 使我们真真的iou 其是通过
        #(bs*8732,21)      conf_t--->(means： 这个东西属于那个类别 ， 其中 pos 的对应位置也是在一起的 32,8732)
        #conf_t 是和 groundtruth 和我们的 prior 核对出来的并且要求 ious 要大于 0.5
        batch_conf = conf_data.view(-1, self.num_classes)
        #log_sum_exp ---->(#return thing ---> (bs*pirors,1))
        '''
        here is gather using method
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        '''
        #这个步骤非常的巧妙，他所获取的是这样的一个的他gather 一个这样子的东西例如 batch_conf --->(bs*prior,21) , 我们只要收集的是二个维度的对应的gt 的val 这个操作非常的巧妙也是我们需要特地去观察的
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #loss_c -->(bs*piror,1)
        # Hard Negative Mining
        '''
        1.根据正样本的个数和正负比例，确定负样本的个数，negative_keep
        2.找到confidence loss最大的negative_keep个负样本，计算他们的分类损失之和
        3.计算正样本的分类损失之和，分类损失是正样本和负样本的损失和
        4.计算正样本的位置损失localization loss.无法计算负样本位置损失
        5. 对回归损失和位置损失之和

---------------------

本文来自 ukuu 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/tomxiaodai/article/details/82354720?utm_source=copy 
        '''
        #pos ---->(32,8732)
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        #32,8732 ----> loss_c
        #将所有的值从大到小排列，主要这里是训练难训练的背景部分
        _, loss_idx = loss_c.sort(1, descending=True)
        # 对我们的已经好了的从大到小的坐标在排一下
        #这个算法很妙 取得是我对应 idx 的rank 所属位置
        _, idx_rank = loss_idx.sort(1)
        # pos ---->(32,8732)  idx_rank --->(32,8732)
        num_pos = pos.long().sum(1, keepdim=True)
        #num_pos ---->(32,1)
        #negpos_ratio ----> 3
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        #num_neg--------->32,1
        #取出我们需要的index
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        #conf_data--->((32,8172,21))
        #pos---->(32,8732,1) ----------> 统计的是正样本的 如果是为真
        #conf_t -----> for me . we use the gt and bbox get iou (32,8732)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)


        target_labels = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, target_labels, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum().type_as(loss_c)
        if N == 0:
            raise BaseException()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
