# -*- coding: utf-8 -*-
import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    #这里是将中心点变化为非中心点
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    #(bbox,4)
    #(priors,4)
    inter = intersect(box_a, box_b)
    #(bbox,priors)----->value 这个bbox 与这个8172的interesce 的值
    #return here is (bbox,pirors)
    # get IoU metrix  !!!
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def refine_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx,arm_loc):
    """Match each arm bbox with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
        arm_loc: (tensor) arm loc data,shape: [n_priors,4]
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # decode arm box
    #先把我们建议的arm 取出来这里的值在(x1,y1,x2,y2)
    decode_arm = decode(arm_loc,priors=priors,variances=variances)
    # jaccard index
    #truth 和 decode_arm 的overlap
    overlaps = jaccard(
        truths,
        decode_arm
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim = True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim = True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    #我们并没有加background 所以这里要加1, background 默认为1
    conf = labels[best_truth_idx]+1           # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, center_size(decode_arm), variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    #0.5,(boxes,4),(8172,4),(0.1,0.2),(8172,1),(8172,4),(8172,21),(bs)
    #match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #overlap --->(bboxes,priors)---->(num_box,8172)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    #对应的最合适的bbox 与我们的groundtruth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    #[1,bbox]
    # [1,num_priors] best ground truth for each prior
    #我们的prior 和那个bbox 最合适
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    #(1,num_priors)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ensure best prior
    #index_fill_(dim, index, val)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    #把对应的最大相交率的overlap 值设为2 也就是满足了最大的为真的条件
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    #for here truths is the bbox and for our priors
    #这里得到的是我们的某一个先验anchor 对那个bbox 负责
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors*1]
    #这一步把所有的label的值都加了1
    conf[best_truth_overlap < threshold] = 0  # label as background
    #(8172,4)   (8172,4)  variances(0.1,0.2)
    loc = encode(matches, priors, variances)
    #返回的是(变化号后的loc---->(dx,dy,dw,dh)
    #idx 对应的是其的batch mark
    # 相当于 return
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    #这里是将其label返回
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    #所以这个函数的整体操作很简单就是求得我们的bbox 于我们的priors 的重合度，并且特别的求出o
def ARM_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    #0.5,(boxes,4),(8172,4),(0.1,0.2),(8172,1),(8172,4),(8172,21),(bs)
    #match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #overlap --->(bboxes,priors)---->(num_box,8172)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    #对应的最合适的bbox 与我们的groundtruth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    #[1,bbox]
    # [1,num_priors] best ground truth for each prior
    #我们的prior 和那个bbox 最合适
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    #(1,num_priors)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ensure best prior
    #index_fill_(dim, index, val)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    #把对应的最大相交率的overlap 值设为2 也就是满足了最大的为真的条件
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
################################################################################


################################################################################

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    #for here truths is the bbox and for our priors
    #这里得到的是我们的某一个先验anchor 对那个bbox 负责
    #这里为什么不加1是有原因的因为这里我们是个二分类所有的为1了何必在加1 呢
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    #这一步把所有的label的值都加了1
    conf[best_truth_overlap < threshold] = 0  # label as background
    #(8172,4)   (8172,4)  variances(0.1,0.2)
    loc = encode(matches, priors, variances)
    #返回的是(变化号后的loc---->(dx,dy,dw,dh)
    #idx 对应的是其的batch mark
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    #这里是将其label返回
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    #所以这个函数的整体操作很简单就是求得我们的bbox 于我们的priors 的重合度，并且特别的求出o


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    #这里match 是四点的坐标 而priors 是中心点坐标和w，h 这里要注意一下区别，非常非常重要的一个区别
    #取中间值 对于我们的match 的groundtruth 而言并且求出偏差
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    #(8172 ，4) ----> 通过变化后的(dx,dy,dw,dh)
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    #why we need priors  因为这里说的是和priors 偏差
    #decode 也就是bbox四点的坐标之后返回的是四点的坐标
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    #获取最大的max 值
    x_max = x.data.max()
    #return thing ---> (bs*pirors,1)
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    #这里他的nms 用的是cpu 版本的 效果也还行，但是如果可以调用gpu 版本的那就更好了
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
