#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        #300
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_priors = len(cfg['aspect_ratios'])
        #'variance': [0.1, 0.2],
        self.variance = cfg['variance'] or [0.1]
        #'feature_maps': [38, 19, 10, 5, 3, 1],
        self.feature_maps = cfg['feature_maps']
        #[30, 60, 111, 162, 213, 264]
        self.min_sizes = cfg['min_sizes']
        #'max_sizes': [60, 111, 162, 213, 264, 315]
        self.max_sizes = cfg['max_sizes']
        #'steps': [8, 16, 32, 64, 100, 300],
        self.steps = cfg['steps']
        # 相对变化率在这里是个很厉害的杀招 在这里2 代表的缩放尺度是0.5 和2 . 3 意味着 1/3 和3/1
	# 相所以如下的结构就有了非常好的一个意义 对应的尺度为 4 6 6 6 4 4
        # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = cfg['aspect_ratios']
        #True
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        #return size for here is (8172,4)----->cx,cy--->all is corresponding point in the feature map , and the bbox size in ratio
        mean = []
        #feature_map----->[38,19,10,5,3,1k]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                #here we can easily get the decare prodcut (0,37),(0,37)
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                # 
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
		

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes:
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
