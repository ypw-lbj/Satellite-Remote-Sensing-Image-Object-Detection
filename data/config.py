# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (74, 85, 84)

# SSD300 CONFIGS
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

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
#this part input 320*320 image ,we called it's refineDet SSD
VOC_320 = {
    'num_classes': 16,

    'feature_maps': [40, 20, 10, 5],

    'lr_epochs': [0.5, 0.75, 0.85],

    'min_dim': 320,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name':"VOC320"
}
VOC_512 = {
    'num_classes': 16,

    'feature_maps': [64, 32, 16, 8],

    'lr_epochs': [0.25,0.5, 0.75],
    # 200 ,300
    'min_dim': 512,

    'steps': [8, 16, 32, 64],
     #51.2, 102.4, 281.6, 460.8
    'min_sizes': [16, 30, 60, 100],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name':"VOC512"
}
VOC_1536 = {
    'num_classes': 16,

    'feature_maps': [192, 96, 48, 24],

    'lr_epochs': [0.5, 0.75, 0.85],

    'min_dim': 512,

    'steps': [8, 16, 32, 64],

    'min_sizes': [15, 32,73,200],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name':"VOC320"
}

