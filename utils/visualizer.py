#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-09-07 15:00
# * Last modified : 2018-09-07 15:00
# * Filename      : visualizer.py
# * Description   : in here we combind two files with one to realize the visualize tool 
# **********************************************************
import tensorboardX
import visdom
import os
import commands
from selenium import webdriver
import subprocess
import time
import cv2
import numpy as np
import torch
import torchvision
class Visualizer():
    def __init__(self,save_where = None,env="main"):
        """
        save_where : where to save the tensorboardX files
        env: where to see the 
        """
        if save_where == None:
            self.line_tools = tensorboardX.SummaryWriter() 
        else:
            save_where  = os.path.abspath(save_where)
            print("you choice save :{}".format(save_where))
            self.line_tools=tensorboardX.SummaryWriter(save_where) 
        self.viz = visdom.Visdom(env = env)
        print("when you using our visdom ,you must open web :"+str(self.viz.server)+":"+str(self.viz.port))
        '''
        clear all subwin
        '''
        self.viz.close()
    def image(self,img,title, w=None ,h=None,win =None):
        '''
        img para must be c,h,w,and numpy or torch
        w: the img's width  in win you want to show
        h: the img's height in win you want to show
        title: the win title 
        win : window to show
        '''
        c = img.shape[0]
        if c>3:
            raise Exception("image channel must be <= 3,but you give {}".format(c))
        if w==None or h==None:
            print("draw here ")
            return self.viz.image(img,win = win,opts=dict(title = title))
        else:
            return self.viz.image(img,win=win,opts=dict(title = title,width = w,height=h))
    def images(self,imgs,title,nrow=8,win =None):
        print(imgs.shape)
        if len(imgs.shape)<4:
            raise Exception("the batch is no find ,we must 4 dim ,but you give {} dim".format(len(img.shape)))
        assert imgs.shape[1] == 1 or imgs.shape[1] == 3
        return self.viz.images(imgs,win = win,nrow=nrow,opts=(dict(title= title)))
    def line(self,tag,scalar_value,ite):
        '''
        tag :aix of y
        scalar_value : y val
        it 
        '''
        self.line_tools.add_scalar(tag,scalar_value,ite)
    def lines(self,tag,ite,**kwargs):
        self.line_tools.add_scalars(tag,kwargs,ite)
    def draw_graph(self,model,name_model,*args):
        with tensorboardX.SummaryWriter(comment=name_model) as w:
            w.add_graph(model.cpu(),(torch.randn(args),))
if __name__ =="__main__":
    vis = Visualizer()
    import torchvision
    import torch
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(size=(255,255)),torchvision.transforms.ToTensor()])
    datasets = torchvision.datasets.ImageFolder("./transfer_img/hymenoptera_data/train",transform=transform)
    dataloader = torch.utils.data.DataLoader(datasets,batch_size=8)
    net  = torchvision.models.resnet18()
    vis.draw_graph(net,"res18",1,3,224,224)
    for i in range(100):
        vis.line("train/loss",torch.randn(1),i)
