#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-09-26 14:09
# * Last modified : 2018-09-26 14:09
# * Filename      : data_fire.py
# * Description   : 
# **********************************************************
import fire
import os
import glob
import tqdm
import shutil
def build_data(**kwargs):
    root =kwargs['root']
    train_dir = kwargs['dir']
    Detections = kwargs['det']
    img_set= 'JPEGImages'
    an_set ='Annotations'
    try:
    	train_dir = os.path.join(root,train_dir) 
    	Detections = os.path.join(train_dir,Detections)
    	img_set = os.path.join(Detections,img_set)
    	an_set = os.path.join(Detections,an_set)
    	os.mkdir(train_dir)
    	os.mkdir(Detections)
    	os.mkdir(img_set)
    	os.mkdir(an_set)
    except:
	print "file exist!"
def move_imgs(**kwargs):
	'''
	:param kwargs: move the img_set to our Detection/JPEGImages
	:return:
	'''
	target_root = kwargs['target_root']
	file_names = glob.glob(target_root+'/*')
	for i in tqdm.trange(len(file_names)):
		file_name = file_names[i]
		target_name = os.path.join("./data/train_dir/Detections/JPEGImages",file_name.split("/")[-1])
		shutil.copyfile(file_name,target_name)
def build_idx(**kwargs):
	'''

	:param kwargs: get the idx of imgs
	:return:
	'''
	root = './data/train_dir/Detections/JPEGImages'
	file_names = glob.glob(root+"/*")
	with open("./data/train_dir/Detections/img_idx.txt",'w') as writer:
		for idx in tqdm.trange(len(file_names)):
			file_name = file_names[idx]
			writer.write(file_name.split('/')[-1].split('.')[0]+'\n')



if __name__ =="__main__":
    fire.Fire()

