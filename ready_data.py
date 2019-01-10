import os
import sys
import cv2
import random
import numpy as np
from tensorflow.python.platform import gfile
import glob


data_dir='/home/ispr/data/MIT_SceneParsing/ADEChallengeData2016/'

# create data dict

def get_imgs_path(data_dir):
	file_list = []
	imgs_list={'training':[],'validation':[]}
	directories = ['training', 'validation']
	for directory in directories:
		file_glob = os.path.join(data_dir,"images/"+directory,"*.bmp")
		print("file_glob:\t",file_glob)
		file_list.extend(glob.glob(file_glob))
		if not file_list:
			print("No files found")
		else:
			for f in file_list:
				#print("f:\t",f)
				filename = os.path.splitext(f.split("/")[-1])[0]
				#print("img_name:\t",img_name)
				anno_path = os.path.join(data_dir,"annotations/",directory,filename+".png")
				#print("anno_name:\t",anno_path)
				if os.path.exists(anno_path):
					record = {"image":f,"annotation":anno_path,"filename":filename}
					imgs_list[directory].append(record)
				else:
					print("%s Annotation file not found for %s - Skipping"%(directory ,filename))
		random.shuffle(imgs_list[directory])
		no_of_imgs = len(imgs_list[directory])
		print("No. of %s files :%d "%(directory,no_of_imgs))
		
		file_list.clear()
	train_records = imgs_list["training"]
	val_records = imgs_list["validation"]
	
	return train_records,val_records

if __name__ == '__main__':
	train_records,val_records=get_imgs_path(data_dir)
	print(val_records)