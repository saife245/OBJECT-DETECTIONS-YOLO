# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:02:30 2019

Topic:- OBJECT DETECTION USING SSD MODEL

@author: MD SAIF UDDIN
"""

#importng the library
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#detection function to detect
def detect(frame, net, transform):
    height, width = frame.shape[:2]#taking only hight and width from the frame
    frame_t = transform(frame)[0]   #transform the frame to give numpy aarray
    x = torch.from_numpy(frame_t).permute(2,0,1)#transform the numpy array to torch tensor
    #permute convert the rgb chnnel to grb so nueral network can take
    #convert the x to batch.so, NN can takes in  batch 
    x = Variable(x.unsqueeze(0))
    #feed x to neural network
    y = net(x)
    #creating the another tensor contain the output of NN
    '''detections =  [batch, number of classes(object like dog,cat), number of occurance of object,
     (score, x0,y0,x1,y1)] score is the thresold to predict the presence of object'''
    detections = y.data
    #creating the tensor of (width, height, width, height)of 4 dimension
    #first width, height is for scale value of upper left corner
    #second width, height is for scale value of lower right corner
    scale = torch.Tensor([width, height, width, height])
    #making the for loop to select from thresold
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:#mapping score must be greater than equal to 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame,(int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)#we draw the rectangle around the object of video per frame
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)#after detection writing the label
            j = j + 1
    return frame

#creating the ssd Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

#creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#doing the Object detection
reader = imageio.get_reader("P1033657.mp4")#open the video for detection
fps = reader.get_meta_data()['fps']# we get the video @ fps 
writer = imageio.get_writer('output2.mp4', fps = fps)#we write the frame in new output file
for i, frame in enumerate(reader):#we iterate the loop for whole video frame
    frame = detect(frame, net.eval(), transform)#detecting the frame
    writer.append_data(frame)#append or, write back the detected object of frame 
    print(i)
writer.close()

