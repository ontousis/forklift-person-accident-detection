import cv2
import time
import torch
import numpy as np
import os
from ultralytics import YOLO
import yaml
import lgpio
import requests

def iou_ltrb(ltrb1,ltrb2):
	xi_lt=max(ltrb1[0],ltrb2[0])
	yi_lt=max(ltrb1[1],ltrb2[1])
	xi_rb=min(ltrb1[2],ltrb2[2])
	yi_rb=min(ltrb1[3],ltrb2[3])
	if xi_lt>xi_rb or yi_lt>yi_rb:
		return 0
	i=(xi_rb-xi_lt)*(yi_rb-yi_lt)
	u=(ltrb1[2]-ltrb1[0])*(ltrb1[3]-ltrb1[1])+(ltrb2[2]-ltrb2[0])*(ltrb2[3]-ltrb2[1])-i
	return i/u

def ioh_ltrb(ltrb1,ltrb2): #intersection over human - ltrb2 corresponds to a person
	xi_lt=max(ltrb1[0],ltrb2[0])
	yi_lt=max(ltrb1[1],ltrb2[1])
	xi_rb=min(ltrb1[2],ltrb2[2])
	yi_rb=min(ltrb1[3],ltrb2[3])
	if xi_lt>xi_rb or yi_lt>yi_rb:
		return 0
	i=(xi_rb-xi_lt)*(yi_rb-yi_lt)
	h=(ltrb2[2]-ltrb2[0])*(ltrb2[3]-ltrb2[1])
	return i/h

def draw_ltwh_box(box,img,color=(255,0,0)):
	cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[0])+int(box[2]),int(box[1])+int(box[3])), color, 2)
	return img

def text_on_img(img,text,coords,font=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,255,0),thickness=4):
	cv2.putText(img,text,coords,font,fontScale,color=color,thickness=thickness)
	return img

def estimate_danger(ltrb_forklift,ltrb_person,idx,idx1,positions,p1=0.5,p2=0.4,p3=0.3,p4=1.5,p5=0.2,p6=0.15):
# positions: list of sequences of positions (x,y,timestamp(ms)) for all objects ( [ [(x1,y1,timestamp(ms)),(x2,y2,timestamp(ms)),..],..,[..] ]
# idx,idx1: indices for forklift and person in "positions"
	vert_d_forklift_front_perc=p1
	vert_d_human_front_perc=p2
	cond1_iou_thr=p3
	cond2_hor_dist_perc=p4
	speed_tot_height_perc=p5
	speed_fork_height_perc=p6
	dt=(positions[idx][-1][2]-positions[idx][0][2])/1000 #time difference in seconds
	forklift_height=ltrb_forklift[3]-ltrb_forklift[1]
	forklift_width=ltrb_forklift[2]-ltrb_forklift[0]
	x0_forklift=positions[idx][0][0]
	y0_forklift=positions[idx][0][1]
	x1_forklift=positions[idx][-1][0]
	y1_forklift=positions[idx][-1][1]
	x0_person=positions[idx1][0][0]
	y0_person=positions[idx1][0][1]
	x1_person=positions[idx1][-1][0]
	y1_person=positions[idx1][-1][1]
	distance_init=np.sqrt((x0_forklift-x0_person)**2+(y0_forklift-y0_person)**2)
	distance_fin=np.sqrt((x1_forklift-x1_person)**2+(y1_forklift-y1_person)**2)
	distance_forklift=np.sqrt((x1_forklift-x0_forklift)**2+(y1_forklift-y0_forklift)**2)
	if (y1_forklift-y1_person>0 and y1_forklift-y1_person<=vert_d_forklift_front_perc*forklift_height) or (y1_person-y1_forklift>=0 and y1_person-y1_forklift<=vert_d_human_front_perc*forklift_height):
		ioh=ioh_ltrb(ltrb_forklift,ltrb_person)
		if ioh>=cond1_iou_thr and (distance_fin-distance_init)/dt<-speed_tot_height_perc*forklift_height and distance_forklift/dt>speed_fork_height_perc*forklift_height:
			return 1
		if abs(x1_person-x1_forklift)<=cond2_hor_dist_perc*forklift_width and (distance_fin-distance_init)/dt<-speed_tot_height_perc*forklift_height and distance_forklift/dt>speed_fork_height_perc*forklift_height:
			return 1
	return 0

def indicate_danger():
	lgpio.gpio_write(h, 17, 1)
	time.sleep(2)
	lgpio.gpio_write(h, 17, 0)

def get_yolov8_results(model,img,thr=0.05):
	r=model(img,conf=thr,imgsz=(320,416))
	boxes_xywh=r[0].boxes.xywh
	boxes_ltwh=[]
	for b in boxes_xywh:
		boxes_ltwh.append([b[0]-b[2]/2,b[1]-b[3]/2,b[2],b[3]])
	conf=r[0].boxes.conf
	cls=r[0].boxes.cls
	res=[]
	for i in range(len(cls)):
		res.append( ([int(x) for x in boxes_ltwh[i]],conf[i].item(),int(cls[i])) )
	return res	# res=[ ([l,t,w,h],conf,cls) , .. ]

def downscale_bbox_list(bboxes,scale_down):
	ret=[]
	for i in range(len(bboxes)):
		ret.append([])
		for j in range(4):
			ret[-1].append(bboxes[i][j]/scale_down)
	return ret

def upscale_bbox_list(bboxes,scale_up):
	ret=[]
	for i in range(len(bboxes)):
		ret.append([])
		for j in range(4):
			ret[-1].append(bboxes[i][j]*scale_up)
	return ret

with open("config/config.yaml") as cf:
    conf=yaml.safe_load(cf.read())

model_path=conf["model_path"]
cam_path=conf["cam_path"]
det_interval=conf["det_interval"]
conf_threshold=conf["conf_threshold"]
p1=conf["vert_d_forklift_front_perc"]
p2=conf["vert_d_human_front_perc"]
p3=conf["cond1_ioh_thr"]
p4=conf["cond2_hor_dist_perc"]
p5=conf["speed_tot_height_perc"]
p6=conf["speed_fork_height_perc"]
tracking_rescale=conf["tracking_rescale"]
backend_url=conf["backend_url"]

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, 17)

camera = cv2.VideoCapture(cam_path)

model_yolo=YOLO(model_path)

time_last_detection=0

while 1:
	ret,img_r=camera.read()
	if ret==0:
		continue
	print("----------------- New Frame------------------ Shape:",img_r.shape)
	img_r_downscaled=cv2.resize(img_r,(int(img_r.shape[1]/tracking_rescale),int(img_r.shape[0]/tracking_rescale)))
	new_detection=0
	if time.time_ns()//1000000-time_last_detection>=det_interval:	# New detections every second
		positions=[]						                        # list of sequences of positions (x,y,timestamp) for all objects ( [ [(x1,y1,timestamp(ms)),(x2,y2,timestamp),..],.. ] )
		results=get_yolov8_results(model_yolo,img_r,thr=conf_threshold)
		time_last_detection=time.time_ns()//1000000
		new_detection=1						                        # 1 for new detection, 0 if only tracking will be performed for the frame
	if new_detection:
		multi_tracker=cv2.legacy.MultiTracker_create()
	else:
		success,tracked_boxes=multi_tracker.update(img_r_downscaled)
		tracked_boxes=upscale_bbox_list(tracked_boxes,tracking_rescale)
		if success==0:
			positions=[]
			ret,img_r=camera.read()
			if ret==0:
				continue
			img_r_downscaled=cv2.resize(img_r,(int(img_r.shape[1]/tracking_rescale),int(img_r.shape[0]/tracking_rescale)))
			results=get_yolov8_results(model_yolo,img_r,thr=conf_threshold)
			time_last_detection=time.time_ns()//1000000
			new_detection=1
			multi_tracker=cv2.legacy.MultiTracker_create()
	idx=0							        # index for correspondence of detections with tracked boxes
	if new_detection==0 and success==1:		# if tracking, update positions
		for r in results:
			bbox=tracked_boxes[idx]
			positions[idx].append( (int(int(bbox[0])+0.5*int(bbox[2])),int(int(bbox[1])+0.5*int(bbox[3])),time.time_ns()//1000000) )
			idx+=1
	idx=0
	for r in results:
		category=r[-1]				# Category: 0->forklift  1->person
		if new_detection:
			bbox=r[0]				# ltwh bounding box
			tracker=cv2.legacy.TrackerKCF_create()
			bbox_d=downscale_bbox_list([bbox],tracking_rescale)[0]
			multi_tracker.add(tracker,img_r_downscaled,bbox_d) 	# Add tracker for each object
			#draw_ltwh_box(bbox,img_r)
			positions.append([(int(int(bbox[0])+0.5*int(bbox[2])),int(int(bbox[1])+0.5*int(bbox[3])),time.time_ns()//1000000)])
		else: # Only tracking
			if success:
				bbox=tracked_boxes[idx]
				if category==0:
					ltrb_forklift=[int(bbox[0]),int(bbox[1]),int(bbox[0])+int(bbox[2]),int(bbox[1])+int(bbox[3])]
					idx1=0
					for r1 in results:
						bbox1=tracked_boxes[idx1]
						category1=r1[-1]
						if category1==1:
							ltrb_person=[int(bbox1[0]),int(bbox1[1]),int(bbox1[0])+int(bbox1[2]),int(bbox1[1])+int(bbox1[3])]
							d=estimate_danger(ltrb_forklift,ltrb_person,idx,idx1,positions,p1,p2,p3,p4,p5,p6)
							if d:
								draw_ltwh_box(bbox,img_r,color=(0,0,255))
								indicate_danger()
								t=time.time_ns()//1000000
								cv2.imwrite("potential_danger"+str(t)+".jpg",img_r)
        							requests.post(backend_url,files={'image':open("potential_danger"+str(t)+".jpg","rb")},data={'time':time.strftime("%d_%m_%Y_%H_%M_%S")})
        							os.remove("potential_danger"+str(t)+".jpg")
						idx1+=1
		idx+=1
