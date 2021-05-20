import pyttsx3 
import numpy as np
import cv2
import pandas as pd
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
engine = pyttsx3.init()

def speak_engine(str):

	engine.say(str)
	
	engine.runAndWait()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, classes,class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (23, 230, 210), 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 230, 210), 2)


def detect_object():
	cam=cv2.VideoCapture(0)
	s,image=cam.read(0)
	classes = None	
	#image = cv2.imread('images/example_03.jpg')
	
	Width = image.shape[1]
	Height = image.shape[0]
	scale = 0.00392

	with open('yolov3.txt', 'r') as f:
		classes = [line.strip() for line in f.readlines()]

	COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

	net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

	net.setInput(blob)

	outs = net.forward(get_output_layers(net))

	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4


	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * Width)
				center_y = int(detection[1] * Height)
				w = int(detection[2] * Width)
				h = int(detection[3] * Height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])


	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	
	classname=[]

	for i in indices:
		i = i[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		classname.append(class_ids[i])
		draw_prediction(image,classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

	new_dict = {}
	for i in classname:
		label = str(classes[i])
		new_dict[label]=classname.count(i)
		       
	speak=""
	for names in new_dict:
		speak += str(new_dict[names])+ " " + str(names)+ " "
		
	print(speak)
	
	speak_engine("there is "+speak)
	#cv2.imshow("object detection", image)
	#cv2.waitKey()
		
	cv2.imwrite("object-detection.jpg", image)
	cv2.destroyAllWindows()

while True:
    detect_object()
    time.sleep(5)
    
    
    
