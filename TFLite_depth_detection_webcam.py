import cv2
import threading
import os
import argparse
import cv2
import numpy as np
import sys
import time
import pyglet
import importlib.util
from gpiozero import Button
from gtts import gTTS


import triangulation as tri
import calibration


btn = Button(24)        # GPIO 24 (pin # 18)

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print ("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


B = 4               #Distance between the cameras [cm]
f = 6             #Camera lense's focal length [mm]
alpha = 56.6  

detect = False
detect_right = False
detect_left = False

center_point_right = [0,0]
center_point_left = [0,0]

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(640), int(480)
use_TPU = args.edgetpu

def audio(depth):
    string = "A human is "
    away = " meters away "
    depth = int(depth)
    depth = str(depth)
    depth = string+depth+away
    print(depth)
    language = 'en'
    
    myobj = gTTS(text=depth, lang=language, slow=False)
    
    myobj.save("depth.mp3")
    
    music = pyglet.resource.media("depth.mp3")
    music.play()

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
def depth(frame,previewName):
    global frame_right
    global frame_left
    
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
    if detect == True:
        #..........  CALCULATING DEPTH ........................................
        depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
        if previewName == "Camera 1":
            cv2.putText(frame, "Distance: " + str(round(depth,1)), (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        if previewName == "Camera 2":
            cv2.putText(frame, "Distance: " + str(round(depth,1)), (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
        print("Depth: ", str(round(depth,1)))
    
    if os.path.exists("depth.mp3"):
        os.remove("depth.mp3")
        #start_time = threading.Timer(300000,audio(depth))
    else:
        audio(depth)
        

def objectDetection(frame,previewName):
    global detect
    global detect_right
    global detect_left
    
    
    global center_point_right
    global center_point_left
    
    global GRAPH_NAME
    
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    #................................................................................................
            # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    #cv2.imshow(previewName, frame)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    if previewName == "Camera 1":
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                center_point_right = ((xmin + xmax)/ 2, (ymin + ymax) / 2)
                
                detect = True
                
                depth(frame,previewName)
                
            else:
                cv2.putText(frame, "TRACKING LOST", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    
    if previewName == "Camera 2":
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                center_point_left = ((xmin + xmax)/ 2, (ymin + ymax) / 2)
                
                detect = True
                
                depth(frame,previewName)
                
            else:
                cv2.putText(frame, "TRACKING LOST", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    
    
    #global frame_rate_calc
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector', frame)
    cv2.imshow(previewName, frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    

def camPreview(previewName, camID):
    global frame_right
    global frame_left
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    ret = cam.set(3,imW)
    ret = cam.set(4,imH)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
        
    if previewName == "Camera 1":
        frame_right = frame
    if previewName == "Camera 2":
        frame_left = frame

    while rval:
        rval, frame = cam.read()
        objectDetection(frame,previewName)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):         
            break
    cv2.destroyWindow(previewName)


try:
    if os.path.exists("depth.mp3"):
        os.remove("depth.mp3")
    
    print('Waiting for button...')
    btn.wait_for_press()
    # Create two threads as follows
    thread1 = camThread("Camera 1", 0)
    thread2 = camThread("Camera 2", 1)
    thread1.start()
    thread2.start()
    
except (Exception, KeyboardInterrupt) as e:
    print('\n>> Process Exception ({})'.format(e))


