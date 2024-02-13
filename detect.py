import os
import cv2

# Load the pre-trained model and class labels
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
label_file = 'Labels.txt'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLables = [] #empty list of python
file_name= "Lables.txt"
with open(file_name,'rt') as fpt:
    classLables = fpt.read().rstrip('\n').split('\n')
    #classLables.append(fpt.read())

# Set input parameters for the model
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

def detect_objects(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image file.")
        return

    # Perform object detection
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    # Get the names of the detected objects
    detected_objects = []
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        class_name = classLables[ClassInd - 1]
        detected_objects.append(class_name)

    return detected_objects

if __name__ == "__main__":
    # Get input image path from user
    image_path = input("Enter the path to the image file: ")

    # Perform object detection
    detected_objects = detect_objects(image_path)

    # Print detected objects
    print("Detected objects:")
    for obj in detected_objects:
        print(obj)
