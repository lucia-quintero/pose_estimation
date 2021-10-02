import cv2
import time
import numpy as np
import math

MODE = "COCO"

if MODE == "COCO":
    protoFile = "C:/body/models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "C:/body/models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI" :
    protoFile = "C:/body/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "C:/body/models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

pic = 'C:/body/models/alejo.jpg'
frame = cv2.imread(pic)
frame = cv2.resize(frame, dsize=(560, 315))

print(frame)
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
pointsList = []

def distance(pt1,pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def getdist(pointsList):
    pt1, pt2 = pointsList[-2:]
    dInch = (distance(pt1,pt2))/100
    dCm = dInch*2.54
    d=2*dCm/11.98
    cv2.putText(img,str("{0:.2f}".format(d)),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

'''
Implementación YOLO
'''
net2 = cv2.dnn.readNet('C:/body/models/yolov3 (1).weights','C:/body/models/yolov3 (1).cfg')
classes = []
with open('C:/body/models/coco.names','r') as f:
    classes = f.read().splitlines()
#print(classes)
img = frame
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
'''
for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)
'''
net2.setInput(blob)

output_layers_names = net2.getUnconnectedOutLayersNames()
layerOutputs = net2.forward(output_layers_names)


boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2 )

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
            pointsList.append([x,y])
        
print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
print(indexes.flatten())

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color = colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)


cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Continuación OpenPose
'''
t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []
arms =[2, 3, 4, 5, 6, 7]
feet = [8,9,10,11,12,13]
#ángulos
def gradient(pt1,pt2):
    if pt1[1] == pt2[1]:
        y = (pt2[1]-(pt1[1]+1))/(pt2[0]-pt1[0])
    elif pt1[0] == pt2[0]:
        y = (pt2[1]-pt1[1])/(pt2[0]-(pt1[0]+1))
    else :
        y = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    return y

def getAngle(points):
    pt1, pt2, pt3 = points[-3:] #tomo los últimos 3 puntos
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))   #cálculo de grados
    angD = round(math.degrees(angR))  #conversión a grados
    print(angD)
    cv2.putText(frameCopy,str(angD),(pt1[0]-40,pt1[1]-20),cv2.FONT_ITALIC,0.5,(255,0,0),2)
    cv2.line(frameCopy, pt1, pt2, (0, 255, 255), 2)
    cv2.line(frameCopy, pt2, pt3, (0, 255, 255), 2)


for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)
    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    
    if i >= 2 and i <=15:
        if len(points) == 5:
            getAngle(points)
        elif len(points) == 8:
            print(points)
            getAngle(points)
        elif len(points) == 11:
            getAngle(points)
        elif len(points) == 14:
            getAngle(points)
                
    cv2.waitKey(0)
'''
# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0] 
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)


cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)
'''
print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)