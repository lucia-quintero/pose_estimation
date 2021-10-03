import argparse
import cv2
import math

from config import Models


class PoseEstimation:
    def __init__(self, mode):
        self.mode = mode
        self.models = Models()
        self.load_model()

        self.threshold = 0.1
        self.angsD = []

    def load_model(self):
        if self.mode == "COCO":
            self.protoFile = self.models.protoFile_coco
            self.weightsFile = self.models.weightsFile_coco
            self.nPoints = self.models.nPoints_coco
            self.pose_pairs = self.models.pose_pairs_coco
        elif self.mode == "MPI":
            self.protoFile = self.models.protoFile_mpi
            self.weightsFile = self.models.weightsFile_mpi
            self.nPoints = self.models.nPoints_mpi
            self.pose_pairs = self.models.pose_pairs_mpi

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    def load_img(self, img_path):
        self.name_file = img_path.split('/')[-1]
        self.frame = cv2.imread(img_path)
        self.frame = cv2.resize(self.frame, dsize=(560, 315))
        self.frameWidth = self.frame.shape[1]
        self.frameHeight = self.frame.shape[0]

    def distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def getdist(self, pointsList):
        pt1, pt2 = pointsList[-2:]
        dInch = (self.distance(pt1, pt2))/100
        dCm = dInch*2.54
        d = 2*dCm/11.98
        cv2.putText(self.frame, str("{0:.2f}".format(
            d)), (pt1[0]-40, pt1[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def gradient(self, pt1, pt2):
        if pt1[1] == pt2[1]:
            y = (pt2[1]-(pt1[1]+1))/(pt2[0]-pt1[0])
        elif pt1[0] == pt2[0]:
            y = (pt2[1]-pt1[1])/(pt2[0]-(pt1[0]+1))
        else:
            y = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        return y
    
    def getAngle(self, points):
        pt1, pt2, pt3 = points[-3:]  # tomo los últimos 3 puntos
        m1 = self.gradient(pt1, pt2)
        m2 = self.gradient(pt1, pt3)
        angR = math.atan((m2-m1)/(1+(m2*m1)))  # cálculo de grados
        angD = round(math.degrees(angR))  # conversión a grados
        self.angsD.append(angD)
        cv2.putText(self.frame, str(angD),
                    (pt1[0]-40, pt1[1]-20), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 2)
        cv2.line(self.frame, pt1, pt2, (0, 255, 255), 2)
        cv2.line(self.frame, pt2, pt3, (0, 255, 255), 2)
            
    def getBlobresult(self):
        inpBlob = cv2.dnn.blobFromImage(self.frame, 1.0 / 255, (368, 368),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        return self.net.forward()
    
    def mainloop(self, verbose):
        
        output = self.getBlobresult()
        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []
        # ángulos
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (self.frameWidth * point[0]) / W
            y = (self.frameHeight * point[1]) / H

            if prob > self.threshold:
                cv2.circle(self.frame, (int(x), int(y)), 4, (0, 255, 255),
                        thickness=-1, lineType=cv2.FILLED)
                cv2.putText(self.frame, "{}".format(i), (int(x), int(
                    y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)
   
            try: 
                if i >= 2 and i <= 15:
                    if len(points) == 5:
                        self.getAngle(points)
                    elif len(points) == 8:
                        self.getAngle(points)
                    elif len(points) == 11:
                        self.getAngle(points)
                    elif len(points) == 14:
                        self.getAngle(points)
            except Exception as d:
                print("Angle not found")
        if verbose:
            cv2.imshow('imagen salida',self.frame)
            cv2.waitKey(0)
        cv2.imwrite('resultado/resultado-{}'.format(self.name_file),self.frame)
        print("Angles: {}".format(self.angsD))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=str, default="COCO",
                    help="Ingrese el modo")
    ap.add_argument("-i", "--img", type=str, default="test/tenis.jpg",
                    help="Ingrese la imagen de prueba")
    ap.add_argument("-s", "--verbose", type=int, default=0,
                    help="True/False mostrar datos")

    args = vars(ap.parse_args())
    mode = args["mode"]
    img_path = args["img"]
    verbose = args["verbose"]

    pose_obj = PoseEstimation(mode)
    pose_obj.load_img(img_path)
    pose_obj.mainloop(verbose)



