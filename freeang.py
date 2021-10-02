import cv2
import math
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


path = 'C:\body\models\int3.jpg'
img = cv2.imread(path) #leer imagen
img = cv2.resize(img, dsize=(600, 1000))
pointsList = []
pointsList2 = []
angu = []
dist = []

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(img,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,0),2)  #
        cv2.circle(img,(x,y),3,(0,0,0),cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        size = len(pointsList2)
        if size != 0 and size % 2 != 0:
            cv2.line(img,tuple(pointsList2[round((size-1)/2)*2]),(x,y),(0,0,0),2)  #
        cv2.circle(img,(x,y),3,(0,0,0),cv2.FILLED)
        pointsList2.append([x,y])
        print(pointsList2)     

def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:] #tomo los últimos 3 puntos
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))   #cálculo de grados
    angD = round(math.degrees(angR))  #conversión a grados
    angD = abs(angD)
    angu.append(angD)
    #print(angD)
    cv2.putText(img,str(angD),(pt1[0]-40,pt1[1]-20),cv2.FONT_ITALIC,0.5,(0,0,255),2)

def distance(pt1,pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def getdist(pointsList2):
    pt1, pt2 = pointsList2[-2:]
    dInch = (distance(pt1,pt2))/100
    dCm = dInch*2.54
    d=2*dCm/11.98
    dist.append("{0:.2f}".format(d))
    cv2.putText(img,str("{0:.2f}".format(d)),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
   
while True:

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)
        pointsList=[]
    
    if len(pointsList2) % 2 == 0 and len(pointsList2) != 0:
        getdist(pointsList2)
        pointsList2=[]

    cv2.imshow('Imagen',img) #mostrar
    cv2.setMouseCallback('Imagen',mousePoints) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv2.imread(path)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        cv2.imwrite('Imagen.jpg',img)
        break


cuello = angu[0]
vision = angu[1]
codo = angu[2]
espalda = angu[3]
cadera = angu[4]
#rodilla = angu[5]
#tobillo = angu[6]

ojos = dist[0]


now = datetime.now()
now = now.strftime('%d/%m/%Y')

c = canvas.Canvas("Informe.pdf", pagesize=letter)
c.setLineWidth(.3)
c.setFont('Helvetica',12)
c.drawString(30,750,'INFORME DE PRUEBA')
c.drawString(30,735,'ANÁLISIS DE LA ERGONOMÍA EN EL PUESTO DE TRABAJO')
c.drawString(500,750,now)
c.line(480,747,580,747)

c.drawString(275,720,'NOMBRE: ')
c.drawString(500,725,"UAO-Catalina Ruiz")
c.line(378,723,580,723)
c.drawString(30,700,"Zona")
c.drawString(275,700,"Rango")
c.drawString(500,700,"Resultado")

c.drawString(30,680,"Ángulo del cuello")
c.drawString(275,680,"[0° - 20°]")
c.drawString(500,680,str(cuello)+"°")

c.drawString(30,660,"Ángulo de visión")
c.drawString(275,660,"[20°-60°]")
c.drawString(500,660,str(vision)+"°")

c.drawString(30,640,"Ángulo del codo")
c.drawString(275,640,"[90° alineados a la superficie]")
c.drawString(500,640,str(codo)+"°")

c.drawString(30,620,"Posición de espalda")
c.drawString(275,620,"[0°-20°]")
c.drawString(500,620,str(espalda)+"°")

c.drawString(30,600,"Ángulo de cadera")
c.drawString(275,600,"[90°-110°]")
c.drawString(500,600,str(cadera)+"°")
'''
c.drawString(30,580,"Ángulo de rodilla")
c.drawString(275,580,"[90°]")
c.drawString(500,580,str(rodilla)+"°")

c.drawString(30,560,"Ángulo de tobillo")
c.drawString(275,560,"[90°]")
c.drawString(500,560,str(tobillo)+"°")
'''
c.drawString(30,580,"Distancia a pantalla")
c.drawString(275,580,"[0.46m-0.70m]")
c.drawString(500,580,str(ojos)+"m")

c.drawImage('Imagen.jpg', 180, 50, 270, 480)
c.showPage() #salto de página
c.save()