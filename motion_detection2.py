import cv2
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorKNN()#detecta sombras por defecto


while cap.isOpened():
    count = 0
    ret, frame1 = cap.read()
    #overlay = frame1.copy()
    #alpha = 0.8
    #gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame1)
    #opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)
    blur = cv2.medianBlur(closing,7)#La difuminamos para sacar el sonido de la imagen
    _, thresh = cv2.threshold(blur, 150 , 0, cv2.THRESH_TOZERO)#dividimos los pixeles en 2 grupos eliminando las sombras
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel=(kernel), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour)<400:
            #cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255),2)
            continue
        else:
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0),2)
            cv2.drawContours(frame1, contour, -1, (0,255,0), 2)
            #cv2.fillPoly(overlay, pts =[contour], color=(0,0,255))
            #xd = cv2.addWeighted(frame1, alpha, overlay, 1, 1)
            count += 1

    cv2.putText(frame1,f'Moving objects: {count}', (0,450), cv2.FONT_HERSHEY_SIMPLEX , 1, (148,0,255), 2)


    #cv2.imshow("mask", fgmask)
    #cv2.imshow("opening", opening)
    #cv2.imshow("closing", closing)
    #cv2.imshow("blur", blur)
    #cv2.imshow("thresh", thresh)
    #cv2.imshow("dilated", dilated)
    cv2.imshow("Detection", frame1)

    if cv2.waitKey(40) == 27: #waitkey renderiza la imagen por dichos milisegundos y lee las teclas que son presionadas
        break


cv2.destroyAllWindows()
cap.release()
