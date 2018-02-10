# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:23:44 2018

@author: Jovana
"""

import numpy as np
import cv2
import Person
import time


f = open("out.txt", "w")
f.write("RA156/2014, Jovana Brasanac" + "\n") 
f.write("file,count" + "\n")

for sledeciVideo in range (1,11):
    
    videoSnimak = "video" + format(sledeciVideo) + ".mp4"
    nazivVideoSnimka = "video" + format(sledeciVideo) + ".mp4"

    cap = cv2.VideoCapture(videoSnimak) # Otvaram video
    
    w = cap.get(3) #sirina videa
    h = cap.get(4) #visina videa
    frameArea = h*w #povrsina frejma
    areaTH = frameArea/1100 #varijabla koju koristimo za detekciju ljudi
    print 'Area Threshold', areaTH
    
    
    up_limit =   int(1*(h/5))
    down_limit = int(4.7*(h/5))
    
    pt5 =  [0, up_limit]; 
    pt6 =  [w, up_limit];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 =  [0, down_limit];
    pt8 =  [w, down_limit];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))
    
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) #odvaja background od foreground  tj. pokretno od nepokretnog
    kernelOp = np.ones((3,3),np.uint8)
    kernelOp2 = np.ones((5,5),np.uint8)
    kernelCl = np.ones((11,11),np.uint8)
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1
    
    while(cap.isOpened()):

        ret, frame = cap.read()  #frame = image.array
    
        for i in persons:
            i.age_one() 
       
        fgmask = fgbg.apply(frame) #primenjuje background subtraction
        fgmask2 = fgbg.apply(frame)
    
        #Binarizacija za eliminisanje senki (gray color)
        try:
            ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
            
            #Otvaranje (erode-> dilate) za uklanjanje suma.
            
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
            
            #Zatvaranje (dilate -> erode) za spajanje belih regiona.
            
            mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        except:
            print 'PROSLO:',pid
            break
       
        
        _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours0:
            area = cv2.contourArea(cnt)
            if area > areaTH:
                
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00']) #centar figure po x
                cy = int(M['m01']/M['m00']) #centar figure po y
                x,y,w,h = cv2.boundingRect(cnt) 
    
                new = True
                
                if cy in range(up_limit,down_limit):
                    for i in persons:
                         if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                             # objekat je blizu onoga sto je vec detektovan
                             new = False
                             i.updateCoords(cx,cy)   #update koordinate i resetuje age
   
                         if i.getState() == '1':
                             if i.getDir() == 'down' and i.getY() > down_limit:
                                 i.setDone()
                             elif i.getDir() == 'up' and i.getY() < up_limit:
                                 i.setDone()
                         if i.timedOut():
                             #obrisi osobu iz liste
                             index = persons.index(i)
                             persons.pop(index)
                             del i     #oslobodi memoriju za i
    
                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1     
  
                # crtanje kontura - tacka, pravougaonik i kontura

                cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
                cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
                
        #END za cnt u contours0
               

        str_up = 'PROSLO: '+ str(pid-1)
        videoprikaz = 'VIDEO' + str(sledeciVideo)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
     
    
        cv2.imshow('Frame',frame)
     #   cv2.imshow('Mask',mask)  #prikazivanje maske
       
        # ESC za exit
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break
    #END while(cap.isOpened())
        
    f.write(str(nazivVideoSnimka) + "," + str(pid-1) + " \n")
    
    cap.release()
    cv2.destroyAllWindows() #brisanje