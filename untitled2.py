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
#f.truncate() #brise prethodni sadrzaj?? nece nista da upise
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
    
    pt5 =  [0, up_limit]; #PROUCI OVO I PROBAJ DA DODAS I SA STRANE LINIJE
    pt6 =  [w, up_limit];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 =  [0, down_limit];
    pt8 =  [w, down_limit];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))
    
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) #odvaja background od foreground  tj. pokretno od nepokretnog
    #OVAJ BACKGROUND SUBTRACTOR JE BITAN JAKO
    #Structural elements for morphological filters
    kernelOp = np.ones((3,3),np.uint8)
    kernelOp2 = np.ones((5,5),np.uint8)
    kernelCl = np.ones((11,11),np.uint8)
    
    #Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1
    
    while(cap.isOpened()):
    ##for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        #Read an image of the video source
        ret, frame = cap.read()
    ##    frame = image.array
    
        for i in persons:
            i.age_one() #age every person one frame
        #########################
        #   PRE-PROCESAMIENTO   #
        #########################
        
        #Applies background subtraction
        fgmask = fgbg.apply(frame) #primenjuje background subtraction
        fgmask2 = fgbg.apply(frame)
    
        #Binarization to eliminate shadows (gray color)
        try:
            ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
            
            #Opening (erode-> dilate) to remove noise.
            
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
            
            #Closing (dilate -> erode) to join white regions.
            
            mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        except:
            print('EOF')
            print 'PROSLO:',pid
            break
        #################
        #   CONTORNOS   #
        #################
        
        # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
        _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours0:
            area = cv2.contourArea(cnt)
            if area > areaTH:
                #################
                #   TRACKING    #
                #################
                
                #Missing conditions for multipersons, outputs and screen entries.
                
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00']) #centar figure po x
                cy = int(M['m01']/M['m00']) #centar figure po y
                x,y,w,h = cv2.boundingRect(cnt)  # x,y su top left koordinate?
    
                new = True
                
                if cy in range(up_limit,down_limit):
                    for i in persons:
                         if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                             # the object is close to one that has already been detected before
                             new = False
                             i.updateCoords(cx,cy)   #update coordinates in the object and resets prethodni
    # =============================================================================
    #                          if i.going_UP(line_down,line_up) == True:
    #                              cnt_up += 1;
    #                              print "ID:",i.getId(),'crossed going up at',time.strftime("%c")
    #                          elif i.going_DOWN(line_down,line_up) == True:
    #                              cnt_down += 1;
    #                              print "ID:",i.getId(),'crossed going down at',time.strftime("%c")
    #                          break
    # =============================================================================
                         if i.getState() == '1':
                             if i.getDir() == 'down' and i.getY() > down_limit:
                                 i.setDone()
                             elif i.getDir() == 'up' and i.getY() < up_limit:
                                 i.setDone()
                         if i.timedOut():
                             #remove persons from the list
                             index = persons.index(i)
                             persons.pop(index)
                             del i     #free the memory of i
    
                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1     
                #################
                #   DRAWINGS    # crtanje kontura - tacka, pravougaonik i kontura
                #################
                cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
                cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
                
        #END for cnt in contours0
                
        #########################
        # DRAW TRAJECTORIES  #
        #########################
        for i in persons:
            if len(i.getTracks()) >= 2:
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                frame = cv2.polylines(frame,[pts],False,i.getRGB())
            if i.getId() == 9:
                print str(i.getX()), ',', str(i.getY())
            #cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,5,i.getRGB(),1,cv2.LINE_AA) #font umesto 10 bio 0.3
            
        #################
        #   IMAGES   #
        #################
        str_up = 'PROSLO: '+ str(pid-1)
        videoprikaz = 'VIDEO' + str(sledeciVideo)
    #    str_down = 'DOWN: '+ str(cnt_down)
    #    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    #    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
     #   cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
     #cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)
    
    
    
        cv2.imshow('Frame',frame)
     #   cv2.imshow('Mask',mask)  
       
           
        #press ESC to exit
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break
    #END while(cap.isOpened())
        
    f.write(str(nazivVideoSnimka) + "," + str(pid-1) + " \n")
    
    #################
    #   CLEANING    #
    #################
    cap.release()
    cv2.destroyAllWindows()