# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 01:11:48 2021

@author: Albert
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from skimage.feature import hog   
from sklearn.externals import joblib 

pca = joblib.load('pca.pkl')
classifier = joblib.load('svm.pkl')


#load the trained model to classify sign
classnames = {}
classnames["1"] = "road hump"
classnames["14"] = "narrow road ahead"
classnames["17"] = "intersection with priority"
classnames["19"] = "give way"
classnames["21"] = "stop"
classnames["35"] = "turn left"
classnames["38"] = "bicycle crossing"
classnames["45"] = "parking"

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)



def classify(image_path):

    def imadjust(x,a,b,c,d,gamma=1):
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y
 
    def image_fill(Binary_image):
        im_th=Binary_image.astype('uint8').copy()
        h, w = im_th.shape[:2]
        im_floodfill = im_th.copy()
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 1);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        im_out[im_out==254]=0
        return im_out
    
    def cnts_find(binary_image_blue,binary_image_red):
        cont_Saver=[]
    
        (cnts, _) = cv2.findContours(binary_image_blue.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
        for d in cnts:
            if cv2.contourArea(d)>700:
                    (x, y, w, h) = cv2.boundingRect(d)
                    if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                        cont_Saver.append([cv2.contourArea(d),x, y, w, h])
    
        (cnts, _) = cv2.findContours(binary_image_red.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
        for d in cnts:
            if cv2.contourArea(d)>700:
                    (x, y, w, h) = cv2.boundingRect(d)
                    if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                        cont_Saver.append([cv2.contourArea(d),x, y, w, h])
        return cont_Saver



    
    img = cv2.imread(image_path)
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb[:,:,0] = cv2.medianBlur(img_rgb[:,:,0],3)
    img_rgb[:,:,1] = cv2.medianBlur(img_rgb[:,:,1],3)
    img_rgb[:,:,2] = cv2.medianBlur(img_rgb[:,:,2],3)

    arr2=img_rgb.copy()
    arr2 = cv2.normalize(arr2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    imgr=arr2[:,:,0]
    imgg=arr2[:,:,1]
    imgb=arr2[:,:,2]

    imgr=imadjust(imgr,imgr.min(),imgr.max(),0,1)
    imgg=imadjust(imgg,imgg.min(),imgg.max(),0,1)
    imgb=imadjust(imgb,imgb.min(),imgb.max(),0,1)

    Cr = np.maximum(0,np.divide(np.minimum((imgr-imgb),(imgr-imgg)),(imgr+imgg+imgb)))
    Cr[np.isnan(Cr)]=0

    Cb = np.maximum(0,np.divide((imgb-imgr),(imgr+imgg+imgb)))
    Cb[np.isnan(Cb)]=0

    [rows,cols]=img[:,:,1].shape

    sc=(cv2.normalize(Cr.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
    mser = cv2.MSER_create(_min_area=100,_max_area=10000)

    regions, _ = mser.detectRegions(sc.astype('uint8'))
    BMred=np.zeros((rows,cols))
    if len(regions)>0:
    
        for i in range(len(regions)):
            for j in range(len(regions[i])):
                BMred[regions[i][j][1],regions[i][j][0]]=1
        
    sb=(cv2.normalize(Cb.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
    mser = cv2.MSER_create(_min_area=100,_max_area=10000)

    regions, _ = mser.detectRegions(sb.astype('uint8'))
    BMblue=np.zeros((rows,cols))
    if len(regions)>0:
        for i in range(len(regions)):
            for j in range(len(regions[i])):
                BMblue[regions[i][j][1],regions[i][j][0]]=1

    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)

    s[s<0.5]=0
    s[s>0.65]=0
    s[s>0]=1

    v[v<0.2]=0
    v[v>0.75]=0
    v[v>0]=1

    redmask=np.multiply(s,v)

    s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)

    s[s<0.45]=0
    s[s>0.80]=0
    s[s>0]=1

    v[v<0.35]=0
    v[v>1]=0
    v[v>0]=1

    bluemask=np.multiply(s,v)

    BMred_mask=np.multiply(BMred,redmask)
    BMblue_mask=np.multiply(BMblue,bluemask)


    BMred_fill=image_fill(BMred_mask)
    BMblue_fill=image_fill(BMblue_mask)

    cont_Saver=cnts_find(BMblue_fill,BMred_fill)
    if len(cont_Saver)>0:
        cont_Saver=np.array(cont_Saver)

        cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
        for conta in range(len(cont_Saver)):
            cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]

        #getting the boundry of rectangle around the contours.

            image_found=img[y:y+h,x:x+w]

            crop_image=image_found.copy()
            img0=cv2.cvtColor(image_found, cv2.COLOR_RGB2GRAY)
            img0 = cv2.medianBlur(img0,3)

            crop_image0=cv2.resize(img0, (64, 64))

            # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
            # need to calculate.
            ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
            descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)


            # descriptor,imagehog = hog(crop_image0, orientations=8, visualize=True)
            descriptor_pca=pca.transform(descriptor.reshape(1,-1))

            # class predition of image using SVM
            Predicted_Class=classifier.predict(descriptor_pca)[0]


            if Predicted_Class !=39:
                print ('Predicted Class: ',classnames[str(Predicted_Class)])
                ground_truth_image=cv2.imread('classes_images/'+classnames[str(Predicted_Class)]+'.png')

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)#drawing a green rectange around it.
                #Putting text on the upward of bounding box
                cv2.putText(img, 'Class: '+classnames[str(Predicted_Class)], (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 6)

                try:

                    ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                    img[y:y+ground_truth_image_resized.shape[0], x-w:x-w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized
                except:
                        ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                        img[y:y+ground_truth_image_resized.shape[0], x+w:x+w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized

                plt.figure(figsize=[10,10])
                plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                cv2.imwrite('Final_Ouput.png',img)
              
                sign = classnames[str(Predicted_Class)]
                label.configure(foreground='#011638', text=sign) 


def show_classify_button(image_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(image_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)


def upload_image():
    try:
        image_path=filedialog.askopenfilename()
        uploaded=Image.open(image_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(image_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
