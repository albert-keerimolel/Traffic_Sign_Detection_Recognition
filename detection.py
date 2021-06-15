# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 06:43:32 2021

@author: Albert
"""

#import glob
#import pandas as pd  
import cv2
import numpy as np
from matplotlib import pyplot as plt  
from skimage.feature import hog   
from sklearn.externals import joblib 
                    
# Loading the mode into same name
pca = joblib.load('pca.pkl')
classifier = joblib.load('svm.pkl')

classnames = {}
classnames["1"] = "road hump"
classnames["14"] = "narrow road ahead"
classnames["17"] = "intersection with priority"
classnames["19"] = "give way"
classnames["21"] = "stop"
classnames["35"] = "turn left"
classnames["38"] = "bicycle crossing"
classnames["45"] = "parking"


def imadjust(x,a,b,c,d,gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def image_fill(Binary_image):
    # Mask used to flood filling.
    im_th=Binary_image.astype('uint8').copy()
    h, w = im_th.shape[:2]
    im_floodfill = im_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # Invert floodfilled image difference through invert filled by black
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

# giving the path of input image
image_path='dataset/input/input image/image.000977.jpg' # Tested image path

img = cv2.imread(image_path)
#converting the image from BGR to RGB
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# applying median blur on image
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

# this is the formula to calculate red channel intensity normalization and stored in cr
Cr = np.maximum(0,np.divide(np.minimum((imgr-imgb),(imgr-imgg)),(imgr+imgg+imgb)))
Cr[np.isnan(Cr)]=0   

# this is the formula to calculate blue channel intensity normalization and stored value is cb
Cb = np.maximum(0,np.divide((imgb-imgr),(imgr+imgg+imgb)))
Cb[np.isnan(Cb)]=0

[rows,cols]=img[:,:,1].shape

# now we are going to normalize the red intensity channel value Cr and pass it to mser red
sc=(cv2.normalize(Cr.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
mser = cv2.MSER_create(_min_area=100,_max_area=10000)

regions, _ = mser.detectRegions(sc.astype('uint8'))
BMred=np.zeros((rows,cols))
if len(regions)>0:
    
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMred[regions[i][j][1],regions[i][j][0]]=1

# now we are going to normalize for blue intensity value Cb and pass it MSER
sb=(cv2.normalize(Cb.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
mser = cv2.MSER_create(_min_area=100,_max_area=10000)

regions, _ = mser.detectRegions(sb.astype('uint8'))
BMblue=np.zeros((rows,cols))
if len(regions)>0:
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMblue[regions[i][j][1],regions[i][j][0]]=1

# first we are going to coonvert the image from BGR to HSV
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)

s[s<0.5]=0
s[s>0.65]=0
s[s>0]=1           #saturation range of red color

v[v<0.2]=0
v[v>0.75]=0
v[v>0]=1           #lightness range of red color

redmask=np.multiply(s,v)   # multiplying values of saturation and lightness of red color and storing in redmask

s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)

s[s<0.45]=0
s[s>0.80]=0
s[s>0]=1          #saturation range of blue color

v[v<0.35]=0
v[v>1]=0
v[v>0]=1           #lightness range of blue color

bluemask=np.multiply(s,v)  # multiplying values of saturation and lightness of blue color and storing in bluemask

BMred_mask=np.multiply(BMred,redmask)
BMblue_mask=np.multiply(BMblue,bluemask)

# calling the image_fill function declared above to fill the difference between mser red and hsv red with black color
BMred_fill=image_fill(BMred_mask)
# calling the image_fill function declared above to fill the difference between mser blue and hsv blue with black color
BMblue_fill=image_fill(BMblue_mask)



# now the below few lines will find if there is any contour detected from red and blue mask calculated and if found then same steps as classification
cont_Saver=cnts_find(BMblue_fill,BMred_fill)
if len(cont_Saver)>0:
    cont_Saver=np.array(cont_Saver)

    cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
    for conta in range(len(cont_Saver)):
        cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]

        #getting the boundry of rectangle around the contours.
        
        image_found=img[y:y+h,x:x+w]
         # here if image found then steps of classification like rgbtogray, medinblur, resize image, threshold, hog, pca and svm
        crop_image=image_found.copy()
        img0=cv2.cvtColor(image_found, cv2.COLOR_RGB2GRAY)
        img0 = cv2.medianBlur(img0,3)

        crop_image0=cv2.resize(img0, (64, 64))

        ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
        descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)


        descriptor_pca=pca.transform(descriptor.reshape(1,-1))

        # class predition of image using SVM
        Predicted_Class=classifier.predict(descriptor_pca)[0]


        if Predicted_Class !=39:
            print ('Predicted Class: ',classnames[str(Predicted_Class)])
            ground_truth_image=cv2.imread('classes_images/'+classnames[str(Predicted_Class)]+'.png')

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)#drawing a green rectange around it.
            #Putting text on the upward of bounding box
            cv2.putText(img, 'Class: '+classnames[str(Predicted_Class)], (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 6)

            #loading the ground truth class respective to the predicted class
            # displaying the ground truth image
            fig = plt.figure()
            plt.imshow(cv2.cvtColor(ground_truth_image,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
            fig.suptitle('Class Ground Truth Image')
            plt.show()
            #ground truth image resize and match according to the sign detected

            try:

                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                img[y:y+ground_truth_image_resized.shape[0], x-w:x-w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized
            except:
                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                img[y:y+ground_truth_image_resized.shape[0], x+w:x+w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized

            plt.figure(figsize=[10,10])
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            cv2.imwrite('Final_Ouput.png',img)


