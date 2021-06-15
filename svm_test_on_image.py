# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 10 07:20:13 2021

@author: Albert
"""

import cv2
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


from skimage.exposure import exposure #for displaying th hog image.
img_path="dataset/Testing/00019/00041_00000.ppm"
print ('Reading Image from Path: ',img_path)
img = cv2.imread(img_path)

crop_image=img
img0=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img0 = cv2.medianBlur(img0,3)

crop_image0=img0
crop_image0=cv2.resize(crop_image0, (64, 64))

# Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
# need to calculate.
ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)

descriptor_pca=pca.transform(descriptor.reshape(1,-1))

# Initilize the 3 axis so that we can plot side by side
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)

#ploting crop image
ax1.axis('off')
ax1.imshow(cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
ax1.set_title('Crop image')

# Rescale histogram for better display,Return image after stretching or shrinking its intensity levels
hog_image_rescaled = exposure.rescale_intensity(imagehog, in_range=(0, 10))
#ploting Hog image
ax2.axis('off')
ax2.imshow(imagehog, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

#ploting Orignal image
ax3.axis('off')
ax3.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
ax3.set_title('Orignal Image')
plt.savefig('hog_crop.png')
plt.show()

# class predition of image using SVM
Predicted_Class=classifier.predict(descriptor_pca)[0]
print ('Predicted Class: ',classnames[str(Predicted_Class)])

ground_truth_image=cv2.imread('classes_images/'+classnames[str(Predicted_Class)]+'.png')

fig = plt.figure()
plt.imshow(cv2.cvtColor(ground_truth_image,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
fig.suptitle('Class Ground Truth Image')
plt.savefig('cropped_image.png')
plt.show()


