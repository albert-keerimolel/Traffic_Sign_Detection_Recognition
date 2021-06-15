# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 10 04:33:17 2021

@author: Albert
"""

import glob           
import pandas as pd  
import cv2            
import numpy as np    
from matplotlib import pyplot as plt 
from skimage.feature import hog 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
from sklearn.externals import joblib 

classnames = {}
classnames["1"] = "road hump"
classnames["14"] = "narrow road ahead"
classnames["17"] = "intersection with priority"
classnames["19"] = "give way"
classnames["21"] = "stop"
classnames["35"] = "turn left"
classnames["38"] = "bicycle crossing"
classnames["45"] = "parking"


#combining csv files of each class to make it easier for svm  
Training_Images_Directory='dataset/Training' #training dataset directory.

csv_files_training=glob.glob(Training_Images_Directory+'/**/*.csv',recursive=True)  
main_Training=pd.read_csv(csv_files_training[0],sep=';') 
for i in range(1,len(csv_files_training)): 
    new_doc=pd.read_csv(csv_files_training[i],sep=';') 
    main_Training=main_Training.append(new_doc, ignore_index=True) 
#print(main_Training)
print ('Total Images found in ',Training_Images_Directory,' :',len(main_Training))


sss=[] 
oneexample=[]   
for i in range(len(main_Training.values)): 
    if main_Training.values[i,-1] not in sss:
        oneexample.append(main_Training.values[i,:]) 
        sss.append(main_Training.values[i,-1]) 
        
One_Example=pd.DataFrame(oneexample,columns=['Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])
#print(One_Example)

plt.figure(figsize=[14,25]) # set image size
plt.subplots_adjust(wspace = 0.2)# set distance between the subplots


#in the below code we are just showing all traffic sign(class) stored in one example by plotting on console
i = 0 
for i in range(len(One_Example)): 
    img_path=csv_files_training[i].split('GT')[0]+One_Example['Filename'][i] 
      
    img = cv2.imread(img_path) # opencv use for reading image.
    # croping the image based on the coordinates given us by csv files
    crop_image=img[One_Example['Roi.Y1'][i]:One_Example['Roi.X2'][i],One_Example['Roi.X1'][i]:One_Example['Roi.Y2'][i]]
    #converting the color RGB so that we can actually view it.
    crop_image=cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB)
    plt.subplot(13,5,i+1)
    i+=1
    imgplot = plt.imshow(crop_image)
plt.show() # plot showing at the end. in console


#in the below code we are just saving each unique classes in classes_images folder
print ('Saving the classes Images in "classes_images"')
for i in range(len(One_Example)):
    img_path=csv_files_training[i].split('GT')[0]+One_Example['Filename'][i] 
    img = cv2.imread(img_path) # opencv use for reading image.

    # croping the image based on the coordinates given us by csv files
    crop_image=img[One_Example['Roi.Y1'][i]:One_Example['Roi.X2'][i],One_Example['Roi.X1'][i]:One_Example['Roi.Y2'][i]]
    cv2.imwrite('classes_images/'+classnames[str(One_Example['ClassId'][i])]+'.png',crop_image) # use for saving the image.


def images_to_hog(main,Images_Directory): # function defining that can be call for both test and training
    Features=[]
    Labels=[]
    for i in range(0,len(main)): #len(main)
        img_path=Images_Directory+'/00000'[:-len(str(main['ClassId'][i]))]+str(main['ClassId'][i])+'/'+main['Filename'][i]
        img = cv2.imread(img_path) # opencv use for reading image.
        # croping the image based on the coordinates given us by csv files
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.medianBlur(img,3)            #image smoothing or blurring is done to remove salt pepper noise
        crop_image=img[main['Roi.Y1'][i]:main['Roi.X2'][i],main['Roi.X1'][i]:main['Roi.Y2'][i]]
        crop_image=cv2.resize(crop_image, (64, 64)) #Resize the image to 64*64.
        # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
        # need to calculate.
        ret,crop_image = cv2.threshold(crop_image,127,255,cv2.THRESH_BINARY)
        descriptor = hog(crop_image, orientations=8,pixels_per_cell=(4,4))
        Features.append(descriptor)#hog features saving
        Labels.append(main['ClassId'][i])#class id saving
    
    Features=np.array(Features)# converting to numpy array.
    Labels=np.array(Labels)
    return Features,Labels

Features_Training,Labels_Training=images_to_hog(main_Training,Training_Images_Directory) # giving values to images_to_hog function
print ('Training HOG output Features shape : ',Features_Training.shape)
print ('Training HOG output Labels shape: ',Labels_Training.shape)


# feteching the testing dataset
Test_Images_Directory='dataset/Testing'


# below combining the csv files of testing dataset together
csv_files_Testing=glob.glob(Test_Images_Directory+'/**/*.csv',recursive=True) 
main_Testing=pd.read_csv(csv_files_Testing[0],sep=';') 
for i in range(1,len(csv_files_Testing)):
    new_doc=pd.read_csv(csv_files_Testing[i],sep=';')
    # appending the csv files making a big csv that consists of all the csv files.
    main_Testing=main_Testing.append(new_doc, ignore_index=True)
#print(main_Testing)
print ('Total Images found in ',Test_Images_Directory,' :',len(main_Testing))


# calling the images_to_hog function declared above and using it on test dataset
Features_Testing,Labels_Testing=images_to_hog(main_Testing,Test_Images_Directory) # giving values to images_to_hog function
print ('Testing HOG output Features shape : ',Features_Testing.shape)
print ('Testing HOG output Labels shape: ',Labels_Testing.shape)


# Applying PCA (PRINCIPAL COMPONENET ANALYSIS)
pca = PCA(n_components = 40)
X_train = pca.fit_transform(Features_Training)  
X_test = pca.transform(Features_Testing)   

print ('New Train Dataset shape after PCA: ',X_train.shape)
print ('New Test Dataset shape after PCA: ',X_test.shape)


# Fitting classifier to the Training set
classifier=SVC(kernel='rbf',gamma='scale') # Calling the function SVC to implement SVM
classifier.fit(X_train,Labels_Training) # Training the Classifier on Train date

print ('SVM Mean Accuracy of Training dataset: ',classifier.score(X_train,Labels_Training))
print ('SVM Mean Accuracy of Test dataset: ',classifier.score(X_test,Labels_Testing))

# Saving the model. this model will be used when we test on on new data rather then train each time.
print ('saving the weights')
joblib.dump(pca, 'pca.pkl') 
joblib.dump(classifier, 'svm.pkl') 
print ('Program Complete')

