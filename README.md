# Traffic_Sign_Detection_Recognition

Steps to train the model and run the code:
step 1: download the repository
step 2: install required libraries given in code
step 3: run the svm_hog_training.py file the model will get trained
step 4: run the svm_test_on_image.py file to test on input image from test dataset
step 5: run the detection.py file to do the detection part
step 6: run the detection_on_each.py file to do the detection part (you will get the outputs of what is happening in each steps)
        you can run either detection.py or detection_on_each.py thats fine too
step 7: run the gui.py file to upload multiple input traffic sign images and to get their output

OR

steps to run the code using already trained model:
step 1: you can just run the gui.py file after downloading everything as I have also uploaded the trained models pca.pkl and svm.pkl so you can directly use them to refer

Technology used:
I have used a combination of Hog,Pca and SVM to do the recognition part
and a combination of Mser and Hsv to do the detection part



