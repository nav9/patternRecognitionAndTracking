# patternRecognitionAndTracking
A program that uses Principal Component Analysis (PCA) and Histograms of Oriented Gradients (HOG) to recognize faces and be able to track them in a video

HOG and HOG2 have been implemented by Sanyam and Ludwig. The necessary attribution and license is mentioned within the respective files. 

The sameFace.m program uses Dr.Libor Spacek’s collection of facial images. You'll have to download the dataset yourself. The program trains a Naive Bayes classifier to be able to distinguish between male and female images after obtaining faces as Eigen Faces.

The video tracking program uses the Visual Tracker Benchmark dataset and attempts two methods. 1. Extract features using HOG, reduce those features with PCA and be able to track a face in a video using Naive Bayes or SVM (Support Vector Machines), given the location of the face in only the first frame. Method 2 takes the face ground truth from all frames, trains a Naive Bayes classifier or an SVM to be able to predict and track the face.
