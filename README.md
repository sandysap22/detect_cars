## Vehicle Detection Project
### Author : Sandeep Patil

[//]: # (Image References)

[demo]: ./sample_images/demo.gif  "demo"
[bonding_boxes]: ./sample_images/bonding_boxes.png  "bonding_boxes"
[car_cb_channel_hog]: ./sample_images/car_cb_channel_hog.png  "car_cb_channel_hog"
[car_cb_hog_histogram]: ./sample_images/car_cb_hog_histogram.png  "car_cb_hog_histogram"
[car_cr_channel_hog]: ./sample_images/car_cr_channel_hog.png  "car_cr_channel_hog"
[car_cr_hog_histogram]: ./sample_images/car_cr_hog_histogram.png  "car_cr_hog_histogram"
[car_histogram]: ./sample_images/car_histogram.png  "car_histogram"
[non_car_histogram]: ./sample_images/non_car_histogram.png "non_car_histogram"
[car_y_channel_hog]: ./sample_images/car_y_channel_hog.png  "car_y_channel_hog"
[car_y_hog_histogram]: ./sample_images/car_y_hog_histogram.png  "car_y_hog_histogram"
[final_boxes]: ./sample_images/final_boxes.png  "final_boxes"
[heat_map]: ./sample_images/heat_map.png  "heat_map"
[image_pipe_line]: ./sample_images/image_pipe_line.png  "image_pipe_line"
[non_car_cb_channel_hog]: ./sample_images/non_car_cb_channel_hog.png  "non_car_cb_channel_hog"
[non_car_cb_hog_histogram]: ./sample_images/non_car_cb_hog_histogram.png  "non_car_cb_hog_histogram"
[non_car_cr_channel_hog]: ./sample_images/non_car_cr_channel_hog.png  "non_car_cr_channel_hog"
[non_car_cr_hog_histogram]: ./sample_images/non_car_cr_hog_histogram.png  "non_car_cr_hog_histogram"
[non_car_y_channel_hog]: ./sample_images/non_car_y_channel_hog.png  "non_car_y_channel_hog"
[non_car_y_hog_histogram]: ./sample_images/non_car_y_hog_histogram.png  "non_car_y_hog_histogram"
[sample_car]: ./sample_images/sample_car.png  "sample_car"
[sample_non_car]: ./sample_images/sample_non_car.png  "sample_non_car"
[window_ranges]: ./sample_images/window_ranges.png  "window_ranges"



The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier.
* Append binned color features, as well as histograms of color, to HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

![demo][demo]

## Overview
When we drive, we can easily detect various objects around us. To take course of actions at different points we need the information of objects around car.
Though it is obvious to us detect objects like other cars, pedestrians etc. Its not easy for computers to detect is from cloud of pixel values. 
In this project I am going to use traditional machine learning algorithms to classify cars and no-cars objects and then will draw rectangles around cars. Following sections would illustrate different steps.

## Train classifier to classify to detect cars.
## Data for classification.
I have used data set of 8K images provided by udacity for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)  images.

Following are details of dataset. 

Number of car samples  :8792  
image size             :64 x 64 x 3  
Number of car samples  :8968  
image size             :64 x 64 x 3  

Sample car images  
![sample_car][sample_car]
Sample non car images  
![sample_non_car][sample_non_car]


## Choose features to train classifier.
Its important to get proper features from images to feed it to classifier. I had done some research and found following features more relevant to use for classification.

1. The histogram of colors 
2. Hog feature histogram of Y, Cr and Cb channels of `YCrCb` color space.

Please refere [this](https://github.com/sandysap22/detect_cars/blob/master/image_feature_details.ipynb) notebook for details about different features.

In following images by looking at histograms only we can clearly distinguish if image is car or non car.


### Histogram of color intensities for Car images
![car_histogram][car_histogram]
### Histogram of color intensities for Car images
![non_car_histogram][non_car_histogram]

Similarly we can clearly distinguish if image is car or non car from histogram of hog features of Y, Cr and Cb channels.

### Hog features for Y channel for cars
![car_y_channel_hog][car_y_channel_hog]
### Hog features for Y channel for non cars
![non_car_y_channel_hog][non_car_y_channel_hog]

### Histogram of Hog features for Y channel for cars
![car_y_hog_histogram][car_y_hog_histogram]
### Histogram of Hog features for Y channel for non cars
![non_car_y_hog_histogram][non_car_y_hog_histogram]

### Hog features for Cr channel for cars
![car_cr_channel_hog][car_cr_channel_hog]
### Hog features for Cr channel for non cars
![non_car_cr_channel_hog][non_car_cr_channel_hog]

### Histogram of Hog features for Cr channel for cars
![car_cr_hog_histogram][car_cr_hog_histogram]
### Histogram of Hog features for Cr channel for non cars
![non_car_cr_hog_histogram][non_car_cr_hog_histogram]

### Hog features for Cb channel for cars
![car_cb_channel_hog][car_cb_channel_hog]
### Hog features for Cb channel for non cars
![non_car_cb_channel_hog][non_car_cb_channel_hog]


### Histogram of Hog features for Cb channel for cars
![car_cb_hog_histogram][car_cb_hog_histogram]
### Histogram of Hog features for Cb channel for non cars
![non_car_cb_hog_histogram][non_car_cb_hog_histogram]

## Concatenate and normalize features
I have concatenate above features for each image and normalized each feature on whole dataset using sklearn.preprocessing.StandardScaler() api.
Then I splited data in ratio of 80:20 for training and testing.

## Choose classifier from different options.
I have trained following classifiers on above dataset.
1. Gaussian Naive Bayes
2. Random Forest Classifier
3. Linear SVC

Following are accuracy results for 100% data set :  
GaussianNB test accuracy    = 0.854  
Linear SVC test accuracy    = 0.983  
Random Forest test accuracy = 0.968  

Following are fbeta_score for 100% data set with beta=0.1  
I have given more preference to precision over recall.  Idea is to get higher accuracy with less false positives results.  
GaussianNB f_test    = 0.807  
Linear SVC f_test    = 0.976  
Random Forest f_test = 0.988 

#### Based on fbeta_score I have chosen *Random Forest* as my classifier. 

## Choose best parameters for classifier.
1. I have used sklearn.model_selection.GridSearchCV to choose my parameters.  
2. Following are parameter of best classifier  
max_depth: 30  
min_samples_split: 16   
min_samples_leaf: 2  
estimators: 20  

## Use classifier to detect cars in images
Following steps are followed to detect car in images using classifier. Please refere [this](https://github.com/sandysap22/detect_cars/blob/master/detect_cars.ipynb) notebook for all details.
1. Get hog features for different regions with different scales.
   Extract image in following patches then, rescale it to match with sliding window and get hog features for entire patch.  
   height 400 to 496 from top with scale 1.0  
   height 400 to 528 from top with scale 1.5  
   height 400 to 528 from top with scale 2.0  
   height 400 to 592 from top with scale 3.0  
   
   ![window_ranges][window_ranges]
3. Extract hog features with **sliding window**.  
   After getting hog feature for patch I have extracted features for 64x64 sliding window.
   I have used following parameters to extract hog features.  
   orient=9,  
   pix_per_cell=8,  
   cell_per_block=2
4. Extract histogram features with **sliding window**.  
   On rescaled path I have extracted color intensity histogram with bin size equal to 32.
5. Concatenate hog and histogram features for each window.  
   I have concatenated hog and histogram and normalized it with StandardScaler.
6. Feed features for classifier.  
   If classifier detects it as car then treat sliding window region (re scaled) as car position.
7. Combine results from different regions of sliding window.  
   Most of times car get detected in multiple regions of sliding window, so I have used heat map to get combined effect.
8. Apply threshold on heats.  
   I have thresolded out heats which are less than 2, with this I was able to eliminated false positive results.
9. Apply label to detected regions.   
   I used scipy.ndimage.measurements.label api to combine different regions to get one bounding box to represent one car.

#### Sample images with detection boxes, heat map and final bounding box by applying labels.
![image_pipe_line][image_pipe_line]
    
## Video pipeline

While processing on video frames I have tried to average out size of detected frames and relative position with last 3 frames. For that I have implemented 
following classes.  
PostitionsQueue  
Car and  
CarTracker

Following is sample of video.
![demo][demo]

You can find whole video [here.](https://github.com/sandysap22/detect_cars/blob/master/project_video_output.mp4)

### Discussion

#### I faced following challenges.
1. Initially I used smaller data set for training classifier which was giving 99% accuracy on test data set,
however with this in the actual video the classifier was not able to detect black car. So I re trained model on larger dataset.
2. In some portion of video we can see that white and black car get detected as one car. It is due to improper labeling. I need to improve my solution to label with different configuration.
3. It is very important to feed classifier with fetures from proper region of frame with proper scaling. I spent most of time to fine tune sliding window region and its respective scale.  

#### Alternate approach.
As current approach have 3 steps. First is proper feature selection, second is to train classifier 64x64 pixel images and third is adjust input to feed it to classifier using window slide and scale code.
With use of neural network like YOLO we can combine these steps and we would not have to worry for which feature to select and which scale to apply on different regions. The one neural net could learn relevant features and varying scales of objects at different regions.


