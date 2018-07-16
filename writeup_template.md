## Writeup Template
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/extra27_copy.png
[image2]: ./output_images/image0533_copy.png
[image3]: ./output_images/hog%20_feature.png
[image4]: ./output_images/search_window.png
[image5]: ./output_images/detection_with_heatmap.png
[image6]: ./output_images/SciPy_Heatmap.png
[image7]: ./output_images/with_both_car.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained  in cell(1) of the file called ` main.ipynb '  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle`
and `non-vehicle` classes:

![alt text][image1] 

![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and 
`cells_per_block`).  I grabbed random images from each of the two classes. The code for extracting HOG features from an 
image is defined by the method get_hog_features and is contained in the cell titled "Define a function of return HOG 
features and visulization " 
The figure below shows a comparison of a car image and its associated histogram of oriented gradients, 
as well as the same for a non-car image.

Here is an example using the  
color space `RGB`
HOG parameters of orientations= 9, 
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)

![alt text][image3]



#### 2. Explain how you settled on your final choice of HOG parameters.

I chose my final choice of HOG parameters based upon the performance of the SVM classifier produced using them. I 
considered not only the accuracy with which the classifier made predictions on the test dataset, There is a balance to 
be struck between accuracy and speed of the  SVM classifier.
So I decided final parameters be like :

color space = 'RGB'
orient = 9 
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = "ALL"
spatial_size = (32, 32) 
hist_bins = 32



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features  (and color features if you used them).

I trained a linear SVM using default classifier parameters and using HOG features .In the cell no[] titled  part 
"Train a Classifier" and was able to achieve a test accuracy of 97.52%. (In cell 3)



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

![alt text][image4]

This image show some random check
I decided to search random window positions at random scales all over the image. In this, I adapted the method 
for car finding from "search window" for finding car we are well known with color,position in image, shape and apperent 
size.The method combines HOG feature extraction with a sliding window search.The HOG features are extracted for the entire 
image (or a selected portion of it i.e.x_start_stop and y_start_stop) and then these full-image features are subsampled 
according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the 
HOG features for each window region and returns objects corresponding to the windows that generated a positive ("car") prediction.
slide window is also the important part because it helps to detect false positive or correct predictions.for serch window 
I decided following parameters :

y_start_stop_list = [[350,656], [350,500], [350,600]]
x_start_stop_list = [[None,None], [800,None], [800,None]]
overlap_list = [0.50, 0.75, 0.75]
xy_window_list = [(96, 96), (64, 64), (96, 128)]
color_list = [(0,0,255), (0,255,0), (255,255,255)]

x_start_stop and y_start_stop gives us the visualized area where we can find the vehicles. the overlaping gives us accurate 
heatmap.(In cell 4)



#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using RGB 3-channel HOG features plus spatially binned color and histograms of color 
in the feature vector, which provided a nice result.  The final implementation performs very well, 
identifying the near-field vehicles in each of the images with no false positives.
The first implementation did not perform as well,  by optimizing the SVM classifier I overcome this . The original classifier 
runs with only one channel it gives us some accuracy but when we moved to all channel it gives us very appropriate and 
increased accuracy  but also tripled the execution time.The optimization techniques included changes to window sizing and 
overlap as described above, and lowering the heatmap threshold to improve accuracy of the detection (higher threshold values 
tended to underestimate the size of the vehicle).
Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time  with minimal false positives.)

Here's a [link to my video result](./project_video_output_4.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob 
corresponded to a vehicle.I constructed bounding boxes to cover the area of each blob detected.with the exception of 
storing the detections of rectangle from video. Rather than performing the heatmap/threshold/label steps for the current 
frame's detections, the detections for the past frames are combined and added to the heatmap and the threshold for the 
heatmap . (In cell 5)

Here's an example result showing the heatmap from a series of frames of video, the result of 
`scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy. Balancing the 
accuracy of the classifier with execution speed was crucial. Scanning  windows using a classifier that achieves 97% 
accuracy .
It also introduces another problem that vehicles that significantly change position from one frame to the next 
(e.g. oncoming traffic) will tend to escape being labeled. so we have to Producing a very high accuracy  classifier and maximizing 
window overlap might improve the per-frame accuracy ,but it would also be far from real-time.The pipeline is probably 
most likely to fail incases where vehicles don't resemble those in the training dataset, butlighting and environmental 
conditions might also play a role (e.g. a white car against a white background). 
