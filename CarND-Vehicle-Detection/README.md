# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier - **done**
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. - **done** 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing. - **done**
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images. - **done**
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. - **done**
* Estimate a bounding box for vehicles detected. - **done**

[//]: # (Image References)
[image1]: ./test_output/001.png
[image2]: ./test_output/00.png
[image30]: ./test_output/scale_1_1.png
[image31]: ./test_output/scale_1_5to1_25.png
[image32]: ./test_output/scale_2to1_75.png
[image33]: ./test_output/scale_2_5to2.png
[image40]: ./test_output/1.png
[image41]: ./test_output/2.png
[image42]: ./test_output/3.png
[image43]: ./test_output/4.png
[image44]: ./test_output/5.png
[image45]: ./test_output/6.png
[image5]:  ./test_output/55.png
[image6]:  ./test_output/66.png
[image7]:  ./test_output/77.png
[video1]: ./test_videos_output/video2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. In terms of colorspace HSV performed much better than RGB, YCbCr. Saturation and Hue had higher contributions to the accuracy. Since we are calculating HOG features only once and then resampling it, I incuded all the 3 channels(HSV). Here is the list of experiments.

| HOG        | Spatial           | HIST  |        Feature Size     | Accuracy |
|:-------------:|:-------------:|:-----:|:---------------------------:|:---------:|
| HSV(8x8,9)    | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> |       6156 |99.02|
| HSV(8x8,9)    |   <ul><li>[ ] </li></ul>|   <ul><li>[x] </li></ul> |           6061 | 98.4|
| HSV(8x8,9)    |     <ul><li>[x] </li></ul>|  <ul><li>[ ] </li></ul> | 5388 | 98.26|
| HSV(8x8,9)    |   <ul><li>[ ] </li></ul> | <ul><li>[ ] </li></ul> | 5292 |96.47|
| YCrCb(8x8,9)  | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul>|6156 | 98.71|
| YCrCb(8x8,9)  | <ul><li>[ ] </li></ul> | <ul><li>[x] </li></ul>|6061 | 98.51|
| YCrCb(8x8,9)  | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul>|5388 | 98.15|
| HSV(16x16,15) | <ul><li>[x] </li></ul>|<ul><li>[x] </li></ul> | 2484 | 99.4|


 Pixels per cell of 8x8 with HSV showed very good results but it was slow so I tried 16x16 pixel per cell. This also worked well,so I settled on `orientations=15`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`as the features extraction was much faster.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of the Spatial Features, Histogram Features & HOG Features. I found that non-linear SVM with 'rbf' kernel performed much better. Here is the training output:

Feature vector length: 2484

Test Accuracy of SVC =  99.4%.

My SVC predicts:  [0. 0. 1. 1. 0. 1. 1. 0. 1. 1.]

For these 10 labels:  [0. 0. 1. 1. 0. 1. 1. 0. 1. 1.]`

I was able to get really good accuracy on SVC but the performance on the windows from the video was not too high. I think it is because of perspective issues as the images in training set are for cars which are in front and not in side lanes. I also did some hard negative mining to supplement the training data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows

There is perspective effects when cars appear in side lanes. This can be handled by using differential scale along x and y axis. Also cars can appear close or far, so it makes sense to use multiple scales. Also we can use smaller scales for cars far in front and larger for cars closer. I tried multiple options but settled for these 4.

|Scale(x axis) | Scale(y axis)|
|:------------:|:------------:|
| 1 | 1|
|1.5|1.25|
|2|1.75|
|2.5|2|

![alt text][image30] ![alt text][image31]

![alt text][image32] ![alt text][image33]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I was able to get to detect the cars in the video but as expected the algorithm was very slow when I tried to run it on multiple-scales. I was able to use heatmap derived from results of multiple scales to improve my accuracy.I tried to use the fact the cars when closer(i.e. bottom of the image) are larger compared to when they are far off. So I choose my start & stop image height windows depending on the scale. For small scales I search on top of the road Image and bigger scales on the bottom of road Image. This resulted in some speedup and I could use upto 4 scales.
I also used a higher pixels_per_cell(16x16). Here is the result for the test_images.

![alt text][image40] ![alt text][image41]
![alt text][image42] ![alt text][image43]
![alt text][image44] ![alt text][image45]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
Another key factor was that the video is around 25 fps, we can do the car detection every 3rd frame and have a real time detection.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I realized non-linear kernel SVM had much better accuracy than linear SVM but was much slower. To handle cars at different distance, I had to use multiple scales but this was slowing down algorithm a lot. I end up running the algorithm every 3rd frame. The algorithm will fail when car is switching lanes as well as particles on the road. Augmenting the data with hard mining and realistic data from Lanes will improve the accuracy even more.

