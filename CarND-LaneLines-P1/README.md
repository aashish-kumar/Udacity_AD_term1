# **Finding Lane Lines on the Road** 

## Problem statement - Challenges
The problem is interesting to solve as there are lot of variations in road conditions.
Optional Problem had following additional challenges:
1) Varying Illumination
2) Curve road
3) Shadows and Glares
4) Bridge road very different from normal road

---
## Reflection

### 1. Pipeline Details

#### My Pipeline consists of following steps:

#### 1) Identify Lane line marking based on edge information
I convert three-channels(RGB) to maximum of 3 channels(RGB), as that shows better highlight effect for white-lines compared to colored lines. I modified the draw-lines function to draw lines in binarized image. I used those hog lines to create the marking and then used morphological closing to make it reliable. Then I applied two masks, one for removing redundant outside regions and other for inside regions. 

<img src=./temp/white_0.jpg width="250"> <img src=./temp/white_1.jpg width="250"> <img src=./temp/white_2.jpg width="250">

#### 2) Identify Lane line marking based on color information
I converted the RGB image to HSV Image. Then I used the opencv in-range function to retrieve the yellow color markings from the image. I then used morphological operation to clean it up and applied the mask to remove redundant regions.

<img src=./temp/yellow_0.jpg width="250"> <img src=./temp/yellow_1.jpg width="250"> <img src=./temp/yellow_2.jpg width="250">

#### 3) Use the result to generate skelton of the Lane lines
The results from (1) and (2) were combined and the resultant image was skeltonized by morphological operations. This helps us in getting single pixel-wide lines in places of thick lane markings. 

<img src=./temp/skel_0.jpg width="250"> <img src=./temp/skel_1.jpg width="250">

#### 4) Fit first-order and second-order polynomial to the lane markings
The mask Image from (3) was converted to indexed image and the white pixels were extracted. The coordinates of white pixels were divided into left & right lane markings. Then I applied a RANSAC based curve fitting(polyfit) of first and second order polynomials(if first order fails) on the (x,y) coordinates. I also used the last-frames estimated polynomial in-case no reliable lines are estimated.

#### 5) Add back to the original input image
The curves identified in step 4 were drawn on a mask which was then added back to the original Image.

<img src=./temp/yellow_0.jpg width="250"> <img src=./temp/lane_mask.jpg width="250"> <img src=./temp/final.jpg width="250">

### 2. Potential shortcomings 
i) Will perform poorly for highly curvy roads as the curve-fitting will misbehave.
ii) Vehicle colors same as the line color can be a problem if they are captured too close to the camera and lines.
iii) Using Temporal information(time-axis) has potential to cause problems if lane lines go missing(actual/our_processing) for a long time.


### 3. Possible improvements to pipeline
i) Perspective correction can be done and then problem will be reduced to find two parallel lines. This will help in finding lane-lines, in case of partial visible lanes.
ii) Curve fitting can be improved by a better preprocessing module.
iii) Yellow lines always occur on the left side. This information can be used to improve yellow lane line detection.
iv) Temporal Data across different frame can be used in a more refined manner to add missing information for better curve fitting.

### Reference/Code Usage Sources:
http://opencvpython.blogspot.in/2012/05/skeletonization-using-opencv-python.html
https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c
