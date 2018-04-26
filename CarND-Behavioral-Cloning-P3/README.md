# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior - **done**
* Build, a convolution neural network in Keras that predicts steering angles from images - **done**
* Train and validate the model with a training and validation set - **done**
* Test that the model successfully drives around track one without leaving the road - **done**
* Summarize the results with a written report - **done**


[//]: # (Image References)

[image1]: ./examples/7.png "Model Visualization"
[image2]: ./examples/1.png "Grayscaling"
[image3]: ./examples/2.png "Recovery Image"
[image4]: ./examples/3.png "Recovery Image"
[image5]: ./examples/4.png "Recovery Image"
[image6]: ./examples/5.png "Normal Image"
[image7]: ./examples/6.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality Criterias

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* easytrack_model.h5 containing a trained convolution neural network for easy track
* hardtrack_model.h5 containing a trained convolution neural network for hard track
* writeup_report.md summarizing the results
* drive_robust.py contains updated drive code as car can get stuck in hard track
* run_final.mp4 is video of the easy track
* https://youtu.be/6ofKRA1ze_k is video of easy track
* https://youtu.be/lfYGOFnpxpU is video of Tough Track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh python drive.py easytrack_model.h5  ```
and
```sh
python drive_robust.py hardtrack_model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I am using the NVIDIA Behavioral Learning Architecture with addition of Batch-Norm Layers. My model consists of a convolution neural network with 3 5x5 filter size Conv layers followed by 2 3x3 filter size Conv layers and depths between 24 and 64 (model.py lines 98-106) 

The model includes RELU layers to introduce nonlinearity and Batch Normalization for efficient training. The data is normalized in the model using a Keras lambda layer (code line 96) and the images are clipped by topCropping = 50 & bottomCropping=25. 

#### 2. Attempts to reduce overfitting in the model
 
The model contains dropout layers in order to reduce overfitting (model.py lines 111 & 114).I have applied dropout only on the FC layers but as of late they are getting used even in Conv layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77-91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and  right sides of the road. I realized capturing the recovering data was very hard task so I also captured slightly choppy driving(continous left/right steer). This stopped my model from overfitting as on a simple track (L->R) and (R->L) transitions are too less to learn from.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure effective features could be learnt from the Image data and make it robust to small changes.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it was proven on an actual Car and Road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding Dropout to he fully connected layers.

Then I added Batch Normalization for faster training and also regularization.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, praticularly around rough corner.To improve the driving behavior in these cases, I captured more train data where I captured hard turns to recover from rough corners.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
Detailed analysis of the layers is as follow:

| Layer                         |     Description                                                               |
|:---------------------:|:-------------------------------------------------:|
| Input                         | 320x160 RGB image -> 320x85(Cropped)                                                            |
| Convolution 5x5       | Depth - 24, 2x2 stride, valid padding         |
| RELU                  |                                               |
| BATCH NORMALIZATION    |                                                                                                       |                                                                       |
| Convolution 5x5       | Depth - 36, 2x2 stride, valid padding         |
| RELU                                							      |
| BATCH NORMALIZATION    |                                                                                                       |                                                                       |
| Convolution 5x5       | Depth - 48, 2x2 stride, valid padding         |
| RELU                                  |
| BATCH NORMALIZATION    |                                                                                                       |                                                                       |
| Convolution 3x3       | Depth - 64, 1x1 stride, valid padding           |
| RELU                                  |
| BATCH NORMALIZATION    |                                                                                                       |                                                                       |
| Convolution 3x3       | Depth - 64, 1x1 stride, valid padding           |
| RELU                                  |
| BATCH NORMALIZATION    |                                                                                                       |                                                                       |
| Fully connected               | shape = ?x512              |
| BATCH NORMALIZATION    |
| Dropout                        | Probability=0.5             |
|                                                                       |
| Fully connected               | shape = 512x100              |
| BATCH NORMALIZATION                                                   |
| Dropout                        | Probability=0.5             |
|                                                                       |
| Fully connected               | shape = 100x50              |                                             |                                                                       |
| Fully connected               | shape = 50x10              |                                             |                                                                       |
| Fully connected               | shape = 10x1              |                                             |                                                                       |
| Regression Output             | shape = [1]                      |


The final model architecture (model.py lines 95-119) consisted of a convolution neural network with the following layers and layer sizes:


Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving(anti-clockwise). Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when its going of the track. These images show what a recovery looks like starting from going out of the track and then take hard left to recover :

![alt text][image3]                         ![alt text][image4]                            ![alt text][image5]

I also collected data by going clockwise on the track to improve the robustness of the model. Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles to make my model robust. For example, here is an image that has then been flipped:

![alt text][image6]                                               ![alt text][image7]

The corresponding change to the steering is to take negative of the measurement. Also I used the left and right camera Images(proxy to center image) and factored in the perceptive angle by subtracting 0.3 from the steering measurement to fit the decision applicable to similar center Image.
After the collection process, I had ~25000 number of data points. I then preprocessed this data by normalazing the image to [0-1] range. I also cropped the top and bottom portion of the Image to remove the regions which are redundent. I also augment the data by taking the left/right camera image and also the flipped Images. I found an interesting fact that leaving uncropped sky in the images have very adverse effects on the learning.


I finally randomly shuffled the data set and put 20% of the data into a validation set and I was also using another dataset for testing, followed by testing on the real track..

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by accuracy stablizing on validation data. I used an adam optimizer so that manually training the learning rate wasn't necessary.

A key problem I faced was the car was getting stuck in certain portions of the tough track. This happens very regularly. I ended up writing a PID for getting it out. I have included as part of the drive_robust.py