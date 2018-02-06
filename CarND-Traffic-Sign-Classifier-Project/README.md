# **Traffic Sign Recognition** 

## Problem Statement: **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set) - **done**
* Explore, summarize and visualize the data set - **done**
* Design, train and test a model architecture - **done**
* Use the model to make predictions on new images - **done**
* Analyze the softmax probabilities of the new images - **done**
* Summarize the results with a written report - **done**


[//]: # (Image References)

[image1_0]: ./writeup_images/train_distrib.png "Visualization_train"
[image1_1]: ./writeup_images/test_distrib.png "Visualization_test"
[image1_2]: ./writeup_images/valid_distrib.png "Visualization_valid"
[image1_2d]: ./writeup_images/visu.png "Visualization_Images"

[image2_1]: ./writeup_images/input_forgray.png "darkinput"
[image2_2]: ./writeup_images/gray.png "darkoutput"

[image3_1]: ./writeup_images/dark_test_image.png "input"
[image3_2]: ./writeup_images/dark_histo.png "output"

[image4_1]: ./writeup_images/rg_chroma.png "input2"
[image4_2]: ./writeup_images/bg_chroma.png "output2"

[image5_1]: ./writeup_images/f1.png "input3"

[image6_1]: ./writeup_images/lr_ix_1.png "flip1"
[image6_2]: ./writeup_images/lr_ix_2.png "flip2"
[image6_3]: ./writeup_images/lr_ix_3.png "flip3"
[image6_4]: ./writeup_images/lr_ix_4.png "flip4"

[image7_1]: ./writeup_images/lr26_1.png "fliplr"
[image7_2]: ./writeup_images/lr26_2.png "fliplr2"

[image8_1]: ./writeup_images/lr_ud_1.png "lrud"
[image8_2]: ./writeup_images/lr_ud_2.png "lrud2"

[image9_1]: ./writeup_images/warp1.png "pp1"
[image9_2]: ./writeup_images/warp2.png "pp2"
[image9_3]: ./writeup_images/warp3.png "PP3"


[image10]: ./writeup_images/network.png "network"

[image11]: ./writeup_images/res1.png "res1"

[image12]: ./writeup_images/res2.png "res2"

[image4]: ./images_internet/1.jpg "Traffic Sign 1"
[image5]: ./images_internet/2.jpg "Traffic Sign 2"
[image6]: ./images_internet/3.jpg "Traffic Sign 3"
[image7]: ./images_internet/4.jpg "Traffic Sign 4"
[image8]: ./images_internet/5.jpg "Traffic Sign 5"
[image81]: ./images_internet/6.jpg "Traffic Sign 6"
[image82]: ./images_internet/7.jpg "Traffic Sign 7"
[image83]: ./images_internet/8.jpg "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

I tried an interesting approach of building the whole network of my own with inspiration from Inception module in googlenet as I wanted to explore the hyperparameters impact on learning. Here is a link to my [project code](https://github.com/aashish-kumar/Udacity_AD_term1/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed across multiple classes. Its very evident from the data that some of the classes are under represented

![alt text][image1_0]

Also the distribution is quite similar in the validation and test data set

![alt text][image1_1] ![alt text][image1_2]

Below is a glimpse of the Images in the Dataset and corresponding labels

![alt text][image1_2d]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried creating a 1x1xn convulational layer which can take the RGB image and learn the preprocessing while training on its own. This approach was encouraging but the improvements were not enough. I think that the data size was not big enough to enable good preprocessing to learn.

Then I decided to convert the images to grayscale. This improved the accuracy by 1-2 percent. 
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2_1]  ![alt text][image2_2]

On careful observation I found that some of the images were not well lit. So I did histogram equalization using openCV.
This improved my validation accuracy. Here is an example of histogram equalization:

![alt text][image3_1]  ![alt text][image3_2]

Additionally since the color information was not incorporated in above techniques, I converted the image to HSV format and took the Hue component as another channel. Additionally I used a chroma-rg and chroma-bg as another color component. This all resulted in gradual improvements to the validation accuracy. Here is an example of original, chroma-rg and chroma-bg sample:

![alt text][image2_1]  ![alt text][image4_1]  ![alt text][image4_2]


**Data Augmentation**

Based on the data visualization, I realized that the data samples are unevenly distributed across different classes. So I decided to augment my data. 

I did it in 3 differnt way

1) **Class specific Augmentation**
    I tried three different class specific augmentation. There are other classes which can be augmented too. I choose the few ones based on the F1 score vs data distribution graph.
    
    **Left-Right FLIP different class:**
    
    ![alt text][image6_1]  ![alt text][image6_2]
    
    and
    
    ![alt text][image6_3]  ![alt text][image6_4]
    
    **Left-Right FLIP same class:**
    
    ![alt text][image7_1]  ![alt text][image7_2]
    
    **Two signs which do not change under LR - UD FLIP same class:**
    
    ![alt text][image8_1]  ![alt text][image8_2]

2) **Perspective Transformation**
    I applied two perspective transforms to the train data set. This make sense to the problem as the traffic signs are captured from a distance. This results in images being deformed.
    An example from this augmentation:
    
    ![alt text][image9_1]  ![alt text][image9_2]  ![alt text][image9_3]
    
3) **Rotation transformation**
    I used tf.contrib.image.rotate to rotate the images during the training time. I choose the rotation to be small as larger rotation could change the sign type. On experimentation 0.15 radian turned out to be a good angle

The difference between the original data set and the augmented data set is the following:

Original Data-set samples: 38099

Augmented Data-set samples: 341001


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Here is the block diagram of my model:

![alt text][image10]

Detailed analysis of the layers is as follow:

| Layer         		|     Description	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   								| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x12 		|
| RELU					|													|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 		|
| RELU					|													|
| Max pooling 2x2      	| 2x2 stride, valid padding, outputs 14x14x32 		|
| Convolution 3x3 , 1x1	| 1x1 stride, same padding, outputs14x14x64,14x14x16|
| RELU					|													|
| CONCAT				| output: 14x14x80									|
| Max pooling 2x2      	| 2x2 stride, valid padding, outputs 7x7x80 		|
| Convolution 3x3 , 1x1	| 1x1 stride, same padding, outputs 7x7x128,7x7x32	|
| RELU					|													|
| CONCAT				| output: 7x7x160									|
| Max pooling 3x3      	| 2x2 stride, valid padding, outputs 3x3x160 		|
| Fully connected		| shape = 1440x512									|
| RELU					|													|
| Fully connected		| shape = 512x256									|
| RELU					|													|
| Fully connected		| shape = 256x43									|
| RELU					|													|
| Softmax				| 43 output class probabilities						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with the Lenet model as suggested in the problem statement. On training, I was able to get 91% accuracy on validation set as mentioned. I incorporated weight Regularization and dropout in the model and I was able to cross the 93% accuracy required for submission.

I wanted to make my own model based on my learnings in the class. So I took inspiration from the Lenet model and made my own model which uses concepts from inception module.  I used a 1x1 conv layer in the beginning so my model can learn the channel transform on its own while training. I also used 1x1 conv layers in parallel to 3x3 conv layers and concat them for better feature learning. I was using a GPU so I could use big batch sizes but I realize too big batch size was slowing my learning. I end up choosing 128 as my batch size as it was working fine for me. The number of epochs (>10) was sufficient. I settled on epoch = 30 as I could monitor the changes and select a good iteration. I settled on early stoppage at 98.4 accuracy. On experimentation I figured out that learning rate = 0.001 and regularization parameter=0.001 were getting me good convergence. I also tried reducing the sigma(0.1 -> 0.01) for the weight initialization. This resulted in slower convergence but no appreciable change in accuracy.

I tried to use batch norm in all the layers and was able to train the network. On using batch norm I had to choose higher learning rate. Regularization was not required at all. I was able to train the network but accuracy was ~97% and I was not able to improve it further. This was a good experience but I was expecting better result from these experiments. In the end I removed it as I was getting better results with my previous model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 99.0%
* test set accuracy of 97.1%

I tried to plot my class accuracies as F1 score (2*precision*recall/(precision + recall)) with the sample size. I identified that classes which have low samples are the ones with lower F1 score.
Here is the plot:

![alt text][image5_1]

If an iterative approach was chosen:
* **What was the first architecture that was tried and why was it chosen?** I choose Lenet as the first model as it was showing good promise with slight modification to stop overfitting.

* **What were some problems with the initial architecture?**
There was overfitting while training using Lenet model from the previous experiment. On further experimentation I realize the network was not complex enough to achieve higher accuracies so I redesigned it.

* **How was the architecture adjusted and why was it adjusted?**
To handle the problem of overfitting, I added L2-regularization and also the dropout layer. This had signifcant improvements to the accuracy. For my redesigned network I used 1x1 and 3x3 conv layers in parrallel so the low level features are not lost.This resulted in 1-2% improvement in validation accuracy. Then I concat the outputs and fed to the next layer. I also made change the size of fc-layers. I realized that increasing fc layer size was resulting in overfitting, so I made sure regularization and dropout layers are taking care of it.

* **Which parameters were tuned? How were they adjusted and why?**
 I tried different layer depths. After some experimentation I found the right balance between accuracy and layer depth. I also tried different pooling layer size & strides combinations, I found it interesting that on choosing a 3x3 pooling with 2x2 stride, it had 0.5-1% improvement on the accuracy. I think it might be because of edge effect in stride. I also tried different sigma(0.1->0.01) for weight initialization but that did not show any improvement. On further experimentation I realized the data size was not sufficient, so I tried different Augmentation techniques. I tried the rotation and perspective correction as two main Augmentation. I also tried to use the fact that some sign had vertical symmetry to augment them. I plot F1 score for class w.r.t sample distribution. This showed good correlation and also help me in understanding which classes are having lower accuracy. 

* **What are some of the important design choices and why were they chosen?** 
Convulational layers makes sense for this problem as the relative position of features in the image defines the sign type. The dropout layers are necessary as they add regularization to the model making it able to detect the correct class in-case some part of the image is washed out. Increasing the size of layers result in overfitting, so using L2-regularization keep it from learning big weights and hence stop overfitting.

I tried batch-normalization for all layers but was not able to fine-tune it to a good extent. My accuracy were in ~97% with batch-norm.

My test accuracy is not comparable to my validation set. I think there is still some amount of overfitting happening in the model. Possible approaches to fix this:
* Batch Normalization
* Bigger Dataset
* More augmentation
* More Fine-tuning
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image81] 
![alt text][image82] ![alt text][image83]  

The first image can be difficult as can be mixed up with pedestrians sign. The second, third and fourth images have lot of dust accumulated on sign board so they would appear noisy to the network. The sixth image has shadow falling on the image and the seventh image has perspective distortion. The eight image is a speed sign so the characters can be mixed ( 70 can be detected as 30)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					        |     Prediction	        		| 
|:-----------------------------:|:---------------------------------:| 
| Right of way next intersection| Right of way next intersection   	| 
| Priority Road     			| Priority Road 					|
| No Passing					| Slippery Road						|
| No Vehicles	      			| No Vehicles					 	|
| Keep Right					| Keep Right      					|
| General Caution				| General Caution      				|
| Keep Left						| Keep Left      					|
| Speed Limit(70kmph)			| Speed Limit(70kmph)      			|


 ![alt text][image11]
 
The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. In the case of Failure it was able to detect it as the second best choice. The model was able to detect with very high confidence in presence of dust,shadow and perspective distortion. No Passing image might have failed because of presence of some anomaly in the top area. I think augmenting some of low sample classes might help it further in improving the accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

On analysis of softmax probabilities I was able to notice that in case of my model failing to detect the correct class, the probability was not very high and the correct class was in top-3 classes. This means on further training and good regularization the model could improve its performance.
I implemented the visualization for the softmax probabilities for all the Images:

 ![alt text][image12]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


