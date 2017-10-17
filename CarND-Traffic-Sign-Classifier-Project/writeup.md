#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/classes_distribution.png "Visualization"
[image1a]: ./writeup/classes_examples.png "Examples"
[image2]: ./writeup/grayscale.png "Grayscaling"
[image4]: ./writeup/internet_images.png "New images from internet"
[image5]: ./writeup/features.png "Feature map of Conv1 Layer"


## Rubric Points

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/diegopdomingos/nanodegree/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) and to [html version](https://github.com/diegopdomingos/nanodegree/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 400
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed along the classes. We can see here that there are some classes (like 1, 2, 3, 4, etc) that have a lot of examples (>1000). Although, there are classes of images that have few examples, like the the 0, which has less than 250 examples.

![alt text][image1]


Then, I printed the images to check which sign belongs to which class.

![alt text][image1a]

###Design and Test a Model Architecture

####Image preprocessing

As a first step, I decided to convert the images to grayscale because its known that it doesn't make much difference in terms of performance in our classifier. However, having 3 channels instead can consumes more CPU/GPU in our training step.

Another thing that I did was to normalize the data with the suggested (pixel-128)/128 equation. This helps our traning step as well to perform faster and better.

Here is an example of a traffic sign image before and after grayscaling.

![alt_text][image4]
![alt text][image2]


I failed trying to put some augmented images due to lack of knowledge using tensorflow. I did try some tensors like tf.image.random_contrast, tf.image.random_hue, etc. This not worked as expected since I'm not familiar with this functions and the tests didn't improved with these new images. However, because the orientation matters, there are some extra work that we need to do: once if you flip an sign, this can change the class of this image and it must be managed to not create bad data on our set.



####2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale Image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| Kernel size: 2x2, Strides: 2x2 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x64    									|
| Fully connected		| Input: 2304, Output: 900        									|
| RELU			| |
| MaxPooling		| Kernel size: 2x2 , Strides: 2x2|
| Fully connected	| Input: 900, output: 350 |
| RELU			| |
| Fully connected	| Input: 350, output: 43|
| Softmax				|         									|
 

To train the model, I used the following:

* Batch size of 128 images
* 5 epochs
* Learning rate of 0.001

The Optimizer choosen was Adam.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.947
* test set accuracy of 0.931

The first archutecture used was LeNet. I did some trainings but the validation accuracy were about 92%. Then, I did some tests rising the number of filters in the convolution steps. One thing that I noticied is that the training starts with more accuracy (0.88 instead 0.77), although it starts to saturate earlier. 

After that, I got a CNN with two convolutional layers with 64 filters each and 3x3 filter. Because it generates a big flatten layer in the dense step, I decided to add dropout layers between the fully connected layer 1 and 2 and another one between 2 and 3 with chance of drop of 50%. This dropout layer can help to prevent overfitting, since in each training step, some alleatory connections between neurons are choosen and set to 0. This feature can force the net to not depend on a single neuron and improves the overall classification of the net. In fact, the discrepancy of the training and validation accuracy decreased.

The number of ephocs was chosen by checking the history of large trainings (ephocs = 90, 100, etc) and I noticied that it stops increasing very fast.

The convolution layer works well in this problem because there are some caracteristics in the traffic signs images that can help a lot the dense net to make the decisions: sign shape (triangle, square, circle), color, size, etc.

The result in the training, validation and test sets provides evidence that the model is working well, since they are close to the 100% and where are no enough discrepances between them.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The third image seems to be difficult to classify since it has small sign if we compare with the training dataset. The last images show a signs with rotation, which can increase the error in the classification.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| 50 km/h     			| 50 km/h										|
| 70 km/h					| 50 km/h											|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection					 				|
| Bumpy Road			| Bumpy Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the test set but can be explained since there are some noises in the images downloaded over the internet that maybe is not in the dataset. One thing that we can try is adding some random noises (rotation, shifting, resizing, contrast, etc) to the training data set. This augmentated dataset can increase the chance of sucess in these new images. However, this is close to the result obtained in our test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

All predictions occured with 100% of probability. This is very strange and need to be investigated further, since 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   									| 
| 1.0     				| 50 km/h 										|
| 1.0					| 50 km/h											|
| 1.0	      			| Right-of-way at the  next intersection					 				|
| 1.0				    | Bumpy Road      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The last exercise was to visualize the features of the convolutional filters. 

![alt_text][image5]

There are some interesting features that we can see in the image above. For example, the feature map 58 is activated showing the circle and the arrow. In the feature map 5 we can see only the arrow. These features can show how they can, for example, approximately detect some edges or a somewhat blur transformation like the feature 32.
