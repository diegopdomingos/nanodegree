#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2017_10_31_21_29_37_313.jpg "Center image inside car"
[image3]: ./examples/center_2017_11_04_16_33_21_586.jpg "Recovery Image"
[image4]: ./examples/center_2017_11_05_12_32_34_861.jpg "Recovery Image"
[image5]: ./examples/center_2017_11_05_12_32_37_115.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 3 and 48 (model.py lines 101-107) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 99). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 26,126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Yet, some training inside the bridge was intense to reduce deviation inside it - my model tries to keep the car exactly in the center of the bridge lane.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train it with enough input images, check the MSE and, once it reaches a ood value (low value), check for the driving behavior of the car in the track 1 of the simulator.

My first step was to use a convolution neural network model similar to the [NVIDIA Team](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) I thought this model might be appropriate because it was used for self driving cars and it has many convolutional layers, which might can extract some interesting features to our dense network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a factor of 0.2. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, first I added more images (ran the simulator again). modified the model so that 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like in the bridge - the car was very unstable and, to improve the driving behavior in these case, I drove in the such manner that showing to the model what is expected when the car is near the left or the right of the lane, i.e driving to the center again. After that, I found some problemas near the dirt track but it improves a little doing more training around it. I found that is very difficult to the Neural Network see it. One improve that I think should perform better is trying to change the contrast and other colors features of the input image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

- Lambda Layer to normalize the image
- Cropping to take just the lane instead trees and the sky, for example.
- Convolution with (5,5) kernel and 3 features (relu as activation and MaxPooling, 2x2, for nonlinearity)
- Convolution with (5,5) kernel and 24 features (relu as activation and MaxPooling, 2x2, for nonlinearity)
- Convolution with (5,5) kernel and 36 features (relu as activation and MaxPooling, 2x2, for nonlinearity)
- Convolution with (3,3) kernel and 48 features (relu as activation and MaxPooling, 2x2, for nonlinearity)
- Flatten layer
- Dense with 1164 neurons
- Dense with 100 neurons
- Dense with 50 neurons
- Dense with 10 neurons
- Dense with 1 neurons (output)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it is going to exit from the lane. These images show what a recovery looks like starting from right to the center :

![alt text][image3]

![alt text][image4]

![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would improves the training. 

After the collection process, I had 131554 number of data points. I then preprocessed this data by normalizing it (dividing by 255 and subtracting by 0.5). Yet, I cropped the image using the lambda layer so the image being used as input is just the lane and not the trees or sky.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the loss of validation that, which as not improving after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.


