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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in the `main.py` file. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found that, using the non-linear SVC, the `YUV` color space worked very well. Then, I tried different values of `pix_per_cell` and `cell_per_block` and realized that, if you set these values very high, we can miss some important windows that currenctly has a car. Otherwise, setting these
values for a very low value, the process of searching the window can be very low.

In the final model, I tried to balance the accuracy of searching window function with the performance of the algorithm.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a non-linear SVM using the `train_model` function (lines 406 to 472). The steps consists in:

1. Load the vehicles and non-vehicles data set
2. Extract the features (hog, color and histogram)
3. Create the StandardScaler to normalize our features vector
4. Split our dataset into train and validation data sets (factor 0.2 and shuffle on)
5. Fit our SVC model and save it using pickle

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The first step was to analyze how is the size of the cars in our video images. I realized that our car can be about 64x64 to 200x200 pixels in size (approximately). So, I first started with scale equal to 1 (i.e, a 64x64 window). Then, I started to play with different scales sizes until scale equal to 3. I noticied that we have a good question here: with lower scales, our image are bigger and we have a lot of loops to slide our windows. Although, if we set our `cells_per_step` too high, we can miss some important windows. However, if we set this value too low, we need a lot of time until see this function ends.

So, by trial and error (and using the experience along the runs) I tried to balance our accuracy and performance.

Here are the result of my sliding windows function:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. The number of past positions that will be saved is stored in the `heatmap_history` variable.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are some situations that my pipeline can fail. For example, depending on the angle of the road, my pipeline can't find cars upper than about the half of the image (my range of y positions is between 400 - end of the picture). All cars that will be upper of this line will be "invisible" to my pipeline.

When two cars are very close to each other, my pipeline draws a big box around them. I imagine that one possible improvement can be tracking each car and then we can separate them while we track the position. Another thing is, use some positions and the previous size of the lane to estimate the maximum size of each car, then splitting the big box.

Another problem is the performance - although we didn't implemented it, the algorithm is capable to be parallelised in the search windows step. So, this can be a good improvement. 

