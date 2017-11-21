## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 262 through 294 (colors_and_gradient()) in `main.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 296 through 304 in the file `main.py`.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[150, img.shape[0]],
    [590, 450],
    [687, 450],
    [1140, img.shape[0]]])
dst = np.float32(
    [[300, img.shape[0]],
    [300, 0],
    [1000, 0],
    [1000, img.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 150, 720      | 300, 720        | 
| 590, 450      | 300, 0      |
| 687, 450     | 1000, 0      |
| 1140, 720      | 1000, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two functions which are responsible to find the lines, find_peaks_initial() and find_peaks(). The first function takes the histogram of the half of the binary image (vertical axis) and find the peaks where we have more white pixels. This corresponds to our lines of the lanes. After that, we start the sliding windows function, which are responsible to take a rectangle from bottom to the top and try to find where our white pixels are. For this, we start from our peak found in histogram, then, we slide our window by some amount of pixels to the top (image height / number of windows) and, inside a margin range, we try to find again where our white pixels are.
Always that we find our pixels, we put it on a list of found x values.  

Then I did a polyfit() function that fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 60 through 65 in my code in `main.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 470 through 493 in my code in `main.py` in the function `drawing()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)
#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The first problem I faced is regarding the thresholds. It takes some time to define which is the best values, once we have a lot of thresholds to define. I figure out that is very interesting to check directly on the result images to see if our thresholds are correct.

Another thing that I faced is regarding the bridges lanes. These lines are very difficult to track once they have a color that is very close to the entire lane in gray color. One thing that makes it possible to regognize it is transforming the color to the HLS space and then extract the S channel from it - this did the job a lot easier.

Then, I found a problem with my warp function. Actually the problem was with the src and dst vectors. Because my transformation was high bias for the right side of the image, so I was missing some points of the right line and then the fit was doing terrible until I find this issue.

Because the road has a lot of noise, there are sometimes that we miss the lines and we get some strange behaviors of our line detection. So, I decided to implement the sanity check - then I was able to check if we need to do the initial detection with sliding windows or not. Basically, it has the following checks:

* The MSE from current and the previous line is small enough?
* Our left and right lines are parallel?
* The lane has about 3.7 meters of width? (For this, we need to implement a function that measures it using a pixel to meter conversion)
* Our camera distance from the lines make senses, i.e, they are not a lot closer or far from the previous measure?

If we get enough failures in this sanity (3 times), we start by searching the lines with sliding window again.

There are some improvement that we can do to make it more robust, for example, tuning the parameters to get yet better results with our binary images. Yet, we can 
  
