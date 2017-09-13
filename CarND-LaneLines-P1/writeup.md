# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied the Gaussian Blur with kernel_size
equal to 7. After that, I applied the canny function with low_threshold as 80 and the high_threshold as 240. One thing that I
realized is that for better definitions (i.e, when the contrast with the gray road and the white lines is high), bigger low_threshold
works very well. Although, when this definitions is low (as in the challenge), a bigger low_threshold can miss some lines, so we need to
figure out a better value to it. Then, I defined my vertices for my region of interest (ROI) definition. After that, I used the rough_lines
function with threshold 30, min_line_len as 5 and max_line_gap as 10. Finally, I called the weighted_img function to get my lines and
the original image together.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by finding the slope of each line
detected by cv2.HoughLinesP(). Negative slopes are in the left side and the positive ones are in the right side of the lane. Because
there are some horizontal lines that are detected, we need to filter them. So, I setup my filter to discard slopes greater than -0.5
or less than 0.5, for left and right sides respectively. Then, my algorithm evaluates the x position of the line in the end of the
image (x=xb,y=image.shape[0]) and in the top of the ROI (x=xt,y = 330). These results are stored in a matrix such that each line is
defined as [xb,xt]. Finally, my algorithm adds a line with the mean of each column (of xb's and xt's) and plot it for the left and the
right line.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when we are in a road with not so well defined lines in the road.
For example, in the challenge, my algorithms failed :(.

Another potential shortcoming would be a noise line in the middle of the ROI.

Another potential is for a more curvy road.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve the contrast in the pipeline, so I believe the lines in the challenge video would
be more easily detected. These lines have a grayscale near the road color, so it is very difficult to extract the lines there.

Another potential improvement could be to better tuning the parameters to filter more the noise in the road, avoiding some
wrong lines. Because we are working with the mean of detected lines, sometimes we just have little noises in the solid line.
Although, I think a more stable line is necessary to a driveless car.
