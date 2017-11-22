# CarND-Advanved-Lane-Lines
# Author: Diego Domingos
# Description: this file is the main file of my project.
#              To  make it easier for the review, all code and functions
#              are located in this single file

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

THRESHOLD_MSE = 1000
MAX_X_DISTANCE = 0.5
MAX_MSE_PARALLEL = 0.03
MAX_FAILURES = 3
NUMBER_OF_RECENT_XFITTED = 3
MAX_LANE_WIDTH = 4.2
MIN_LANE_WIDTH = 3.6

mtx, dist = None, None
visualize = False # Set True to see the transformation images

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.fits = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/635 # meters per pixel in x dimension
        # TODO: drop out this hardcoded guy
        self.camera_width = 640 + 5  # Center of image + correction factor
        # Number of failures
        self.failures = MAX_FAILURES+1
        # If we needed to rescan with sliding windows in the
        # previous loop
        self.rescanned = True


    def calc_radius_of_curvature(self):
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*np.max(self.ally)*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad

    def force_rescan(self):
        # Check if we need to rescan
        # using sliding windows
        if self.failures > MAX_FAILURES:
            self.failures = 0
            #self.recent_xfitted = []
            #self.bestx = self.allx
            self.rescanned = True
            return True
        self.rescanned = False
        return False

    def distance_from_camera(self):
        # Calculate the distance of the line
        # to the camera
        return self.xm_per_pix*np.abs(self.camera_width-self.allx[-1])

    def check_parallelism(self, fit):
        # Check for parallelism by comparing
        # the first two coefficients
        mse = ((np.array(self.current_fit[0:2])-np.array(fit[0:2]))**2).mean(axis=0)
        if mse > MAX_MSE_PARALLEL:
            return False
        return True

    def reset_cache(self):
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.fits = []

    def sanity_check(self, other_line, avoid_mse = False):
        ''' This functions checks for the sanity of
            the found lines '''

        self.detected = False

        # Calculate MSE for fitx
        if type(self.bestx) != type(None) and not avoid_mse:
            mse = ((np.array(self.allx)-np.array(self.bestx))**2).mean(axis=0)
            if mse > THRESHOLD_MSE:
                print("Reprovado por MSE")
                self.failures += 1
                return False

        # If the camera is too far from the left line comparing,
        # with the previous image, something is wrong
        if type(self.line_base_pos) != type(None):
            if self.distance_from_camera() > self.line_base_pos + MAX_X_DISTANCE:
                print("Reprovado por posição")
                self.failures += 1
                return False
        
        lane_width = self.distance_from_camera() + other_line.distance_from_camera()
        if (lane_width > MAX_LANE_WIDTH) or (lane_width < MIN_LANE_WIDTH):
            print("Reprovado pelo tamanho da pista")
            self.failures += 1
            return False

        if not self.check_parallelism(other_line.current_fit):
            print("Reprovado por paralelismo")
            self.failures += 1
            return False

        self.fits.append(np.array(self.current_fit))
        self.best_fit = np.array(self.fits).mean(axis=0)
        self.radius_of_curvature = self.calc_radius_of_curvature()
        self.line_base_pos = self.distance_from_camera()
        self.detected = True
        self.failures = 0

        self.recent_xfitted.append(self.allx)
        self.bestx = np.array(self.recent_xfitted).mean(axis=0)

        return True

    def update(self, fit, fitx, ploty):
        ''' Update our line with new data '''
        self.current_fit = fit
        self.allx = fitx
        self.ally = ploty

        # If we are in the first loop
        # our bestx is our currently fitx
        if type(self.bestx) == type(None):
            self.bestx = fitx

        if type(self.best_fit) == type(None): 
            self.best_fit = fit

        # Slide the most recent XFITTED
        if len(self.recent_xfitted) > NUMBER_OF_RECENT_XFITTED:
            self.recent_xfitted = self.recent_xfitted[1:]

        # Update our data
        #self.recent_xfitted.append(self.allx)
        #self.bestx = np.array(self.recent_xfitted).mean(axis=0)


# Define our lines
left_line = Line()
right_line = Line()

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(np.sqrt(sobelx**2 + sobely**2))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan2(abs_sobely,abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >=thresh[0]) & (dir_grad <= thresh[1])] = 1
    return dir_binary

def camera_calibration(nx,ny,path, visualize = False):

    ''' This function is used to find the corners in
        a image for calibration '''

    objpoints = []
    imgpoints = []

    # Prepare objects points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    images = glob.glob(path)

    for fname in images:

        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, \
                                                imgpoints, gray.shape[::-1], \
                                                None, None)

            dst = cv2.undistort(img, mtx, dist, None, mtx)

            if visualize:
                f, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
                f.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=10)
                ax2.imshow(dst)
                ax2.set_title('Undistorted Image', fontsize=10)
                plt.show()
        else:
            print("No corners found")

    return mtx, dist

def corners_unwarp(img, nx, ny, mtx, dist):
    # Undistort using mtx and dist
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If corners found:
    if ret == True:
            # draw corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            
            # define 4 source points src
            src = corners[[0,7,40,47]].reshape(4,2)
            # define 4 destination points dst
            dst = np.float32([[110,110],[1150,110],[110,850],[1150,850]])
            # get M, the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # warp image to a top-down view
            warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M

def color_and_gradient(img, grad_threshold=(0,255), mag_threshold=(0,255), dir_threshold=(0,255), s_threshold=(170,255), ksize=3, visualize=False):

    # Compute our binary image by applying Sobel and color transformations

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get Sobel, Magnitude and Direction images with thresholds
    gradx = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=grad_threshold)
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=grad_threshold)
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=mag_threshold)
    dir_binary = dir_thresh(gray, sobel_kernel=ksize, thresh=dir_threshold)

    # Combine the images
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Get our S channel from HLS transformation
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    s_binary = np.zeros_like(S)
    s_binary[(S >= s_threshold[0]) & (S <= s_threshold[1])] = 1

    # Combine with previous results
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    
    # If we want to visualize the result
    if visualize:
        plt.imshow(combined_binary, cmap="gray")
        plt.savefig("./examples/binary.png")
        plt.show()

    return combined_binary

def warp_image(img, mtx, dist, src, dst):
    # Warp the image

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def find_peaks_initial(img, nwindows = 9, margin = 100, visualize = False):
    # Find the initial line using sliding window

    # Get the histogram of the half the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    if visualize:
        plt.plot(histogram)
        plt.show()

    out_img = np.dstack((img, img, img))#*255
    out_img = out_img.astype(np.uint8)

    # Find the bases of our left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    binary_warped = img
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set minimum number of pixels found to recenter window
    #minpix = 50 
    minpix = 25

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if visualize:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return out_img, left_fit, right_fit, left_fitx, right_fitx, ploty



def find_peaks(binary_warped, left_fit, right_fit, margin=100, visualize=False):
    # Find the lines by looking into the margins of
    # the existing lines

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if visualize:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return result, left_fit, right_fit, left_fitx, right_fitx, ploty


def drawing(undist, warped, Minv, left_fitx, right_fitx, ploty):

    image = undist

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros([image.shape[0],image.shape[1],3], dtype=np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 0.7, newwarp, 0.3, 0)

    if visualize:
        plt.imshow(result)
        plt.show()

    return result

def visualize_perspective(img, pts):
    # Function to help visualize
    # the images with perspective lines
    vis_img = np.copy(img)
    cv2.line(vis_img, tuple(pts[0]), tuple(pts[1]), (255,0,0), 2)
    cv2.line(vis_img, tuple(pts[1]), tuple(pts[2]), (255,0,0), 2)
    cv2.line(vis_img, tuple(pts[2]), tuple(pts[3]), (255,0,0), 2)
    cv2.line(vis_img, tuple(pts[3]), tuple(pts[0]), (255,0,0), 2)
    plt.imshow(vis_img, cmap="gray")
    plt.show()

def process_image(img):
    # The function that will run the pipeline
    # and will draw our result image
    global left_line
    global right_line

    # Undistort image using the found mtx and dist
    # values
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # Perspective Transform
    src = np.float32([[150, img.shape[0]],[590, 450],[690, 450],[1140, img.shape[0]]])
    dst = np.float32([[300, img.shape[0]],[300, 0],[1000, 0],[1000, img.shape[0]]])
    
    if visualize:
        visualize_perspective(img, src)

    warped, M, Minv = warp_image(img, mtx, dist, src, dst)

    if visualize:
        visualize_perspective(warped, dst)

    warped = color_and_gradient(warped, grad_threshold=(30,60), mag_threshold=(230,240), dir_threshold=(0.9,1.7), s_threshold=(135,255), ksize=7, visualize=visualize)

    # Finding the lanes using sliding window if we need it
    if left_line.force_rescan() or right_line.force_rescan():
        print("Buscando com sliding window")
        warped, left_fit, right_fit, left_fitx, right_fitx, ploty = find_peaks_initial(warped, visualize=visualize,margin=60)
        # Update our lines and do the sanity check
        left_line.update(left_fit, left_fitx, ploty)
        right_line.update(right_fit, right_fitx, ploty)
        if left_line.sanity_check(right_line, avoid_mse = True) and right_line.sanity_check(left_line, avoid_mse = True):
            left_line.reset_cache()
            right_line.reset_cache()
            left_line.update(left_fit, left_fitx, ploty)
            right_line.update(right_fit, right_fitx, ploty)
    else:
        #left_fit = left_line.current_fit
        #right_fit = right_line.current_fit
        left_fit = left_line.best_fit
        right_fit = right_line.best_fit
        #print(left_fit)
        #print(right_fit)
        warped, left_fit, right_fit, left_fitx, right_fitx, ploty = find_peaks(warped, left_fit, right_fit,margin=60,visualize=visualize)

        # Update our lines and do the sanity check
        left_line.update(left_fit, left_fitx, ploty)
        right_line.update(right_fit, right_fitx, ploty)

        left_line.sanity_check(right_line)
        right_line.sanity_check(left_line)

    # Draw the found lanes
    if type(left_line.bestx) != type(None) and type(right_line.bestx) != type(None):
        img = drawing(img, warped, Minv, left_line.bestx, right_line.bestx, ploty)

    return img


def main():

    global mtx
    global dist
    global visualize

    visualize = False

    # Camera calibration to undistort images
    mtx, dist = camera_calibration(9,6,"camera_cal/calibration*.jpg")

    # Load image
    #for i in range(1,2):
        #img = plt.imread("test_images/straight_lines%s.jpg" % i)
        #img = process_image(img)
        #plt.imshow(img)
        #plt.show()

    clip1 = VideoFileClip("project_video.mp4").subclip(0,45)
    clip = clip1.fl_image(process_image)
    clip.write_videofile("test6.mp4")

        

if __name__ == '__main__':

    main()
