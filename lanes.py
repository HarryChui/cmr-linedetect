import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_canny_edges(image):
  # convert image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # apply gaussian blur to reduce noise in the image (using a 5x5 kernal)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # apply canny edge detection using minVal of 50 and maxVal of 150
  canny = cv2.Canny(blur, 50, 150)

  return canny

def colorFilter(img):
    # apply a color filter on the yellows and whites of the lane lines
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([18,94,140])
    upperYellow = np.array([48,255,255])
    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([255, 255, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow)
    return combinedImage

def thresholding(image):
  # apply the canny edge and the color filter together
  canny = get_canny_edges(image)
  color = colorFilter(image)

  combinedImage = cv2.bitwise_or(color, canny)

  return canny,color,combinedImage

def mask(image,verts=None):
  # mask the image with a set of vertices
  # if no vertices are provided, we use a default mask
  mask = np.zeros_like(image)
  height,width = image.shape

  if verts is None:
    verts = np.array([[[.23*width,.92*height],[.42*width,.6*height],[.58*width,.66*height],[.87*width,.92*height]]],dtype=np.int32)
    print(verts)
  cv2.fillPoly(mask,verts,255)

  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

def perspective_transform(image,verts=None):
  # perform the perspective transform on the image
    height, width = image.shape
    if verts is not None:
      src = np.array(verts,dtype=np.float32)
    else:
      src = np.array([[(width*0.1,height*0.9),
                      (width/2.3,height/1.6),
                      (width/1.7,height/1.6),
                      (width*0.9,height*0.9)]],dtype=np.float32)

    dst = np.array([[(0.25*width,height),
                    (0.25*width,0),
                    (0.75*width,0),
                    (0.75*width,height)]],dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src,dst)
    top_down_image = cv2.warpPerspective(image, transform_matrix, (width,height), flags = cv2.INTER_LINEAR)
    return top_down_image, transform_matrix

def inverse_perspective(image,verts=None):
  # create the inverse perspective change
    height = image.shape[0]
    width = image.shape[1]

    if verts is not None:
      src = np.array(verts,dtype=np.float32)
    else:
      dst = np.array([[(width*0.1,height*0.9),
                      (width/2.3,height/1.6),
                      (width/1.7,height/1.6),
                      (width*0.9,height*0.9)]],dtype=np.float32)

    src = np.array([[(0.25*width,height),
                    (0.25*width,0),
                    (0.75*width,0),
                    (0.75*width,height)]],dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src,dst)
    top_down_image = cv2.warpPerspective(image, transform_matrix, (width,height), flags = cv2.INTER_LINEAR)
    return top_down_image, transform_matrix

def get_hist(img): # return the histogram of the warped image based on white pixels in the x axis
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def sliding_window(image, nwindows=15, margin=50, minpix=1, draw_windows=True):
  # perform the sliding window algorithm
  finalimage = np.dstack((image,image,image)) * 255

  histogram = get_hist(image)
  # find the peaks of the left and right halves indicating where to start sliding window
  midpoint = int(histogram.shape[0]/2)
  leftx_start = np.argmax(histogram[:midpoint])
  rightx_start = np.argmax(histogram[midpoint:]) + midpoint

  # set height of the windows
  window_h = int(image.shape[0] / nwindows)
  nonzero = image.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])

  # current position
  leftx_curr = leftx_start
  rightx_curr = rightx_start

  # lists to store the indices of the pixels
  left_lane_inds = []
  right_lane_inds = []

  for window in range(nwindows):
    # get window boundaries
    win_ylow = image.shape[0] - (window+1) * window_h
    win_yhigh = image.shape[0] - window * window_h
    win_xbotleft = leftx_curr - margin
    win_xtopleft = leftx_curr + margin
    win_xbotright = rightx_curr - margin
    win_xtopright = rightx_curr + margin

    #draw windows
    if draw_windows:
      cv2.rectangle(finalimage, (win_xbotleft,win_ylow), (win_xtopleft,win_yhigh),
                    (100,255,255),1)
      cv2.rectangle(finalimage, (win_xbotright,win_ylow), (win_xtopright,win_yhigh),
                    (100,255,255),1)
    
    # find the nonzero pixels in the window
    left_inds = ((nonzeroy >= win_ylow) & (nonzeroy < win_yhigh) & 
                  (nonzerox >= win_xbotleft) & (nonzerox < win_xtopleft)).nonzero()[0]
    right_inds = ((nonzeroy >= win_ylow) & (nonzeroy < win_yhigh) & 
                  (nonzerox >= win_xbotright) & (nonzerox < win_xtopright)).nonzero()[0]

    # add these indices to the lists
    right_lane_inds.append(right_inds)
    left_lane_inds.append(left_inds)

    if len(right_inds) > minpix:
      rightx_curr = int(np.mean(nonzerox[right_inds]))
    if len(left_inds) > minpix:
      leftx_curr = int(np.mean(nonzerox[left_inds]))

  # concatenate the indices together
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # get the left and right lane pixels
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  # fit a second order polynomial to the left and right
  if leftx.size and rightx.size:

    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    finalimage[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    finalimage[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return finalimage,(left_fitx,right_fitx),ploty

def draw_lane(image, left_fit, right_fit):
  # draw the lane on the original image
  ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
  color_img = np.zeros_like(image)

  # get the points we need to fill
  left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
  right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
  points = np.hstack((left, right))

  cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))

  # put the poly back on the origina; image with the inverse perspective change
  inv_perspective,inv_matrix = inverse_perspective(color_img)
  inv_perspective = cv2.addWeighted(image, 0.5, inv_perspective, 0.7, 0)
  return inv_perspective