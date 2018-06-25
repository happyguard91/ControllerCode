import matplotlib.pyplot as plt
import numpy as np
import cv2
import lane_detection3

image = cv2.imread('/Users/edwardkim/Downloads/webotsEx-master/controllers/foo/webots_pics/yo70.png')
# cv2.imshow("actual", image)
# cv2.imshow("cropped", image[0:200, 0:100])
# cv2.waitKey(0)

img = cv2.resize(image, (960, 540))
gray_image = lane_detection3.discard_colors(img)
image = lane_detection3.detect_edges(gray_image, low_threshold=50, high_threshold=150)

# plt.imshow(image)
# plt.pause(5)

xsize = image.shape[1]
ysize = image.shape[0]
print(xsize, ysize)
dx1 = int(0 * xsize)
dx2 = int(0.4 * xsize)
dy = int(0.3 * ysize)
# calculate vertices for region of interest
vertices = np.array([[(0, 540), (350, 300), (600, 300), (800, 540)]], dtype=np.int32)
print(vertices)

image = lane_detection3.region_of_interest(image, vertices)

# plt.imshow(image)
# plt.pause(10)

rho = 0.8
theta = np.pi/180
threshold = 25
min_line_len = 50
max_line_gap = 200

lines = lane_detection3.hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
# print("lines : ")
# print(lines)
right_lines, left_lines = lane_detection3.separate_lines(lines)
# print(right_lines)
# print(left_lines)

if right_lines and left_lines:
    right = lane_detection3.reject_outliers(right_lines,  cutoff=(0.75, 0.95))
    left = lane_detection3.reject_outliers(left_lines, cutoff=(-0.65, -0.45))

# print("right and left:")
# print(right)
# print(left)

x, y, m, c = lane_detection3.lines_linreg(right)
# This variable represents the top-most point in the image where we can reasonable draw a line to.
min_y = np.min(y)
# Calculate the top point using the slopes and intercepts we got from linear regression.
top_point = np.array([(min_y - c) / m, min_y], dtype=int)

# Repeat this process to find the bottom left point.
max_y = np.max(y)
bot_point = np.array([(max_y - c) / m, max_y], dtype=int)

x1e, y1e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
x2e, y2e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point

# print(x1e, y1e)
# print(x2e, y2e)

# return the line.
right_line = np.array([[x1e,y1e,x2e,y2e]]) ####FIX  SOMETHING HERE1!!!!!!!

x, y, m, c = lane_detection3.lines_linreg(left)
# This variable represents the top-most point in the image where we can reasonable draw a line to.
min_y = np.min(y)
# Calculate the top point using the slopes and intercepts we got from linear regression.
top_point = np.array([(min_y - c) / m, min_y], dtype=int)

# Repeat this process to find the bottom left point.
max_y = np.max(y)
bot_point = np.array([(max_y - c) / m, max_y], dtype=int)

x1e, y1e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
x2e, y2e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point

# print(x1e, y1e)
# print(x2e, y2e)

left_line = np.array([[x1e,y1e,x2e,y2e]]) ####FIX  SOMETHING HERE1!!!!!!!
lines = np.append(np.array([right_line], dtype=np.int32), np.array([left_line], dtype=np.int32), axis=0)
# print(lines)
# print(image.shape)

line_image = np.copy(img*0)
lane_detection3.draw_lines(line_image, lines, thickness=10)
# plt.imshow(line_image)
# plt.pause(5)
# lane_detection3.draw_lines(line_image, np.array([left_line], dtype=np.int32), thickness=3)

line_image = lane_detection3.region_of_interest(line_image, vertices)
# plt.imshow(line_image)
# plt.pause(5)

final_image = lane_detection3.weighted_image(line_image, img)

plt.imshow(final_image)
# plt.imshow(img)
plt.pause(10)

