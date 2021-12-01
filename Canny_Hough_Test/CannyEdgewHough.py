import cv2
import numpy as np
import matplotlib.pyplot as plt

# the polar form of a lin is represented as rho = x * cos(theta) + y * sin(theta)
# rho is the perpendicular distance from  the line from the origin in pixels and theta is
# the angle, the line makes with the origin

# reading color image to grayscale
img = cv2.imread("chess.jpg", cv2.IMREAD_COLOR)

# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, apertureSize = 3)
edgesLine = cv2.HoughLines(edges, 1, np.pi/180, 200)
# draw the lines on the image
for line in edgesLine:
    # assign rho and theta value from (0,0)
    rho, theta = line[0]
    # get sin and cos values
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)

    cv2.line(img, (x1, y1), (x2,y2), (0, 0, 255), 2)

cv2.imshow('Lane Detected',img)


cv2.waitKey(0)

cv2.destroyAllWindows()