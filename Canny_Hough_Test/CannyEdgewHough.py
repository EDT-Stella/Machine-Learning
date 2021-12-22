import cv2
import numpy as np

# helper function for region of interest
def region_of_interest(image, polygons):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# helper function to apply the lines to the cropped image
def display_lines(image, lines):
    makeLineImage = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(makeLineImage, (x1, y1), (x2, y2), (0, 255, 0), 7)
    return makeLineImage

# reading image
img = cv2.imread("road2T.jpg")
cv2.imshow("Original Image", img)
# get the height and weight of
height = img.shape[0]
width = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert image to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale Image", gray)
# using the Canny filter
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny Image", edges)
# set the borders for the region of interest
# Note: This may vary for image with types of roads
polygons = [(0, height), (width/2, height/3), (width, height)]

# get the region of interest
croppedImage = region_of_interest(edges, np.array([polygons], np.int32))
cv2.imshow("Cropped Image", croppedImage)
# use probabilistic Hough Lines for the cropped image with a threshold of 150
edgesLine = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 150, np.array([]), 40, 5)

# display the image with the lines
line_image = display_lines(img, edgesLine)

# combines the image with the lines
combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
# draw the lines on the image

cv2.imshow("Edge Detected", combo_image)

cv2.waitKey(0)

cv2.destroyAllWindows()