import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def canny_detection(image):
    """
    Apply canny function to identify edges in the image
    Which in process applies the Gaussian Blur
    Traces and outlines the edges that correspond or above the highest intensity
    Gradients that exceed the high threshold are traced as bright pixels
    Small gradients are not traced at all and are black and below the lower threshold
    """"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny


def region_of_interest(image):
    """Define a triangular polygon as ROF and return a mask"""
    height = image.shape[0]
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])  # Triangle polygon
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)  # 255 color of polygon
    return mask


def masked_image(canny, mask):
    """Frame Masking the ROF"""
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def display_lines(image, lines):
    """Display lines on the ROF"""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a line
            # cv2.line(image, coordinates1, coordinates2, RGBCOLOR, line-thickness)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_points(image, line):
    """Specify point coordinates of the line (y = mx + c)"""
    try:
        slope, intercept = line
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([[x1, y1, x2, y2]])
 

def average_slope_intercept(image, lines):
    """Optimize line display and have a smooth trace"""
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image (Negative slope)
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def main():
    video = "test2.mp4"

    # set a video capture object
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        _, frame = cap.read()  # bool, frame/image
        # if frame is not None:
        canny_image = canny_detection(frame)
        mask = region_of_interest(canny_image)
        mask_image = masked_image(canny_image, mask)
        lines = cv2.HoughLinesP(mask_image, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result_video", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
