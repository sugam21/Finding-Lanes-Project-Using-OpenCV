import cv2 # For image processiing
import numpy as np # For handling arrays
import matplotlib.pyplot as plt #For displaying image 

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        param = np.polyfit((x1,x2),(y1,y2),1) # Fits the line in Cartesian co-ordinate system i.e y = mx + c
        
        slope = param[0]
        intercept = param[1]
        
        # If the line has negative slope then append it in one list otherwise append it in another list
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    print(left_fit_average,right_fit_average)
    # Converting the slope and intercept into the co-ordinate form
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)

    return np.array([left_line,right_line])




def canny(image):
    # Converting the colored image into gray scaled image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Applying the Gaussian blur
    blur = cv2.GaussianBlur(gray,(5,5),0) # Blur kernel is of size 5 x 5 and deviation is 0
    # Applying canny method to identify edges
    canny = cv2.Canny(blur,50,150) # If gradient is greater the high_threshold then it is accepted as edge
    # If gradient is lower than the lower threshold then it is rejected 
    # Displaying the image
    # The low to high threshold should be in the ratio of 1:2 or 1:3
    return canny






## Function to display lines in the actual image
def display_lines(image,lines):
    # Declaring the image of zeros of the same shape as original image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4) # Unpacking the array into its 4 parts
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image 



# Identifying the lane lines in the image 
# What we want to do is to mask the image with our filter to show only the requered area i.e the lane
# So creating a masking a triangle to filter the lane area
def region_of_interest(image):
    height = image.shape[0]
    polygons =   np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image) # same number of rows and column as image but with zeros
    cv2.fillPoly(mask,polygons,255) # this will fill the mask with the traingle with colour white
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
 
# Reading the image 
img = cv2.imread('test_image.jpg')

# Copying the image in numpy array
lane_image = np.copy(img)

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

#Creating hough lines 
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)#image where you want to put lines,rho(precision of 2 pixels),theta(precision of 1 pixel),threshold(min number of intersections in a bin in hough space)

# Averaging the multiple lines to create a single line in the image 
averaged_line = average_slope_intercept(lane_image,lines)


line_image = display_lines(lane_image,averaged_line)


## Blending the lines into the original image
combined_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

cv2.imshow('Image',combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cap = cv2.VideoCapture("test2.mp4")
# while (cap.isOpened()):
#     ret,frame = cap.read()    
#     if ret==True:
#         canny_image = canny(frame)
#         cropped_image = region_of_interest(canny_image)

#         #Creating hough lines 
#         lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)#image where you want to put lines,rho(precision of 2 pixels),theta(precision of 1 pixel),threshold(min number of intersections in a bin in hough space)

#         # Averaging the multiple lines to create a single line in the image 
#         averaged_line = average_slope_intercept(frame,lines)


#         line_image = display_lines(frame,averaged_line)


#         ## Blending the lines into the original image
#         combined_image = cv2.addWeighted(frame,0.8,line_image,1,1)
#         cv2.imshow('Video',combined_image)
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             break
#     else:
#         break
# cap.release() # Closing the video file
# cv2.destroyAllWindows()