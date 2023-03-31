"""Module with functions for object classification."""

import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

def create_image_path_list(path):
    """Create a list of paths to images in a directory."""
    images = [join(path, f) for f in sorted(listdir(path)) if isfile(join(path, f))]
    return images

def save_four_images(image_list, number, file_name):
    """
    Save four images from images list on one picture, 
    starting from given number of image.
    """
    image1 = np.concatenate((image_list[number], image_list[number + 1]), axis =1)
    image2 = np.concatenate((image_list[number+2], image_list[number + 3]), axis = 1)
    image = np.concatenate((image1, image2), axis = 0)
    cv2.imwrite(file_name, image)

def convert_to_grey_scale(images_path_list):
    """Load the images, convert images to grey scale and keep them in list."""
    grey_images = []
    for image_path in images_path_list:
        image_temp = cv2.imread(image_path)
        image_grey = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
        grey_images.append(image_grey)
    return grey_images

def thresholding(images_list):
    """Perform adaptive thresholding of images and keep them in list."""
    thresh_images = []
    for image in images_list:
        thresh_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                             cv2.THRESH_BINARY_INV,11,2)
        thresh_images.append(thresh_image)
    return thresh_images

def erosion(images_list, mask_size, i):
    """
    Perform i times the erosion of images 
    using a mask full of ones and keep them in list.
    """
    mask = np.ones((mask_size, mask_size), np.uint8)
    eroded_images = []
    for image in images_list:
        eroded_image = cv2.erode(image, mask, iterations = i)
        eroded_images.append(eroded_image)
    return eroded_images

def dilation(images_list, mask_size, i):
    """
    Perform i times the dilation of images 
    using a mask full of ones and keep them in list.
    """
    mask = np.ones((mask_size, mask_size), dtype = int)
    dilated_images = []
    for image in images_list:
        dilated_image = cv2.dilate(image, mask, iterations = i)
        dilated_images.append(dilated_image)
    return dilated_images

def opening(images_list, mask_size):
    """
    Perform opening of images (erosion followed by dilation)
    using a mask full of ones and keep them in list.
    """
    mask = np.ones((mask_size, mask_size), dtype = int)
    open_images = []
    for image in images_list:
        open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, mask)
        open_images.append(open_image)
    return open_images

def closing(images_list, mask_size):
    """
    Perform closing of images (dilation followed by erosion)
    using a mask full of ones and keep them in list.
    """
    mask = np.ones((mask_size, mask_size), dtype = int)
    closed_images = []
    for image in images_list:
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, mask)
        closed_images.append(closed_image)
    return closed_images

def canny_edge(images_list):
    """
    Perform edge detectionon on images using Canny's method
    and keep them in list.
    """
    edges = []
    for image in images_list:
        edge = cv2.Canny(image, 100 , 200)
        edges.append(edge)
    return edges

def find_centroid(images_list):
    """
    Find centroid of each image using cv2.moments() function
    and keep them in the array.
    """
    contours = []
    for image in images_list:
        contour, hierarchy = cv2.findContours(image, 1 , 2)
        contours.append(contour[0])
    size = (len(contours), 2)
    centroid = np.zeros(size)
    for i, value in enumerate(contours):
        M = cv2.moments(value)
        centroid[i][0] = int(M['m10']/M['m00'])
        centroid[i][1] = int(M['m01']/M['m00'])

    return centroid

def find_shape_coefficients(edges, centroid):
    """
    For each image, find the shape coefficients such as: 
    - (maximum distance from centroid)/(minimum distance from centroid)
    - (maximum distance from centroid)/(mean distance from centroid)
    - (minimum distance from centroid)/(minimum distance from centroid)
    and keep the in the array.
    """
    size = (len(centroid),3)
    distance = np.zeros(size)
    for i, image in enumerate(edges):
        distance[i][0] = image.shape[0] # min distance
        distance[i][1] = 0 # max distance
        distance[i][2] = 0 # mean distance
        countour = np.where(image == 255)
        sum = 0
        for x, y in zip(countour[0], countour[1]):
            dist = np.sqrt((x-centroid[i][0])**2 + (y-centroid[i][1])**2)
            sum +=dist
            if dist < distance[i][0]:
                distance[i][0] = dist
            if dist > distance[i][1]:
                distance[i][1] = dist
        distance[i][2] = (sum/float(len(edges[0])))

    shape_coefficients = np.zeros_like(distance)

    for i in range(len(edges)):
        if distance[i][0] != 0 and distance[i][2] != 0:
            shape_coefficients[i][0] = distance[i][1]/distance[i][0]
            shape_coefficients[i][1] = distance[i][1]/distance[i][2]
            shape_coefficients[i][2] = distance[i][0]/distance[i][2]
        else:
            shape_coefficients[i][0] = image.shape[0]
            shape_coefficients[i][1] = distance[i][1]/distance[i][2]
            shape_coefficients[i][2] = distance[i][0]/distance[i][2]

    return shape_coefficients

def identification (shape_coefficients, image_path):
    """
    Based on the shape coefficients, classify the objects and calculate
    the accuracy od the classification.
    Store result in the array, where "1" means circle and "0" means rectangle.
    """
    result = np.zeros(shape_coefficients.shape[0])
    for i in range(shape_coefficients.shape[0]):
        if (shape_coefficients[i][0] < 1.4 or (shape_coefficients[i][1]<1.51 
            and shape_coefficients[i][2] < 0.53)):
            result[i] = 1
        else:
            result[i] = 0
    acc = 0
    image_list = sorted(listdir(image_path))
    for i in range(shape_coefficients.shape[0]):
        if result[i] == 1 and image_list[i][0] == "c":
            acc +=1
        elif result[i] == 0 and image_list[i][0] == "r":
            acc +=1
        elif result[i] == 1 and image_list[i][0] == "r":
            print(f"Wrong classification of the object no. {i} (rectangle)")
        elif result[i] == 0 and image_list[i][0] == "c":
            print(f"Wrong classification of the object no. {i} (circle)")
    acc = acc/shape_coefficients.shape[0]

    return result, acc
