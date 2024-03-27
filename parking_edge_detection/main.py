#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import math
import struct
from datetime import datetime
import glob



def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, one_c):
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def calculate_f1_score(true_answers, my_answers):
    true_positives = sum([1 for t, m in zip(true_answers, my_answers) if t == 1 and m == 1 or t == 0 and m == 0])
    false_positives = sum([1 for t, m in zip(true_answers, my_answers) if t == 0 and m == 1])
    false_negatives = sum([1 for t, m in zip(true_answers, my_answers) if t == 1 and m == 0])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def min_val(img, tmp) -> bool:
    screen = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
    
    template_out = cv.matchTemplate(screen, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(template_out)
    
    # print("Min val: ", min_val)
    return min_val

def light_slot(min_val) -> bool:
    return min_val <= 0.36

def shadow_slot(min_val) -> bool:
    return min_val <= 0.32

def night_slot(min_val) -> bool:
    return min_val <= 0.5

def main(argv):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    
    result = []

    for file_path in glob.glob("test_images_zao/*.txt"):
        file_data = []
        with open(file_path, 'r') as file:
            data = [int(x) for x in file.read().split()]
            file_data.extend(data)
        result.append(file_data)
    # print(result)

    


    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    # print(pkm_coordinates)
    # print("********************************************************")   
    # print(test_images)
    cv.namedWindow("one_place", cv.WINDOW_NORMAL)
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.namedWindow("image2", cv.WINDOW_NORMAL)
    f1_scores = []
    i = 0
    for img_name in test_images:
        
        my_result = []

        image = cv.imread(img_name,1)
        image2 = cv.imread(img_name,1)

        image3 = image2.copy()
        
        image2 = cv.blur(image2, (4,4))
        image2 = cv.Canny(image2, 100, 200)

        adaptive_threshold = cv.adaptiveThreshold(image2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)


        # image2 = cv.Sobel(image2, cv.CV_64F, 1, 0, ksize=5)
        # image2 = cv.matchTemplate(image2, img_name, cv.TM_CCOEFF_NORMED)
        
        num = 0
        for coord in pkm_coordinates:
            
            color = (0, 255, 0)
            num += 1
            # print("coord: ", coord)
            one_place_img = four_point_transform(adaptive_threshold, coord)
            one_place_res = cv.resize(one_place_img, (200, 200))

            one_place_img_template = four_point_transform(image3, coord)
            one_place_res_template = cv.resize(one_place_img_template, (200, 200)) 

            # cv.imwrite("template_night.jpg", one_place_res_template)

            min_val_light = min_val(one_place_res_template, cv.imread("template_sun.jpg"))
            min_val_shadow = min_val(one_place_res_template, cv.imread("template_shadow.jpg"))
            min_val_shadow3 = min_val(one_place_res_template, cv.imread("template_shadow3.jpg"))
            min_val_template = min_val(one_place_res_template, cv.imread("template.jpg"))
            
            light = light_slot(min_val_light)
            shadow3 = shadow_slot(min_val_shadow3)
            shadow = shadow_slot(min_val_shadow)
            template = shadow_slot(min_val_template)

            countwhite = cv.countNonZero(one_place_res)
            
            print("light", light, "min_val_light", min_val_light,
                   "\nshadow3", shadow3, "min_val_shadow3", min_val_shadow3,
                     "\nshadow", shadow, "min_val_shadow", min_val_shadow, 
                        "\ntemplate", template, "min_val_template", min_val_template,
                     "\ncountwhite", countwhite)
            print("---------------------------------------------")

            if countwhite == 0:
                my_result.append(0)
            elif countwhite < 8900:
                if not light and ( not shadow3 or not shadow):
                    # print("light: ", light, "shadow3: ", shadow3)
                    color = (0, 0, 255) # red
                    my_result.append(1)
                else:
                    my_result.append(0)
            else:
                color = (0, 0, 255) # red
                my_result.append(1)

            

            cv.line(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color, 4)
            cv.line(image, (int(coord[2]), int(coord[3])), (int(coord[4]), int(coord[5])), color, 4)
            cv.line(image, (int(coord[4]), int(coord[5])), (int(coord[6]), int(coord[7])), color, 4)

            centerX = int((int(coord[0]) + int(coord[4])) / 2)
            centerY = int((int(coord[1]) + int(coord[5])) / 2)

            cv.circle(image, (centerX, centerY), 15, color, -1)
            # cv.putText(image, str(num), (centerX + 10, centerY - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(image, str(countwhite), (centerX + 10, centerY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
            
            cv.imshow("image", image)
            cv.imshow("image2",adaptive_threshold)
            

            cv.imshow("one_place", one_place_res)
            # cv.waitKey(50000)
        print("********************************************************")
        
        print(my_result)
        print(result[i])
        
        f1_score = calculate_f1_score(result[i], my_result)
        
        cv.putText(image, str(f"F1:{f1_score}"), (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow("image",image)
        print("F1 score:", f1_score)
        f1_scores.append(f1_score)
        i += 1


        if cv.waitKey(0) == ord('q'):
            break
    print("FINAL F1 SCORE:", sum(f1_scores) / len(f1_scores))
        
            
    
if __name__ == "__main__":
   main(sys.argv[1:])     
