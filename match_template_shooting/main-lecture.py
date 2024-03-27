import numpy as np
import cv2 as cv
from pynput.mouse import Button, Controller
from PIL import ImageGrab
import time

window_scaling = 1.25

def template_simple(screen_mat, template):
    temp_mat = cv.imread(template)
    screen_mat_gray = cv.cvtColor(screen_mat, cv.COLOR_BGR2GRAY)
    temp_mat_gray = cv.cvtColor(temp_mat, cv.COLOR_BGR2GRAY)

    
    cv.imshow("Template", temp_mat)
    # cv.imshow("Mat_copy", screen_mat)


    result = cv.matchTemplate(screen_mat_gray, temp_mat_gray, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, mac_loc = cv.minMaxLoc(result)

    if min_val >= 0.10:
        print("Template not found")
        return -1, -1
    print("Min val", min_val)
    # if min_val >= 0.10:
    #     print("Template not found")
    #     return -1, -1
    top_left = min_loc
    bottom_right = (top_left[0] + temp_mat_gray.shape[1], top_left[1] + temp_mat_gray.shape[0])
    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2

    cv.rectangle(screen_mat, top_left, bottom_right, (0, 255, 0), 2)
    cv.circle(screen_mat, (center_x, center_y), 5, (0, 0, 255), -1)

    return center_x, center_y

def capture():
    image_grab = ImageGrab.grab(bbox=None)
    screen_mat = np.array(image_grab)
    return screen_mat

if __name__ == "__main__":
    print("Started")
    # templates = ["template2.png", "template3.png", "template4.png", "template5.png", "template6.png", "template7.png"]
    templates = ["head.png" ,"head2.png"]
    # templates = ["template.png"]
    mouse = Controller()
    future_x = mouse.position[0]
    future_y = mouse.position[1]

    while True:
        if(mouse.position[0] != future_x and mouse.position[1] != future_y):
            future_x = mouse.position[0]
            future_y = mouse.position[1]
            print("Waiting")
            cv.waitKey(5000)
            continue
        screen_mat = capture()
        cv.waitKey(10)
        screen_mat_2 = capture()

        for template in templates:
            prev_x, prev_y = template_simple(screen_mat, template)
            x, y = template_simple(screen_mat_2, template)
            if x != -1 and y != -1 and prev_x != -1 and prev_y != -1:
                print("Difference in position:", abs(prev_x - x), abs(prev_y - y))
                if(abs(prev_x - x) < 100 and abs(prev_y - y) < 100):

                    velocity_x = x - prev_x
                    velocity_y = y - prev_y

                    future_x = x + velocity_x * 4
                    future_y = y + velocity_y * 4

                    print("Predicted position:", future_x, future_y)
                    print("Position:", x, y)
                    
                    print("The current position is {0}".format(mouse.position))
                    mouse.position = (1 / window_scaling * future_x, 1 / window_scaling * future_y)
                    future_x = mouse.position[0]
                    future_y = mouse.position[1]
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                else:
                    print("Different template")

            # else:
            #     continue
        cv.waitKey(650)






# A = [4, 5, 8, 2, 9, 9, 1, 3, 7, 4, 5]
# B = [6, 7, 9, 3, 9, 3, 1, 6, 7]

# SSD = 0
# CC = 0
# SAD = 0

# for i in range(len(A) if len(A) <= len(B) else len(B)):
#     SAD += np.abs(A[i] - B[i])
#     SSD += (A[i] - B[i]) ** 2
#     CC += A[i] * B[i]

# print("\033[1;36mInfo:\033[0m")
# print(f" SSD[{SSD}]\n CC[{CC}]\n SAD[{SAD}]")

# A = np.array(A, dtype=np.uint8)
# B = np.array(B, dtype=np.uint8)
# TM_SQDIFF = cv.matchTemplate(A, B, cv.TM_SQDIFF)
# print("\n\033[1;36mTM_SQDIFF:\033[0m\n", TM_SQDIFF)

# # ----------------------------------------------------

# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# cv.namedWindow("img", 0)
# cv.namedWindow("out", 0)

# template = cv.imread("./cv02/oblicej.png", 0)
# cv.imshow("template", template)

# while True:
#     ret, frame = cap.read()
#     if ret:
#         frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         tmp_out = cv.matchTemplate(frame_gray, template, cv.TM_SQDIFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)
#         cv.circle(frame, min_loc, 30, (0, 255, 0), -1)
#         cv.imshow('img', frame)
#         cv.imshow('out', tmp_out)
#         if cv.waitKey(1) == ord('q'):
#             break
#     else:
#         print('Failed to get Camera')
#         break