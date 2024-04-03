import cv2 as cv
import numpy as np
import time

def draw_rect_by_cascade(face_cascade, frame, scaleFactor, minNeighbors, minSize):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, maxSize=(500, 500))
    return faces

def remove_duplicates(locations, threshold_distance):
    non_duplicates = []
    for i in range(len(locations)):
        is_duplicate = False
        for j in range(i+1, len(locations)):
            dist = ((locations[i][0] + locations[i][2] / 2) - (locations[j][0] + locations[j][2] / 2))**2 + \
                   ((locations[i][1] + locations[i][3] / 2) - (locations[j][1] + locations[j][3] / 2))**2
            dist = dist ** 0.5
            if dist < threshold_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            non_duplicates.append(locations[i])
    return non_duplicates

def draw_rectangle (paint_frame, locations, color1, color2):
    for (x, y, w, h) in locations:
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color1, 8)
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color2, 2)

def eye_open(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 5), 2)
    cv.imshow("Preprocessed Image", blurred)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 1, param1=50, param2=20, minRadius=10, maxRadius=40)
    if circles is not None:
        return True

    return False

def face_detect():
    video_cap = cv.VideoCapture("fusek_face_car_01.avi")
   # video_cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_profile = cv.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    #eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
    eye_cascade = cv.CascadeClassifier("haarcascades/eye_cascade_fusek.xml")
    mouth_cascade = cv.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    threshold_distance = 100.0   
    while True:
        start_time = time.time()
        ret, frame = video_cap.read()
        if frame is None:
            break
        paint_frame = frame.copy()
        locations_face = []
        if ret is True:
            location_face = draw_rect_by_cascade(face_cascade, paint_frame, 1.2, 7,  (100,100))
            for l_face in location_face:
                locations_face.append(l_face)
            location_face_profile = draw_rect_by_cascade(face_cascade_profile, frame, 1.2, 7, (100,100))
            for l_face in location_face_profile:
                locations_face.append(l_face)

            locations_face = remove_duplicates(locations_face, threshold_distance)
            # print(locations_face)

            for one_face in locations_face:
               draw_rectangle(paint_frame, [one_face], (0,0,255), (203, 192, 255))


            for i,(x, y, w, h) in enumerate(locations_face):
                face_roi = frame[y:y+h, x:x+w]
                cv.imshow(f"Face", face_roi)
                eyes = draw_rect_by_cascade(eye_cascade, face_roi, 1.3, 13,(30,30))
                for eye_index, eye in enumerate(eyes):
                    eye_x, eye_y, eye_w, eye_h = eye
                    
                    #cv.imshow(f"Eye", face_roi[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w])
                    if eye_open(face_roi[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]):
                        #print(f"Eye {eye_index} is open")
                        draw_rectangle(paint_frame[y:y+h, x:x+w], [eye], (0,255,0), (203, 192, 255))
                    else:
                        #print(f"Eye {eye_index} is closed")
                        draw_rectangle(paint_frame[y:y+h, x:x+w], [eye], (0,0,255), (203, 192, 255))
            
                mouth = draw_rect_by_cascade(mouth_cascade, face_roi, 1.2, 50, (40,40))
                for m in mouth:
                    draw_rectangle(paint_frame[y:y+h, x:x+w], [m], (255,0,0), (203, 192, 255))
            end_time = time.time()

            final_time = end_time - start_time
            cv.putText(paint_frame, f"FPS: {1/final_time:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(paint_frame, f"DELTA: {final_time:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            
            



            cv.imshow("face_detect", paint_frame)
            if cv.waitKey(2) == ord('q'):
                break

if __name__ == "__main__":
    face_detect()
    cv.destroyAllWindows()

        