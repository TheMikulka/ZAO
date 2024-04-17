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
        x += 50
        y += 50
        w -= 100
        h -= 100
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color1, 8)
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color2, 2)

def calculate_face_center(face_location):
    x = face_location[0] + face_location[2] / 2
    y = face_location[1] + face_location[3] / 2
    return x, y

def face_detect():
    video_cap = cv.VideoCapture("fusek_face_car_01.avi")
   # video_cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_profile = cv.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    # src_img = cv.imread("src_images/Milos_Zeman.jpg")
    src_img = cv.imread("src_images/jan_platos.png")
    src_img_copy = src_img.copy()
    face_roi_src = None
    src_face_mask = None

    if src_img is not None:
        location_src = draw_rect_by_cascade(face_cascade, src_img, 1.1, 5, (100, 100))
        draw_rectangle(src_img, location_src, (0, 0, 255), (255, 0, 0))
        for i,(x, y, w, h) in enumerate(location_src):
            x += 50
            y += 50
            w -= 80
            h -= 50
            face_roi_src = src_img_copy[y:y+h, x:x+w]
    
        cv.imshow("src_img", src_img)
        

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

            # for one_face in locations_face:
            #    draw_rectangle(paint_frame, [one_face], (0,0,255), (203, 192, 255))
            for i,(x, y, w, h) in enumerate(locations_face):
                x += 50
                y += 70
                w -= 100
                h -= 50
                face_roi = frame[y:y+h, x:x+w]
            
                height,width,_ = face_roi.shape
                face_roi_src_r = cv.resize(face_roi_src, (width, height))
                cv.imshow(f"src_face", face_roi_src_r)

                cv.circle(face_roi, (width//2, height//2), 10, (0, 255, 0), -1)
                cv.imshow(f"Face", face_roi)

                src_face_mask = np.ones_like(face_roi) * 255
                cv.imshow("src_face_mask", src_face_mask)
                            
                seamlessclone = cv.seamlessClone(face_roi_src_r, face_roi, src_face_mask, (width//2, height//2), cv.MONOCHROME_TRANSFER)
                cv.imshow("seamlessclone", seamlessclone)

                if y + height <= paint_frame.shape[0] and x + width <= paint_frame.shape[1]:
                    paint_frame[y:y + height, x:x + width] = seamlessclone

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

        