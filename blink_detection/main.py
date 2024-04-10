import dlib
import cv2 as cv
import time
import glob
import numpy as np

def draw_rectangle (paint_frame, locations, color1, color2):
    for (x, y, w, h) in locations:
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color1, 2)
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color2, 1)

def draw_rect_by_cascade(face_cascade, frame, scaleFactor, minNeighbors, minSize):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, maxSize=(500, 500))
    return faces

def calculate_EAR(points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    points = [(p.x, p.y) for p in points]
    p1, p2, p3, p4, p5, p6 = points

    numerator = euclidean_distance((p2, p6), (p3, p5))
    denominator = 2 * euclidean_distance(p1, p4)
    EAR = numerator / denominator
    return EAR

def ear_state(ear):
    if ear < 0.26:
        return 0
    else:
        return 1
    
def calculate_color_ear(eye_states):
    length = len(eye_states)
    summary = sum(eye_states)
    if summary < length/2:
        return (0, 255, 255)
    else:
        return (0, 255, 0)

def face_detect():
    images = [img for img in glob.glob("anomal_hd_30fps_01/*.jpg")]
    images.sort()
    cv.namedWindow("face_detect", cv.WINDOW_NORMAL)
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("landmark_models/dlib_shape_predictor_68_face_landmarks.dat")
    face_cascade = cv.CascadeClassifier("landmark_models/lbpcascade_frontalface_improved.xml")
    left_eye_states = []
    right_eye_states = []

    max_eye_states_l = 3
    max_eye_states_r = 3

    for image in images:
        img = cv.imread(image)
        start_time = time.time()
        # faces = detector(img, 0)
        faces = draw_rect_by_cascade(face_cascade, img, 1.2, 7,  (100,100))

        for i,face in enumerate(faces):
            face_x, face_y, face_w, face_h = face
            # pt1 = (f.left(), f.top())
            # pt2 = (f.right(), f.bottom())
            face = dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h)
            shape = landmark_predictor(img, face)
            
            draw_rectangle(img, [(face_x, face_y, face_w, face_h)], (0, 255, 0), (255, 0, 0))
            
            if max_eye_states_r != 0:
                right_eye_states.append(ear_state(calculate_EAR(shape.parts()[42:48])))
                max_eye_states_r -= 1
            else:
                right_eye_states.pop(0)
                right_eye_states.append(ear_state(calculate_EAR(shape.parts()[42:48])))
                
            if max_eye_states_l != 0:
                left_eye_states.append(ear_state(calculate_EAR(shape.parts()[36:42])))
                max_eye_states_l -= 1
            else:
                left_eye_states.pop(0)
                left_eye_states.append(ear_state(calculate_EAR(shape.parts()[36:42])))
            for ip,p in enumerate(  shape.parts()[0:35]):
                cv.circle(img, (p.x, p.y), 2, (255, 255, 255), -1)
                # cv.putText(img, str(ip), (p.x, p.y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)

            for ip, p in enumerate(shape.parts()[43:]):
                cv.circle(img, (p.x, p.y), 2, (255, 255, 255), -1)
                # cv.putText(img, str(ip), (p.x, p.y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)

            for ip, p in enumerate(shape.parts()[36:42]):
                cv.circle(img, (p.x, p.y), 2, calculate_color_ear(left_eye_states), -1)
                
            for ip, p in enumerate(shape.parts()[42:48]):
                cv.circle(img, (p.x, p.y), 2, calculate_color_ear(right_eye_states), -1)

            
            print(f"LEFT EAR: {left_eye_states}, RIGHT EAR: {right_eye_states}")
        end_time = time.time()
        final_time = end_time - start_time
        cv.putText(img, f"FPS: {1/final_time:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(img, f"DELTA: {final_time:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("face_detect", img)
        
        
        if cv.waitKey(2) == ord('q'):
            break
if __name__ == "__main__":
    face_detect()