import cv2 as cv
import numpy as np
import glob
from skimage.feature import local_binary_pattern

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

imgs_free = []
fullTraining = [img for img in glob.glob("full/*.png")]
freeTraining = [img for img in glob.glob("free/*.png")] 
labels = []
training = []
faceRecognizer = cv.face.LBPHFaceRecognizer.create(radius = 1, neighbors=14)

if input("Train again: ").lower() in ["y", "yes"]:
    for path in fullTraining:
        img = cv.imread(path,0)
        resized_img = cv.resize(img, (80, 80))
        labels.append(1)
        training.append(resized_img)

    for path in freeTraining:
        img = cv.imread(path,0)
        resized_img = cv.resize(img, (80, 80))
        labels.append(0)
        training.append(resized_img)

    print("\033[1;36mTraining\033[0m")
    print("GridX", faceRecognizer.getGridX())
    print("GridY", faceRecognizer.getGridY())
    print("Neighbors", faceRecognizer.getNeighbors())
    print("Radius", faceRecognizer.getRadius())

    faceRecognizer.train(training, np.array(labels))
    #faceRecognizer.write("out.yaml")
    print()
else:
    faceRecognizer.read("out.yaml")

def calculate_f1_score(true_answers, my_answers):
    true_positives = sum([1 for t, m in zip(true_answers, my_answers) if t == 1 and m == 1 or t == 0 and m == 0])
    false_positives = sum([1 for t, m in zip(true_answers, my_answers) if t == 0 and m == 1])
    false_negatives = sum([1 for t, m in zip(true_answers, my_answers) if t == 1 and m == 0])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

result = []

for file_path in glob.glob("test_images_zao/*.txt"):
    file_data = []
    with open(file_path, 'r') as file:
        data = [int(x) for x in file.read().split()]
        file_data.extend(data)
    result.append(file_data)

test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
test_images.sort()
cv.namedWindow("image", cv.WINDOW_NORMAL)

pkm_file = open('parking_map_python.txt', 'r')
pkm_lines = pkm_file.readlines()

pkm_coordinates = []

for line in pkm_lines:
    st_line = line.strip()
    sp_line = list(st_line.split(" "))
    pkm_coordinates.append(sp_line)

position_result = 0
final_f1 = []
for img_name in test_images:
    my_result = []
    image = cv.imread(img_name,1)
    image2 = cv.imread(img_name,1)
    image2 = cv.blur(image2, (4,4))
    image2 = cv.Canny(image2, 100, 200)

    my_result = []

    for coord in pkm_coordinates:
        color = (0, 255, 0)
        is_empty = False
        one_place_img = four_point_transform(image, coord)
        one_place_img = cv.cvtColor(one_place_img, cv.COLOR_BGR2GRAY)
        one_place_res = cv.resize(one_place_img, (80, 80))
        is_full_lbp = faceRecognizer.predict(one_place_res)[0] == 1
        
        adaptive_threshold = cv.adaptiveThreshold(image2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)
        one_place_img_edge = four_point_transform(adaptive_threshold, coord)
        one_place_res_edge = cv.resize(one_place_img_edge, (200, 200))
        countwhite = cv.countNonZero(one_place_res_edge)
        print("is_full_lbp ", is_full_lbp)
        print("countwhite ", "full" if countwhite > 10300 else "empty")


        if countwhite != 0:
            if is_full_lbp or countwhite > 10300:
                color = (0, 0, 255)
                is_empty = True
        my_result.append(int(is_empty))
        

        cv.line(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color, 4)
        cv.line(image, (int(coord[2]), int(coord[3])), (int(coord[4]), int(coord[5])), color, 4)
        cv.line(image, (int(coord[4]), int(coord[5])), (int(coord[6]), int(coord[7])), color, 4)

        # cv.imshow("image", image)

        # if cv.waitKey(1) == ord('q'):
        #     break
        # cv.waitKey(1)
    f1_score = calculate_f1_score(result[position_result], my_result)
    print("\033[1;34mF1 score:\033[0m", f1_score)
    final_f1.append(f1_score)
    position_result += 1

    # if cv.waitKey(0) == ord('w'):
    #         break
print("FINAL F1 SCORE:", sum(final_f1) / len(final_f1))
