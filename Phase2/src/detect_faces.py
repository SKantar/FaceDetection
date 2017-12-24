import cv2 as cv
import pickle
from haar_features import *

# Function detects faces on image specified by it's filename.
# step_size is how many pixels detection window moves horizontaly/verticaly
# Function returns list of tuples where each tuple is one bounding-box in the following format
# (x1, y1, x2, y2) where (x1, y1) is upper-left, and (x2, y2) is lower-right points

def detect_faces(image_filename, step_size=10):
    color_img = cv.imread(image_filename)
    image = cv.imread(image_filename, 0)
    #image = cv.pyrDown(image)

    window_size = 50
    model50 = pickle.load(open('trained_models/model50x50.pkl', 'rb'))
    model25 = pickle.load(open('trained_models/model25x25.pkl', 'rb'))

    i = 0
    img_height, img_width = image.shape

    out_width = img_width - window_size + 1
    out_height = img_height - window_size + 1

    rects = []

    for y in range(0, out_height, step_size):
        for x in range(0, out_width, step_size):
            image_patch = image[y:y+window_size, x:x+window_size]

            features = [flat_features(get_haar_features_stacked(image_patch, 6))]
            features.append(np.zeros_like(features[0]))
            features = np.array(features)

            face_pred50 = model50.predict_proba(np.array(features))[0][1]

            #cv.imshow('patch1', image_patch)
            image_patch = cv.pyrDown(image_patch)

            #cv.imshow("patch2", image_patch)
            #cv.waitKey()

            features = [flat_features(get_haar_features_stacked(image_patch, 6))]
            features.append(np.zeros_like(features[0]))
            features = np.array(features)

            face_pred25 = model25.predict_proba(np.array(features))[0][1]
            mean = (face_pred25 + face_pred50) / 2
            if mean > 0.8:
                #cv.rectangle(color_img, (x, y), (x+window_size, y+window_size), (0, 255, 0))
                rects.append((mean, x, y, x+window_size, y+window_size))
        print('Finished ', i, " rows of ", out_height//step_size)
        i += 1

    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j: continue
            rect1 = rects[i]
            rect2 = rects[j]
            area1 = (rect1[3] - rect1[1])*(rect1[4] - rect1[2])
            area2 = (rect2[3] - rect2[1])*(rect2[4] - rect2[2])
            if area(intersection(rect1, rect2)) >= max(area1, area2)*0.4:
                score1 = rect1[0]
                score2 = rect2[0]
                if score1 > score2:
                    del rect2
                else:
                    del rect1

    for rect in rects:
        cv.rectangle(color_img, (rect[1], rect[2]), (rect[3], rect[4]), (0, 255, 0))
    cv.imshow('asd', color_img)
    cv.waitKey()
    cv.imwrite(image_filename, color_img)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
    return (x, y, w, h)


def area(rect):
    if rect is None or rect == (): return 0
    return rect[2]*rect[3]
