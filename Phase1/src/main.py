#!/usr/bin/env python -W ignore::DeprecationWarning
from evaluate_model import *
import numpy as np
from train_model import *
DATA_DIR = "/home/skantar/Desktop/train_data/train"

# train_model('train_data/data25x25.pkl', 'trained_models/model25x25.pkl', cv_fold_evaluation=5)
# train_model('train_data/data50x50.pkl', 'trained_models/model50x50.pkl', cv_fold_evaluation=5)
# train_model('data_inv.pkl', 'training_models/inv50x50.pkl', verbose=True)

# print("Evaluating 50x50...")
# evaluate_model('training_models/inv50x50.pkl', 'test/test50x50.pkl')
# evaluate_test()
import cv2, os
from preprocess_data import process_image_and_flat



def test_image(image_path):
    """Function calculate image label for image with path image_path
        prediction for picture size 200x200, 100x100, 50x50, 25x25
        then we use most common result
    """
    image = cv2.imread(image_path, 0)
    # image = cv2.equalizeHist(image)
    # image = cv2.blur(image, (5, 5))
    #
    # kernel = np.ones((5, 5), np.float32) / 10
    # image = cv2.filter2D(image, -1, kernel)

    feature = process_image_and_flat(image, 6)
    p1 = predict_image_200(feature)

    image = cv2.pyrDown(image)
    feature = process_image_and_flat(image, 6)
    p2 = predict_image_100(feature)

    image = cv2.pyrDown(image)
    feature = process_image_and_flat(image, 6)
    p3 = predict_image_50(feature)
    #
    image = cv2.pyrDown(image)
    feature = process_image_and_flat(image, 6)
    p4 = predict_image_25(feature)

    return ((p1 + p2 + p3 + p4) > 1)



# #
import warnings
from os import listdir
from os.path import isfile, join
# i = 1
#
import csv
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    with open('/home/skantar/Desktop/test_data/train_image.txt') as file:
        with open('labels/results_other.txt', "w") as out:
            for line in file:
                line = line.strip()
                image_path = os.path.join("/home/skantar/Desktop/test_data/test", line)
                # print(image_path)
                try:
                    res = int(test_image(image_path)[0])
                    out.write(str(res) + "\n")
                    print(line, res)
                except:
                    out.write("0\n")
            # for row in reader:
            #     print(row)
                # image_path = os.path.join(DATA_DIR, row['image'])

#
#
#
    # onlyfiles = [f for f in listdir('/home/skantar/Desktop/bbbb') if isfile(join('/home/skantar/Desktop/bbbb', f))]
    # for file in onlyfiles:
    #     path = os.path.join('/home/skantar/Desktop/bbbb', file)
    #     print(path, test_image(path))
    #
    # print(test_image("/home/skantar/Desktop/test.jpg"))









#     with open('labels/other_labels.csv') as csvfile:
#         reader = csv.DictReader(csvfile)
#         with open('labels/results_other.txt', "w") as out:
#             for row in reader:
#                 image_path = os.path.join(DATA_DIR, row['image'])
#                 res = int(test_image(image_path)[0])
#
#                 out.write(str(res)+"\n")
#                 if str(res) == row['label']:
#                     print(i, "CORRECT")
#                 else:
#                     print(i)
#                 i += 1
                # if i > 100:
                #     break

# image_path = os.path.join(DATA_DIR, "12752.jpg")
# test_image(image_path)







