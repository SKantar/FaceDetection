from detect_faces import *
from train_model import *
from os import listdir
from os.path import isfile, join
from preprocess_data import *

# Usage:
# 1) Set DATA_DIR in preprocess_data.py to point to folder into which training images are
# 2) Set LABEL_DIR in preprocess_data.py to point to the folder into which labels are
# 3) call process_data
# 4) call train_model
# 5) call detect_faces

#proces_data('train_labels.csv', 6)


'''train_model('train_data/data50x50.pkl', 'trained_models/model50x50.pkl',
            cv_fold_evaluation=0, verbose=True)
train_model('train_data/data25x25.pkl', 'trained_models/model25x25.pkl',
             cv_fold_evaluation=0, verbose=True)
'''


test_detection_path = 'test_j/detection_images/images'
onlyfiles = [f for f in listdir(test_detection_path) if isfile(join(test_detection_path, f))]

results = []
for file in onlyfiles:
    file = os.path.join(test_detection_path, file)
    img_name = file
    print(file)
    detect_faces(file, step_size=20)


