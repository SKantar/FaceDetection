# FaceDetection

Taks at EESTech Challenge 2017

In this challenge, we was implementing a sliding window face detector. The sliding window model is
conceptually simple: independently classify all image patches as being face or non-face. Face detection
is one of the most noticeable successes of computer vision. For example, modern cameras and photo
organization tools have prominent face detection capabilities.

####Task 1: Face Classifier

Write your code to train a model to classify images to face vs. non-face:

1. write your own code to extract this feature 2 from an image; given an image patch, the value of a
feature =
P (pixel values in white area(s)) − P pixel values in black area(s). The white area and
black area are equal in size. You may want to extract the features at multiple scales (why?). Using
an image pyramid 3 for that purpose, where the five features are extracted from each of the cells
of the pyramid. For example, an 1-layer pyramid leads to a 5-dimensional feature, a 2-layer pyramid
leads to a 25-dimensional feature (1+ 4) ∗ 5 = 25, and a 3-layer leads to an 105-dimensional feature
(1 + 4 + 16) ∗ 5 = 105 ...
2. select a good trade-off for your feature dimensions. Explain why this is good. You can do it either
empirically or by techniques such as Cross-Validation.
3. train a classifier of your choice for a binary classification of face vs. non-face. you are allowed to use
any existing library for the classifier. SVMs, Random Forest, Adaboost, Logistic Regression are among
the popular choices.
4. predict the labels for the test data provided for this task; the predicted labels should be written into
a ’.txt’ file, each line contains a filename and its label, separated by a space. The same format as the
training data.
5. (Optional) use Hard Negative Mining to further improve your classifier. A hard negative is when you
take that falsely detected image, and explicitly create a negative example out of that image, and add
that negative to your training set. When you retrain your classifier, it should perform better with this
extra knowledge, and not make as many false positives.
