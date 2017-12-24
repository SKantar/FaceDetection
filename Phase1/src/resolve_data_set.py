import pandas as pd
import numpy as np
from sklearn.utils import shuffle
lable_path = '/home/skantar/Desktop/train_data/training_labels.csv'
labels_df = pd.read_csv(lable_path, index_col=0)

print("====================================")
print(labels_df[labels_df.label == 0].count())
print(labels_df[labels_df.label == 1].count())

train_face = labels_df[labels_df.label > 0].iloc[0:2500, :]
train_noface =labels_df[labels_df.label == 0].iloc[0:2500, :]

train_frames = [train_face, train_noface]
train_df = pd.concat(train_frames)
train_df = shuffle(train_df)

print("====================================")
print(train_df[train_df.label == 0].count())
print(train_df[train_df.label == 1].count())

train_df.to_csv('/home/skantar/Documents/Projects/Studies/hakaton/labels/train_labels.csv')



test_face = labels_df[labels_df.label > 0].iloc[2500:2800, :]
test_noface = labels_df[labels_df.label == 0].iloc[2500:2800, :]

test_frames = [test_face, test_noface]
test_df = pd.concat(test_frames)
test_df = shuffle(test_df)
print("====================================")
print(test_df[test_df.label == 0].count())
print(test_df[test_df.label == 1].count())
test_df.to_csv('/home/skantar/Documents/Projects/Studies/hakaton/labels/test_labels.csv')



other_face = labels_df[labels_df.label > 0].iloc[2800:, :]
other_noface = labels_df[labels_df.label == 0].iloc[2800:, :]

other_frames = [other_face, other_noface]
other_df = pd.concat(other_frames)
other_df = shuffle(other_df)
print("====================================")
print(other_df[other_df.label == 0].count())
print(other_df[other_df.label == 1].count())
other_df.to_csv('/home/skantar/Documents/Projects/Studies/hakaton/labels/other_labels.csv')
