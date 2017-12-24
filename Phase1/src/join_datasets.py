import pickle

data1 = pickle.load(open('train/data200X200_1.pkl', 'rb'))
data2 = pickle.load(open('train/data200X200_2.pkl', 'rb'))

data = list()

for elem in data1:
    data.append(elem)
for elem in data2:
    data.append(elem)
pickle.dump(data, open("train/data200x200.pkl", "wb"))