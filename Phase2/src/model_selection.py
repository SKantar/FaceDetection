import pickle, numpy, pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

data = pickle.load(open('data100x100.pkl', 'rb'))

numpy.random.shuffle(data)

images = list()
labels = list()

for elem in data:
    new_elem = numpy.array(elem[0])
    x, y, z = new_elem.shape
    images.append(new_elem.reshape(x * y * z))
    labels.append(elem[1])

images = numpy.array(images)
labels = numpy.array(labels)

kf = KFold(n_splits=5)
models = []
train_score = []
test_score = []

# Every tuple in params is
# (num_of_estimators, max_depth, max_features, min_impurity_split)

params = [
          (40,  3, 'sqrt', 1e-7),
          (100,  7, 'sqrt', 1e-6),
          (60,  9, 'sqrt', 1e-5),
          (90, 5, 'sqrt', 1e-8),
          (80, 6, 'sqrt', 1e-4)]


fold_iter = 0
best_accuracy = 0
best_model = None
best_params = None
for train, test in kf.split(images):
    print("Starting fold:", fold_iter)
    X_train = images[train]
    X_test = images[test]
    Y_train = labels[train]
    Y_test = labels[test]

    estimators, depth, features, split = params[fold_iter]
    lr = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth, max_features=features, min_impurity_split=split, verbose=True)
    lr.fit(X_train, Y_train)

    models.append(lr)

    y = lr.predict(X_test)

    train_score.append(lr.train_score_)

    correct = 0
    for i, l in enumerate(Y_test):
        if y[i] == l: correct+=1

    accuracy = correct / len(Y_test)
    print("Fold:", i, "Accuracy:", accuracy)
    if(accuracy > best_accuracy):
        best_accuracy = accuracy
        best_modesl = lr
        best_params = params[fold_iter]
    fold_iter += 1


