import pickle, numpy, pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def train_model(data_pickle_filename, output_filename,
                estimators=100,
                depth=7,
                features='sqrt',
                split=1e-7,
                cv_fold_evaluation=5,
                verbose=False):

    data = pickle.load(open(data_pickle_filename, 'rb'))

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

    lr = GradientBoostingClassifier(verbose=verbose, n_estimators=estimators, max_depth=depth, max_features=features, min_impurity_split=split)

    if cv_fold_evaluation > 0:
        scores = cross_val_score(lr, images, labels, cv=cv_fold_evaluation)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    lr.fit(images, labels)

    pickle.dump(lr, open(output_filename, 'wb'))

