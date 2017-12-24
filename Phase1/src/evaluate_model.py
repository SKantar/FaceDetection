import numpy, pickle

def evaluate_model(model_pickle_filename, test_data_filename):
    """Make prediction for data_test files using trained 
    pickle model stored in filesistem"""
    model = pickle.load(open(model_pickle_filename, 'rb'))
    data = pickle.load(open(test_data_filename, 'rb'))

    images = list()
    labels = list()

    for elem in data:
        new_elem = numpy.array(elem[0])
        x, y, z = new_elem.shape
        images.append(new_elem.reshape(x * y * z))
        labels.append(elem[1])

    images = numpy.array(images)
    labels = numpy.array(labels)

    y = model.predict(images)

    correct = 0
    for i, l in enumerate(labels):
        if y[i] == l: correct += 1

    accuracy = correct / len(labels)
    print("Accuracy:", accuracy)


def evaluate_test():
    """Function evalueate while test data and return accurancy"""

    model200x200 = pickle.load(open('training_models/model200x200.pkl', 'rb'))
    model100x100 = pickle.load(open('training_models/model100x100.pkl', 'rb'))
    model50x50 = pickle.load(open('training_models/model50x50.pkl', 'rb'))
    model25x25 = pickle.load(open('training_models/model25x25.pkl', 'rb'))

    data200x200 = pickle.load(open('test/test200x200.pkl', 'rb'))
    data100x100 = pickle.load(open('test/test100x100.pkl', 'rb'))
    data50x50 = pickle.load(open('test/test50x50.pkl', 'rb'))
    data25x25 = pickle.load(open('test/test25x25.pkl', 'rb'))

    images200x200 = list()
    images100x100 = list()
    images50x50 = list()
    images25x25 = list()

    labels200x200 = list()
    labels100x100 = list()
    labels50x50 = list()
    labels25x25 = list()

    for elem in data200x200:
        new_elem = numpy.array(elem[0])
        x, y, z = new_elem.shape
        images200x200.append(new_elem.reshape(x * y * z))
        labels200x200.append(elem[1])

    for elem in data100x100:
        new_elem = numpy.array(elem[0])
        x, y, z = new_elem.shape
        images100x100.append(new_elem.reshape(x * y * z))
        labels100x100.append(elem[1])

    for elem in data50x50:
        new_elem = numpy.array(elem[0])
        x, y, z = new_elem.shape
        images50x50.append(new_elem.reshape(x * y * z))
        labels50x50.append(elem[1])

    for elem in data25x25:
        new_elem = numpy.array(elem[0])
        x, y, z = new_elem.shape
        images25x25.append(new_elem.reshape(x * y * z))
        labels25x25.append(elem[1])

    images200x200 = numpy.array(images200x200)
    images100x100 = numpy.array(images100x100)
    images50x50 = numpy.array(images50x50)
    images25x25 = numpy.array(images25x25)

    labels200x200 = numpy.array(labels200x200)
    labels100x100 = numpy.array(labels100x100)
    labels50x50 = numpy.array(labels50x50)
    labels25x25 = numpy.array(labels25x25)

    y200x200 = model200x200.predict(images200x200)
    y100x100 = model100x100.predict(images100x100)
    y50x50 = model50x50.predict(images50x50)
    y25x25 = model25x25.predict(images25x25)

    y = list()


    correct = 0
    for i, l in enumerate(labels200x200):
        if(y200x200[i] + y100x100[i] + y50x50[i] + y25x25[i] > 1):
            if 1 == l:
                correct += 1
        else:
            if 0 == l:
                correct += 1
    accuracy = correct / len(labels200x200)
    print("Accuracy:", accuracy)

def predict_image_200(image):
    model = pickle.load(open('training_models/model200x200.pkl', 'rb'))
    # print(model.predict_proba(image)[0])
    return model.predict(image)

def predict_image_100(image):
    model = pickle.load(open('training_models/model100x100.pkl', 'rb'))
    # print(model.predict_proba(image)[0])
    return model.predict(image)

def predict_image_50(image):
    model = pickle.load(open('training_models/model50x50.pkl', 'rb'))
    # print(model.predict_proba(image)[0])
    return model.predict(image)

def predict_image_25(image):
    model = pickle.load(open('training_models/model25x25.pkl', 'rb'))
    # print(model.predict_proba(image))
    return model.predict(image)
