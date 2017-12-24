import numpy, pickle


def evaluate_model(model_pickle_filename, test_data_filename):
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