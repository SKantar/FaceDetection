import os, cv2, numpy, pandas, pickle,traceback
from haar_features import compute_haar_features
DATA_DIR = "/home/skantar/Desktop/train_data/train"
LABEL_DIR = "/home/skantar/Documents/Projects/Studies/hakaton/labels"

def process_image(image, kernel_size):
    """ Function apply features on image an return 
    2d image with 5 canels(featur values)"""
    # image_path = os.path.join(DATA_DIR, image_name)
    # image = cv2.imread(image_path, 0)
    # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    # image = cv2.pyrDown(image)
    # image = cv2.pyrDown(image)
    Hx, Hy, Hd, HLx, HLy = compute_haar_features(image, kernel_size)

    x = Hx.shape[0]
    y = Hx.shape[1]

    result = list()

    for i in range(x):
        row = list()
        for j in range(y):
            row.append(numpy.array((Hx[i, j], Hy[i, j], Hd[i, j], HLx[i, j], HLy[i, j],)))
        result.append(numpy.array(row))
    return numpy.array(result)

def process_image_and_flat(image, kernel_size):
    """Flat 3d Matrix to 1d array"""
    result = process_image(image, kernel_size)
    x, y, z = result.shape
    return result.reshape(x * y * z)


def proces_data(labels_filename, kernel_size):
    """Save processed data in pickle file
        Files used for training for faster execution
    """
    data = list()

    lables_path = os.path.join(LABEL_DIR, labels_filename)

    labels_df = pandas.read_csv(lables_path)
    labels_df = labels_df
    print(labels_df.shape)
    for index, row in labels_df.iterrows():
        print(index)
        try:
            data.append((process_image(row.image, kernel_size), row.label, row.name))
        except Exception as ex:
            error_message = str(ex)
            stack_trace = traceback.format_exc()
            print(error_message)
            print(stack_trace)

    pickle.dump(data, open("data_normalized.pkl", "wb"))

    print("FINISHED")

# proces_data("train_labels.csv", 6)

