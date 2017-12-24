import os, cv2, numpy, pandas, pickle,traceback
from haar_features import compute_haar_features
DATA_DIR = "C:/Users/stank/Documents/Dev/hakaton/train_data/train"
LABEL_DIR = "C:/Users/stank/Documents/Dev/hakaton/labels"

# This function produces data .pkl files needed by later stages
# @labels_filename - name of the .csv file used in constructing
# @kernel_size - size of Haar Feature kernel
# @output - path to the file where .pkl will be stored
def proces_data(labels_filename, kernel_size, output = 'data50x50.pkl'):

    data = list()

    lables_path = os.path.join(LABEL_DIR, labels_filename)

    labels_df = pandas.read_csv(lables_path)
    print(labels_df.shape)
    for index, row in labels_df.iterrows():
        print(index)
        try:
            data.append((process_image(row.image, kernel_size), row.label, row.image))
        except Exception as ex:
            error_message = str(ex)
            stack_trace = traceback.format_exc()
            print(error_message)
            print(stack_trace)

    pickle.dump(data, open(output, "wb"))

    print("FINISHED")


def process_image(image_name, kernel_size):
    image_path = os.path.join(DATA_DIR, image_name)
    image = cv2.imread(image_path, 0)
    image = cv2.pyrDown(image)
    image = cv2.pyrDown(image)
    #image = cv2.equalizeHist(image)
    #image = adjust_gamma(image, gamma=1.5)
    Hx, Hy, Hd, HLx, HLy = compute_haar_features(image, kernel_size)

    x = Hx.shape[0]
    y = Hx.shape[1]

    result = list()

    for i in range(x):
        row = list()
        for j in range(y):
            row.append(numpy.array((Hx[i, j], Hy[i, j], Hd[i, j], HLx[i, j], HLy[i, j],)))
        result.append(numpy.array(row))
    return result


def process_image_and_flat(image_name, kernel_size):
    result = process_image(image_name, kernel_size)
    x, y, z = result.shape
    return result.reshape(x * y * z)


