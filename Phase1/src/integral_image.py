import numpy as np
import cv2, copy

def create_data():
    total = list()
    for x in range(5):
        row = list()
        for y in range(5):
            colum = list()
            for c in range(3):
                colum.append(1)
            row.append(np.array(colum))
        total.append(np.array(row))
    return np.array(total)


def integral_image(image):
    new_image = copy.deepcopy(image).astype(np.int32)

    sy = image.shape[0]
    sx = image.shape[1]
    nc = len(image.shape) == 3 and image.shape[2] or 1

    for x in range(sx):
        for y in range(sy):
            value = image[y][x]
            if x > 0:
                if y > 0:
                    xo = new_image[y][x - 1]
                    yo = new_image[y - 1][x]
                    xoyo = new_image[y - 1][x - 1]
                else:
                    xo = new_image[y][x - 1]
                    yo = 0
                    xoyo = 0
            else:
                if y > 0:
                    xo = 0
                    yo = new_image[y - 1][x]
                    xoyo = 0
                else:
                    xo = 0
                    yo = 0
                    xoyo = 0
            new_image[y][x] = value - xoyo + xo + yo
    return new_image



image = cv2.imread('/home/skantar/Desktop/train_data/train/15.jpg')
image_path = integral_image(image)
pass
# plt.imshow(image)
# plt.show()