import numpy as np
import copy


def flat_features(features):
    new_elem = np.array(features)
    x, y, z = new_elem.shape
    return new_elem.reshape(x * y * z)


def get_haar_features_stacked(image, kernel_size=6):
    Hx, Hy, Hd, HLx, HLy = compute_haar_features(image, kernel_size)

    x = Hx.shape[0]
    y = Hx.shape[1]

    result = list()

    for i in range(x):
        row = list()
        for j in range(y):
            row.append(np.array((Hx[i, j], Hy[i, j], Hd[i, j], HLx[i, j], HLy[i, j],)))
        result.append(np.array(row))
    return result


def calculate_rect_sum(integral_image, x1, y1, x2, y2):
    upper_sum = 0
    left_sum = 0
    upper_left_sum = 0

    if y1 != 0:
        upper_sum = integral_image[y1-1, x2]
    if x1 != 0:
        left_sum = integral_image[y2, x1-1]
    if x1 != 0 and y1 != 0:
        upper_left_sum = integral_image[y1-1, x1-1]
    return integral_image[y2, x2] + upper_left_sum - upper_sum - left_sum


def compute_integral_image(image):
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


def compute_haar_features(img, haar_filter_size):
    integral_img = compute_integral_image(img)
    img_height, img_width = integral_img.shape

    filter_middle_point = haar_filter_size//2

    output_width = img_width-haar_filter_size + 1
    output_height = img_height-haar_filter_size + 1

    Hx = np.zeros(shape=(output_height, output_width), dtype=np.int32)
    Hy = np.zeros(shape=(output_height, output_width), dtype=np.int32)
    Hd = np.zeros(shape=(output_height, output_width), dtype=np.int32)
    HLx = np.zeros(shape=(output_height, output_width), dtype=np.int32)
    HLy = np.zeros(shape=(output_height, output_width), dtype=np.int32)

    for row in range(output_height):
        for col in range(output_width):
            x1 = col
            y1 = row
            x2 = x1 + haar_filter_size - 1
            y2 = y1 + haar_filter_size - 1

            # Hx
            leftSum = calculate_rect_sum(integral_img, x1, y1, x1 + filter_middle_point - 1, y2)
            rightSum = calculate_rect_sum(integral_img, x1 + filter_middle_point, y1, x2, y2)
            Hx[row, col] = leftSum - rightSum

            #Hy
            upperSum = calculate_rect_sum(integral_img, x1, y1, x2, y1 + filter_middle_point - 1)
            lowerSum = calculate_rect_sum(integral_img, x1, y1 + filter_middle_point, x2, y2)
            Hy[row, col] = upperSum - lowerSum

            #Hd
            topLeftSum = calculate_rect_sum(integral_img, x1, y1, x1 + filter_middle_point - 1, y1 + filter_middle_point - 1)
            topRightSum = calculate_rect_sum(integral_img, x1 + filter_middle_point, y1, x2, y1+filter_middle_point - 1)
            bottomLeftSum = calculate_rect_sum(integral_img, x1, y1+filter_middle_point, x1+filter_middle_point-1, y2)
            bottomRightSum = calculate_rect_sum(integral_img, x1+filter_middle_point, y1+filter_middle_point, x2, y2)
            Hd[row, col] = topLeftSum + bottomRightSum - topRightSum - bottomLeftSum

            #HLx
            filter_third_point = haar_filter_size // 3
            leftWhiteSum = calculate_rect_sum(integral_img, x1, y1, x1 + filter_third_point - 1, y2)
            middleBlackSum = calculate_rect_sum(integral_img, x1 + filter_third_point, y1, x2 - filter_third_point, y2)
            rightWhiteSum = calculate_rect_sum(integral_img, x2 - filter_third_point + 1, y1, x2, y2)
            HLx[row, col] = leftWhiteSum + rightWhiteSum - middleBlackSum

            #HLy
            upperWhiteSum = calculate_rect_sum(integral_img, x1, y1, x2, y1 + filter_third_point - 1)
            middleBlackSum = calculate_rect_sum(integral_img, x1, y1 + filter_third_point, x2, y2 - filter_third_point)
            lowerWhiteSum = calculate_rect_sum(integral_img, x1, y2 - filter_third_point + 1, x2, y2)
            HLy[row, col] = upperWhiteSum + lowerWhiteSum - middleBlackSum

    return (Hx, Hy, Hd, HLx, HLy)
