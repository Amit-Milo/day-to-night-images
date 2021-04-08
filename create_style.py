from math import sqrt, e

import numpy as np
from PIL import Image

from object_detection import get_od_objects


def __get_main_object_box(boxes):
    """
    return the largest box, surface area wise.
    if boxes list is empty, raise Exception
    """
    if len(boxes) == 0:
        raise Exception("Could not find any object in the image...")
    return max(boxes, key=lambda obj: (obj[1][0] - obj[0][0]) * (obj[1][1] - obj[0][1]))


def __distance_2d(x, y, x2, y2):
    return sqrt((x - x2) ** 2 + (y - y2) ** 2)


def __light_gradient(mx, my, dbox, dbox_factor, x, y):
    """
    calculate the light removal factor of the pixel, based on mx, my
    """
    factor_from_middle = dbox / dbox_factor
    d = max(0, __distance_2d(mx, my, x, y) - factor_from_middle)
    return d


def __calc_div_factor(mean):
    """
    We want to decrease the flash's brightness to achieve a more realistic looking.
    We use the div_factor to divide the style image by and thus make it darker.
    Calculate the div factor, based on the mean of the image's pixels
    """
    # The brighter the style is, the bigger div_factor we nead to make it realistic,
    # thus, we know the div_factor depends on the mean brightness of the style.

    # We assume linear relation between the two. After measuring two states we
    # could calculate the liner equation relates the two.
    return mean / 20 + 1


def __calc_dbox_factor(r):
    """
    Calculate dbox_factor as a function of the ratio between the diagonals.
    The dbox_factor is used to make the weights of the style dependent also of the main object size.
    """
    # 1 / (1 - r) takes a value in the range [0, 1) and transforms it to the range [1, inf)
    r = 1 / (1 - r)

    # We use sigmoid function and multiply by a constant, to transform the value into the range [1.82, 2.5]
    # The values graduate slowly and thus the effect of the object's size is moderated.
    r = pow(e, -r)
    r = 1 / (1 + r)
    r = 2.5 * r
    return r


def create_final_style(content_path, model, depth_style, device):
    # convert the depth style to numpy
    ds_arr = np.asarray(depth_style).astype('float')
    # get objects
    boxes = get_od_objects(model, content_path, device)
    # get the main object box
    main_object_box = __get_main_object_box(boxes)
    # get box middle coordinates
    mx = (main_object_box[0][0] + main_object_box[1][0]) / 2
    my = (main_object_box[0][1] + main_object_box[1][1]) / 2
    # get length of diagonal of the box
    dbox = __distance_2d(main_object_box[0][0], main_object_box[0][1], main_object_box[1][0], main_object_box[1][1])
    # get length of diagonal of the whole image
    big_diag = __distance_2d(0, 0, ds_arr.shape[0], ds_arr.shape[1])
    # get the factor of dbox, based on the ratio between the two diagonals
    dbox_factor = __calc_dbox_factor(dbox / big_diag)
    # get max light_gradient out of all image corners
    maxd = max([__light_gradient(mx, my, dbox, dbox_factor, 0, 0),
                __light_gradient(mx, my, dbox, dbox_factor, ds_arr.shape[1], 0),
                __light_gradient(mx, my, dbox, dbox_factor, 0, ds_arr.shape[0]),
                __light_gradient(mx, my, dbox, dbox_factor, ds_arr.shape[1], ds_arr.shape[0])])

    # set p as a function of dbox_factor
    p = 0.2 * dbox_factor + 1
    print(dbox / big_diag, dbox_factor, p)
    for y in range(ds_arr.shape[0]):
        for x in range(ds_arr.shape[1]):
            # transform each pixel, as a function of its distance from the middle of the main box
            d = (1 - __light_gradient(mx, my, dbox, dbox_factor, x, y) / maxd) ** p
            ds_arr[y, x, :] = ds_arr[y, x, :] * d

    # remove brightness in another way, as a function of the mean of the whole image after the first transformation
    ds_arr //= __calc_div_factor(np.mean(ds_arr))
    # convert to image and return
    ds_arr = ds_arr.astype('uint8')
    return Image.fromarray(ds_arr)
