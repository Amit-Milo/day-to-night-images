from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from image_utils import image_to_numpy


def __get_depth_map(midas, midas_transform, content_path, device):
    # load image
    img = cv2.imread(content_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img).to(device)

    # get depth map
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output


def __convert_depth_map_bw(depth_map, new_size=None):
    # create a black-white theme version of the depth map, and return it.
    fig = plt.figure()
    plt.axis('off')

    # display the image onto a grayscale plot
    plt.set_cmap("gray")
    plt.imshow(depth_map)

    # save to the plot to a temp bytes IO and then read from it as an Image
    temp_image_mem = BytesIO()
    fig.savefig(temp_image_mem, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    image = Image.open(temp_image_mem)
    img = image_to_numpy(image, new_size)

    return img


def get_depth_style(content_path, midas, midas_transform, device, new_size=None):
    depth_map = __get_depth_map(midas, midas_transform, content_path, device)
    depth_map_bw = __convert_depth_map_bw(depth_map, new_size)
    return depth_map_bw
