from math import sqrt
import torch

from create_style import create_final_style
from depth_map import get_depth_style
from image_utils import image_path_to_numpy, image_to_tensor
from plotting import plot_result
from preparation import get_depth_map_model, get_object_detection_model, get_style_transfer_model
from style_transfer import style_transfer


def get_content_path():
    """
    set the path of the image to apply the model on
    """
    # path = "images/rhino/002.jpg"
    # path = "images/elephant/054.jpg"
    # path = "images/zebra/073.jpg"
    # path = "images/buffalo/051.jpg"
    path = "images/others/sun.jpeg"

    return path


def get_device(strong_cuda):
    if strong_cuda:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


if __name__ == '__main__':
    # get device
    device = get_device(False)

    # get models
    od_model = get_object_detection_model(device)
    dm_model, dm_model_transforms = get_depth_map_model(device)
    st_model = get_style_transfer_model(device)

    # get image path
    content_path = get_content_path()

    # operate
    content_np = image_path_to_numpy(content_path)
    depth_style_np = get_depth_style(content_path, dm_model, dm_model_transforms, device, new_size=content_np.size)

    style_np = create_final_style(content_path, od_model, depth_style_np, device)

    content_tensor = image_to_tensor(content_np).to(device)
    style_tensor = image_to_tensor(style_np).to(device)

    alpha = 1
    beta = 1.5
    final_image_tensor = style_transfer(st_model, content_tensor, style_tensor,
                                        sqrt(beta / alpha), device, n_iterations=500, learning_rate=0.02)
    # final results

    plot_result(final_image_tensor)

    plot_result(content_tensor)
