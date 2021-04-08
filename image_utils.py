import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import transforms as transforms


def image_to_numpy(image, new_size=None):
    if new_size is not None:
        image = image.resize(new_size, Image.ANTIALIAS)
    return image.convert('RGB')


def image_path_to_numpy(image_path, new_size=None):
    # load image into a numpy array from the given path
    image = Image.open(image_path)
    return image_to_numpy(image, new_size)


def image_to_tensor(image_numpy, max_size=400, shape=None):
    # crop image if image is too big
    if max(image_numpy.size) > max_size:
        size = max_size
    else:
        size = max(image_numpy.size)

    size = (size, int(1.5 * size))
    # if shape is given use it
    if shape is not None:
        size = shape

    # resize and normalize the image
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image_numpy)[:3, :, :].unsqueeze(0)

    return image


def tensor_to_numpy(image_tensor):
    image = image_tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image
