import torch
import torchvision
from torchvision import models


def get_object_detection_model(device):
    od_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    # load the modle on to the computation device and set to eval mode
    od_model.to(device).eval()
    return od_model


def get_depth_map_model(device):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.default_transform
    return midas, midas_transform


def __freeze_network(net):
    """
    The function freezes all layers of a pytorch network (net).
    """
    for param in net.parameters():
        param.requires_grad = False


def __replace_max_pooling(net):
    """
    The function replaces max pooling layers with average pooling layers with
    the following properties: kernel_size=2, stride=2, padding=0.
    """
    current_features = net.features

    for i, feature in enumerate(current_features):
        if isinstance(feature, torch.nn.MaxPool2d):
            current_features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)


def get_style_transfer_model(device):
    vgg = models.vgg19(pretrained=True)
    # freeze all VGG parameters
    __freeze_network(vgg)
    # replace max pooling with avg pooling
    __replace_max_pooling(vgg)
    # move the model to the GPU
    vgg = vgg.to(device).eval()
    return vgg
