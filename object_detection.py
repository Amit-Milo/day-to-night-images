import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import transforms as transforms


def __get_od_model_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [idx for idx, score in enumerate(scores) if score > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    return boxes


def get_od_objects(od_model, content_path, device, threshold=0.5):
    image = Image.open(content_path).convert('RGB')
    # transform to convert the image to tensor
    transform_image_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform_image_to_tensor(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    return __get_od_model_outputs(image, od_model, threshold)
