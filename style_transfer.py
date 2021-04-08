import torch
import torch.nn.functional as F

"""#Style Transfer

##### **2.1 Content Loss**

We start with implementing the content loss.  
The content loss will help us preserve the original content of the content image in the generated image.  
It is simply the mean squared difference between the target and content features.

More formally: Given a content image **C**, a target (generated) image **G** and 
a layer **L** the *Content Loss* is defined as

$$Loss_{content}(C,G,L) = \frac{1}{D_LH_LW_L}\sum_{ij}{(a[L](C)_{ij} - a[L](G)_{ij})^2}$$

where $a[L](C)_{ij}$ is the is the activation of the $i$-th filter at position $j$ in layer $L$ when passing 
an image $C$. Same for $a[L](G)_{ij}$.  
$D_L$ - Depth/number channels of the output of layer $L$  
$H_L$ - Height of the output of layer $L$  
$W_L$ - Width of the output of layer $L$

Before implementing the loss, let's implement a function that extracts features given a model and an image.

The paper suggests extracting features from several convolutional layers.  
Layer 21 will be used for the Content Loss and Layers 0, 5, 10, 19, 28 will be used for the Style Loss 
(which we will implement in the next section).
"""


def __get_features(net, image_tensor):
    """
    The function runs a forward pass given an image and extracts the features for
    several conv layers. It returns a dictionary where the keys are the
    layers name and the values are the features.
    """
    layers_idx = ['0', '5', '10', '19', '21', '28']
    features = {}
    for i, layer in enumerate(net.features):
        image_tensor = layer.forward(image_tensor)
        if str(i) in layers_idx:
            features[str(i)] = image_tensor
    return features


def __calculate_content_loss(target_features, content_features, layer_id='21'):
    t = target_features[layer_id]
    c = content_features[layer_id]
    return F.mse_loss(t, c)


"""##### **2.2 Style Loss**"""


def __get_style_weights(style_image, prop, device):
    res = torch.max(style_image) - style_image
    res /= torch.max(res)
    res = torch.exp(res)
    res = torch.pow(res, 1.5)
    return res.to(device) * prop


def __new_style_loss(target_image, style_image, prop, device):
    style_w = __get_style_weights(style_image, prop, device)
    return F.mse_loss(torch.mul(target_image, style_w), torch.mul(style_image, style_w))


"""##### **2.3 Total Loss & Gradient Descent**

Now that we have both content and style loss we can add them up and perform gradient descent to change
the generated image such that it decreases its loss after each iteration.

The total loss would be:  

$$Loss_{total} = Loss_{content}(C,G) + prop * Loss_{style}(S,G)$$

where alpha and beta are hyperparameters that weights each loss to control the tradeoff between content and style.
"""


def style_transfer(vgg_model, content_tensor, style_tensor,
                   prop, device, n_iterations, learning_rate):
    """
    The function runs the style transfer algorithm using a pretrained and freezed vgg_model,
    a content image tensor and style image tensor. It weights the content loss with alpha
    and style loss with beta. It runs for n_iterations.
    """
    # creating a random image and set requires_grad to True
    target_image = torch.randn_like(content_tensor).requires_grad_(True).to(device)
    # extract content features
    content_features = __get_features(vgg_model, content_tensor)
    # create optimizer to optimize the target image
    optimizer = torch.optim.Adam([target_image], lr=learning_rate)
    for i in range(n_iterations):
        optimizer.zero_grad()

        target_features = __get_features(vgg_model, target_image)
        content_loss = __calculate_content_loss(content_features, target_features, "10")
        style_loss = __new_style_loss(target_image, style_tensor, prop, device)
        total_loss = content_loss + style_loss

        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(
                f"Iteration {i}, Total Loss: {total_loss.item():.2f}, Content Loss: {content_loss.item():.2f}"
                f", Style  Loss {style_loss.item():.2f}")

    return target_image
