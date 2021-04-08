# day-to-night-images
The Mission:
Our goal is to create a machine learning model that when given an image which was taken in daylight conditions, and simulates a similar image, as if it was taken during the night, using flash, where the flash is (reasonably) focusing on the main object in the picture.
Note this is not trivial - we can’t simply make the picture darker or do some similar operation. as it would simply look darker and not like it was taken at night.
Our model is focusing on images which consist of one single main object. Images in which there are more than one main object, aren’t guaranteed to be simulated well in night conditions using our model. However, this may be a great future work on our model.


Related Work:
We have found two websites that are based on the same goal,of converting a daylight image to a nighttime one:
https://funny.pho.to/day-to-night-effect/
https://photofunia.com/effects/dark_night
Both of them do not perform to the required quality, and mostly apply a blue filter on the image.
In addition, there is a website that applies a “night vision” filter on the image, and turns it greener and darker: https://www10.lunapic.com/editor/?action=night-vision. The two problems are quite similar and similar approaches could be taken to solve the two.
We have not found any article about our mission, but there has been a lot of work done on the opposite mission, of converting a dark image to a daytime one. An example of such article is: https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn



Our Solution:
We used the dataset from https://www.kaggle.com/biancaferreira/african-wildlife and also some photos of our own (see below).
Our solution is built based on the structure given in https://arxiv.org/abs/1508.06576. From now on we will refer to this article as “the base style transfer”.
Our solution consists of applying style transfer, when the style image is built using object detection and distance mapping, as we discuss below, and tries to represent a dark background with a flash aimed at the main object. While the style transfer tries to minimize the difference between the features (denoted by a pretrained model as explained below) of the original image and the one we create, it also tries to minimize the difference between the customized style image and the image we create. Note that the style loss uses the style image as is and not features of it, as opposed to many usages of style transfer. We do that because the location of elements in the style image matters.
We use the pretrained model vgg19 to denote features of the original image. We use the features denoted by its 10th layer. We calculate the content loss in the style transfer similarly to the calculation marked in the base style transfer.
The composition of the style involves quite a lot of elements. First we use object detection to locate the objects in the picture. We define a heuristic that considers the main object as the biggest one, and use this heuristic to find that main object. Side by side, we use the pretrained model MiDaS to calculate a depth map of the original image. We use this map to create an image with the texture of the original image, but whose colors are on the white-black scale, where closer parts of the image are brighter. Then, by considering each pixel’s distance from the main object, and the main object’s size, we give a weight for each pixel, where the weight is smaller than 1, and multiply the depth image by these weights. The multiplication of the two, creates the effect that brightness of a pixel depends on the distance of it both from the camera (the closer the element is to the flash, the more light hits it) and from the main object (which the flash is focused on).
Finally, similarly to the original style transfer, we set the ratio between the weight given to the content loss and the weight given to the style loss in the calculation of the total loss. However, we also consider that in real photos taken in night conditions, the dark background is usually quite fuzzy (and the elements in it aren’t seen in first look) while the bright parts (where the flash hits) are mostly very clear. Thus, we re-weight the ratio between the style loss considering the brightness of the pixel in the style image - the brighter the pixel, the more weight we give to the content in this pixel.
All of that results in images which  transform to new images that seem to be taken at night, with a flash aimed at the main object and a gradually dark background.
