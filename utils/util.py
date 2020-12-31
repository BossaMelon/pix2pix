import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def crop(image, new_shape):
    """
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    """
    # There are many ways to implement this crop function, but it's what allows
    # the skip connection to function as intended with two differently sized images!
    _, _, height_orig, width_orig = image.shape
    _, _, height_new, width_new = new_shape
    height_begin = (height_orig - height_new) // 2
    width_begin = (width_orig - width_new) // 2
    cropped_image = image[:, :, height_begin:height_begin + height_new, width_begin:width_begin + width_new]
    return cropped_image
