import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils.path_handle import visualization_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()


def save_tensor_images(image_tensor, file_name, num_images=25, size=(1, 28, 28), show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4).permute(1, 2, 0).squeeze().numpy()
    if show:
        plt.imshow(image_grid)
        plt.show()
        return
    file_path = visualization_path / '{}.jpg'.format(file_name)
    plt.imsave(file_path, image_grid)


def image_cat(*args, size, file_name, show=False):
    image_list = []
    for image in args:
        image_unflat = image.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat, nrow=4).permute(1, 2, 0).squeeze()
        image_list.append(image_grid)
    stacked_image = torch.cat(image_list,dim=0).numpy()
    if show:
        plt.imshow(stacked_image)
        plt.show()
        return
    file_path = visualization_path / '{}.jpg'.format(file_name)
    plt.imsave(file_path, stacked_image)



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
