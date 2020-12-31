import torch.utils.data
from skimage import io
from torch.utils.data import DataLoader

from utils.path_handle import data_path
from utils.util import crop


def get_dataloader(target_shape=373, batch_size=4):
    volumes = torch.Tensor(io.imread(str(data_path / 'train-volume.tif')))[:, None, :, :] / 255
    labels = torch.Tensor(io.imread(str(data_path / 'train-labels.tif'), plugin="tifffile"))[:, None, :, :] / 255
    labels = crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
    dataset = torch.utils.data.TensorDataset(volumes, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()
    volume, label = dataloader.__iter__().__next__()
    print(volume.shape, label.shape)
