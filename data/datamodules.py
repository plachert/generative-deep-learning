import pytorch_lightning as pl
import torchvision.datasets as vision_datasets
from torchvision import transforms
from utils import get_project_root
from typing import Optional
from torch.utils.data import random_split, DataLoader
import matplotlib
matplotlib.use('TkAgg')

downloaded_data_path = get_project_root() / "data" / "downloaded" 


class MNISTDataModule(pl.LightningDataModule):

    path = downloaded_data_path / "MNIST"

    def __init__(
        self, 
        train_batch_size=32, 
        val_batch_size=32,
        ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = 1
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def prepare_data(self):
        vision_datasets.MNIST(self.path, train=True, download=True)
        vision_datasets.MNIST(self.path, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            mnist_full = vision_datasets.MNIST(self.path, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.test_set = vision_datasets.MNIST(self.path, train=False, transform=self.transform)
        if stage == "predict" or stage is None:
            self.predict_set = vision_datasets.MNIST(self.path, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=8)


if __name__ == "__main__":
    data = MNISTDataModule()
    data.prepare_data()
    data.setup()
    x, y  = data.train_set[0]
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(np.squeeze(x.numpy()))
    plt.show()