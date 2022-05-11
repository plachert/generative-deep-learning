import pytorch_lightning as pl
import torch

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            )
    def forward(self, input):
        output = self.block(input)
        print(output.shape)


encoder = Encoder()

encoder(torch.rand((1, 1, 28, 28)))