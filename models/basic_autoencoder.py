import pytorch_lightning as pl
import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 2),
            )
    def forward(self, input):
        latent = self.encoding_block(input)
        return latent


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoding_block = torch.nn.Sequential(
            torch.nn.Linear(2, 3136),
            torch.nn.Unflatten(1, torch.Size([64, 7, 7])),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
    def forward(self, input):
        reconstructed = self.decoding_block(input)
        return reconstructed


class AutoEncoderSystem(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        learning_rate=0.001,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.loss = torch.nn.MSELoss()
    
    def forward(self, image):
        latent = self.encoder(image)
        return latent

    def training_step(self, batch, batch_idx):
        image, _ = batch
        latent = self(image)
        reconstructed = self.decoder(latent)
        loss = torch.sqrt(self.loss(image, reconstructed))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.learning_rate)


    


decoder = Decoder()
decoder(torch.rand((1, 2)))