import pytorch_lightning as pl
import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(2, 2), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 2),
            )
    def forward(self, input):
        latent = self.encoding_block(input)
        return latent


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoding_block = torch.nn.Sequential(
            torch.nn.Linear(2, 1568),
            torch.nn.Unflatten(1, torch.Size([32, 7, 7])),
            torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=(2, 2), padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2), padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(1),
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
        self.unreduced_loss = torch.nn.MSELoss(reduction="none")
    
    def forward(self, image):
        latent = self.encoder(image)
        return latent

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        latent = self(image)
        reconstructed = self.decoder(latent)
        unreduced_loss = torch.sqrt(torch.mean(self.unreduced_loss(image, reconstructed), dim=(-1, -2))) # reduce spatial dims only
        loss = torch.sqrt(self.loss(image, reconstructed))
        output_dict = {
            "image": image.detach(),
            "val_unreduced_loss": unreduced_loss.detach(),
            "val_loss": loss.detach(),
            "reconstructed": reconstructed.detach(),
            "latent": latent.detach(),
        }
        return output_dict

    def training_step(self, batch, batch_idx):
        image, _ = batch
        latent = self(image)
        reconstructed = self.decoder(latent)
        loss = torch.sqrt(self.loss(image, reconstructed))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.learning_rate)
