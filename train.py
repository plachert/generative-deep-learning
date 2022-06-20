from data.datamodules import MNISTDataModule
from models.basic_autoencoder import AutoEncoderSystem, Encoder, Decoder
from pytorch_lightning import Trainer


def main():
    encoder = Encoder()
    decoder = Decoder()
    system = AutoEncoderSystem(encoder, decoder)
    datamodule = MNISTDataModule()
    trainer = Trainer(max_epochs=1)
    trainer.fit(system, datamodule)


if __name__ == "__main__":
    main()



