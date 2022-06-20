from data.datamodules import MNISTDataModule
from models.basic_autoencoder import AutoEncoderSystem, Encoder, Decoder
import pytorch_lightning as pl
from callbacks.mnist_callbacks import VisualizeCallback


def main():
    encoder = Encoder()
    decoder = Decoder()
    system = AutoEncoderSystem(encoder, decoder)
    datamodule = MNISTDataModule(
        train_batch_size=512,
        val_batch_size=512,
    )
    vis_callback = VisualizeCallback()
    logging_path = "experiments/test"
    tensorboard_logger = pl.loggers.TensorBoardLogger(logging_path)
    trainer = pl.Trainer(
        callbacks=[vis_callback],
        logger=tensorboard_logger,
        max_epochs=2,
        gpus=1,
        )
    trainer.fit(system, datamodule)


if __name__ == "__main__":
    main()



