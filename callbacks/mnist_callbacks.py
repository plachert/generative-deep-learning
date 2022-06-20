from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import PIL


def fig2png(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png = buf.getvalue()
    decoded_png = np.array(PIL.Image.open(io.BytesIO(png)))
    return decoded_png


class VisualizeCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.val_batch_outputs = []
        self.labels = []

    def aggregate_and_clean(self):
        aggregated = {}
        for batch_dict in self.val_batch_outputs:
            for key, val in batch_dict.items():
                if val.size(): # skip reduced loss
                    try:
                        aggregated[key].append(val)
                    except KeyError:
                        aggregated[key] = [val]
        for key, val in aggregated.items():
            aggregated[key] = torch.cat(val, axis=0)
            aggregated[key] = np.squeeze(aggregated[key].cpu().numpy())
        self.val_batch_outputs = []
        return aggregated

    def make_worst_reconstruction_plot(self, aggregated):
        k = 5
        worst_indices = np.argsort(aggregated["val_unreduced_loss"]) [-k:]
        org_images = aggregated["image"][worst_indices]
        reconstructed_images = aggregated["reconstructed"][worst_indices]
        fig, axs = plt.subplots(2, k, figsize=(10, 10))
        for i in range(k):
            axs[0, i].imshow(org_images[i, :, :])
            axs[1, i].imshow(reconstructed_images[i, :, :])
        reconstruction_png = fig2png(fig)
        return reconstruction_png

    def make_latent_space_plot(self, aggregated):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        latent = aggregated["latent"]
        ax.scatter(x=latent[:, 0], y=latent[:, 1], c=self.labels)
        latent_png = fig2png(fig)
        return latent_png

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, labels = batch
        self.val_batch_outputs.append(outputs)
        self.labels.extend(labels)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        aggregated = self.aggregate_and_clean()
        worst_reconstruction_plot = self.make_worst_reconstruction_plot(aggregated)
        latent_space_plot = self.make_latent_space_plot(aggregated)
        trainer.logger.experiment.add_image(
            "worst_reconstructions",
            worst_reconstruction_plot,
            trainer.current_epoch,
            dataformats="HWC",
        )
        trainer.logger.experiment.add_image(
            "latent_space_plot",
            latent_space_plot,
            trainer.current_epoch,
            dataformats="HWC",
        )