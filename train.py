import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import ModuleList
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, transforms

from classifier import Classifier
from doe import DOE
from propagate import Propagate
from spatial_coherence import get_exponentially_decaying_spatial_coherence, get_source_modes


class DiffractiveSystem(pl.LightningModule):
    def __init__(self, learning_rate, gamma, coherence_degree, wavelength, pixel_size):
        super().__init__()
        self.save_hyperparameters()
        self.doe_list = ModuleList([DOE(shape=100) for _ in range(5)])
        self.initial_propagate = Propagate(
            preceding_shape=28 * 4,
            succeeding_shape=100,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
        )
        self.intralayer_propagate = Propagate(
            preceding_shape=100,
            succeeding_shape=100,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
        )
        self.classifier = Classifier(shape=100, region_size=25)
        self.source_modes = get_source_modes(shape=28, image_pixel_size=4)
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        coherence_tensor = get_exponentially_decaying_spatial_coherence(
            torch.squeeze(x, -3).to(torch.cdouble), self.hparams.coherence_degree
        )

        modes = self.source_modes
        modes = self.initial_propagate(modes)
        for doe in self.doe_list:
            modes = doe(modes)
            modes = self.intralayer_propagate(modes)

        batch_size = coherence_tensor.shape[0]
        total_input_pixels = coherence_tensor.shape[-2] * coherence_tensor.shape[-1]
        total_output_pixels = modes.shape[-2] * modes.shape[-1]
        output_intensity = (
            torch.einsum(  # Reduce precision to cfloat for performance
                "bij, io, jo-> bo",
                coherence_tensor.view(batch_size, total_input_pixels, total_input_pixels).to(torch.cfloat),
                modes.view(total_input_pixels, total_output_pixels).conj().to(torch.cfloat),
                modes.view(total_input_pixels, total_output_pixels).to(torch.cfloat),
            )
            .real.view(batch_size, *modes.shape[-2:])
            .to(torch.double)
        )
        return self.classifier(output_intensity)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


def main(args):
    torch.manual_seed(args.seed)
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    transform = transforms.Compose(transform_list)
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(
        datasets.MNIST("../data", train=False, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DiffractiveSystem(args.lr, args.gamma, args.coherence_degree, args.wavelength, args.pixel_size)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], accelerator=accelerator)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--coherence-degree", type=float, required=True, help="coherence degree")
    parser.add_argument("--wavelength", type=float, default=700e-9, help="field wavelength (default: 700 nm)")
    parser.add_argument("--pixel-size", type=float, default=10e-6, help="field pixel size (default: 10 um)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for training (default: 32)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate (default: 1e-2)")
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers (default: 1)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate step gamma (default: 0.95)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    main(args)
