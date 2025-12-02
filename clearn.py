import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from configargparse import ArgParser

def parse_args():
    parser = ArgParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of steps to train')
    parser.add_argument('--max_time', type=int, default=None, help='Maximum training time in seconds')
    parser.add_argument('--size', type=int, default=1000, help='Size of synthetic dataset')
    parser.add_argument('--experiment', type=str, default='clearn_experiment', help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

class Model(torch.nn.Sequential):
    def __init__(self):
        super(Model, self).__init__(
            nn.Linear(1,1),
            nn.ReLU(),
            nn.Linear(1,1),
            nn.Sigmoid()
        )

class SynthData(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.randn(size, 1)
        self.y = (self.x > 0).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, train_size=800, val_size=100, test_size=100):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def setup(self, stage=None):
        self.train_dataset = SynthData(self.train_size)
        self.val_dataset = SynthData(self.val_size)
        self.test_dataset = SynthData(self.test_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

class LitModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super(LitModel, self).__init__()
        self.model = Model()
        self.criterion = nn.BCELoss()
        self.lr = lr
        self.preds = torch.tensor([])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        src, ref = batch
        pred = self(src)
        loss = self.criterion(pred, ref)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, ref = batch
        pred = self(src)
        loss = self.criterion(pred, ref)
        self.log('val_loss', loss)
        #self.preds.extend(((pred > 0.5).float() == ref).float().cpu().numpy().tolist())
        self.preds = torch.cat([torch.tensor(self.preds), ((pred > 0.5).float() == ref).float().cpu()])
    
    def on_validation_epoch_end(self):
        accuracy = self.preds.sum() / self.preds.size(0)
        self.log('val_accuracy', accuracy)
        self.preds = []

    def test_step(self, batch, batch_idx):
        src, ref = batch
        pred = self(src)
        loss = self.criterion(pred, ref)
        self.log('test_loss', loss)
        self.preds = torch.cat([torch.tensor(self.preds), ((pred > 0.5).float() == ref).float().cpu()])

    def on_test_epoch_end(self):
        accuracy = self.preds.sum() / self.preds.size(0)
        self.log('test_accuracy', accuracy)
        self.preds = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    args = parse_args()
    # seed everthing
    pl.seed_everything(42)
    data_module = DataModule(batch_size=args.batch_size, train_size=args.size)
    model = LitModel(lr=args.lr)
    logger = pl.loggers.TensorBoardLogger(os.path.join("logs", args.experiment))
    trainer = pl.Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, max_time=args.max_time, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()