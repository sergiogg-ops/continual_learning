import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from configargparse import ArgParser
from torch.nn.functional import binary_cross_entropy

def parse_args():
    parser = ArgParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--distillation', type=float, default=0.0, help='Distillation factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of steps to train')
    parser.add_argument('--max_time', type=int, default=None, help='Maximum training time in seconds')
    parser.add_argument('--val_check_interval', default=1, help='Validation check interval')
    parser.add_argument('--size', type=int, default=1000, help='Size of synthetic dataset')
    parser.add_argument('--experiment', type=str, default='clearn_experiment', help='Experiment name')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--load', type=str, default=None, help='Path to load model checkpoint from')
    parser.add_argument('--teacher', type=str, default=None, help='Path to teacher model checkpoint for distillation')
    return parser.parse_args()

class LossFn(torch.nn.Module):
    def __init__(self, distillation_factor=0.0):
        super(LossFn, self).__init__()
        self.distillation_factor = distillation_factor
        self.bce = nn.BCELoss()

    def forward(self, pred, ref, teacher_pred=None):
        loss = self.bce(pred, ref)
        #print(f"Primary Loss: {loss.item()}")
        if teacher_pred is not None:
            distill_loss = self.bce(pred, teacher_pred)
            #print(f"Distillation Loss: {distill_loss.item()}")
            loss = (1 - self.distillation_factor) * loss + self.distillation_factor * distill_loss
            #print(f"Combined Loss: {distill_loss.item()}")
        return loss

class Model(torch.nn.Sequential):
    def __init__(self):
        super(Model, self).__init__(
            nn.Linear(1,100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

class OldData(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.randn(size, 1)
        self.y = (self.x > 0).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NewData(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.randn(size, 1) * 1000
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
        self.train_dataset = NewData(self.train_size)
        self.new_val_dataset = NewData(self.val_size)
        self.old_val_dataset = OldData(self.val_size)
        self.new_test_dataset = NewData(self.test_size)
        self.old_test_dataset = OldData(self.test_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(self.new_val_dataset, batch_size=self.batch_size),
            torch.utils.data.DataLoader(self.old_val_dataset, batch_size=self.batch_size)
            ]

    def test_dataloader(self):
        return [
            torch.utils.data.DataLoader(self.new_test_dataset, batch_size=self.batch_size),
            torch.utils.data.DataLoader(self.old_test_dataset, batch_size=self.batch_size)
            ]

class LitModel(pl.LightningModule):
    def __init__(self, 
                 model,
                 teacher=None,
                 lr=0.001,
                 distillation_factor=0.0
                 ):
        super(LitModel, self).__init__()
        self.model = model
        self.teacher = teacher
        self.criterion = LossFn(distillation_factor=distillation_factor)
        self.lr = lr
        self.distillation_factor = distillation_factor

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        src, ref = batch
        print()
        if self.teacher:
            with torch.no_grad():
                teacher_pred = self.teacher(src)
        else:
            teacher_pred = None
        pred = self(src)
        loss = self.criterion(pred, ref, teacher_pred)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        src, ref = batch
        pred = self(src)
        loss = self.criterion(pred, ref)
        self.log(f'val_loss', loss)
        
        acc = ((pred > 0.5).float() == ref).float().cpu().mean()
        self.log(f'val_accuracy', acc)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        src, ref = batch
        pred = self(src)
        loss = self.criterion(pred, ref)
        self.log(f'test_loss', loss)
        
        acc = ((pred > 0.5).float() == ref).float().cpu().mean()
        self.log(f'test_accuracy', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    args = parse_args()
    if args.teacher is not None:
        print(f"Loading teacher model from {args.teacher}")
        teacher_model = Model()
        # teacher_model.load_state_dict(torch.load(args.teacher))
        # teacher_model.eval()
        # for param in teacher_model.parameters():
        #     param.requires_grad = False
    
    # seed everthing
    pl.seed_everything(42)
    data_module = DataModule(batch_size=args.batch_size, train_size=args.size)

    model = Model()
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model = LitModel(model=model, 
                        teacher=teacher_model if args.teacher is not None else None,
                        distillation_factor=args.distillation,
                        lr=args.lr)
    logger = pl.loggers.TensorBoardLogger(os.path.join("logs", args.experiment))
    callbacks = []
    if args.early_stop:
        early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_accuracy/dataloader_idx_0', 
                                                         patience=args.patience, 
                                                         mode='max',
                                                         min_delta=0)
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         max_steps=args.max_steps, 
                         max_time=args.max_time,
                         val_check_interval=float(args.val_check_interval),
                         logger=logger, 
                         callbacks=callbacks, num_sanity_val_steps=0)
    #trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    torch.save(model.model.state_dict(), os.path.join("logs", args.experiment, "final_model.pth"))

if __name__ == "__main__":
    main()