import os
import torch
import torch.nn as nn
from operator import attrgetter
from einops import rearrange
import pytorch_lightning as pl
from safetensors.torch import load_file
from configargparse import ArgParser
from torchvision.models import vgg16
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    Grayscale,
)

def parse_args():
    parser = ArgParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--distillation', type=float, default=0.0, help='Distillation factor')
    parser.add_argument('--ewc', type=float, default=0.0, help='EWC factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
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
    parser.add_argument('--model', type=str, default=None, help='Path to load model checkpoint from')
    parser.add_argument('--teacher', type=str, default=None, help='Path to teacher model checkpoint for distillation')
    parser.add_argument('--fim', type=str, default=None, help='Path to Fisher Information Matrix file for EWC')
    return parser.parse_args()

class CLLoss(torch.nn.Module):
    def __init__(self, model, fisher, distillation_factor=0.0, ewc_factor=0.0):
        super(CLLoss, self).__init__()
        self.model = model
        self.fisher = fisher
        self.distillation_factor = distillation_factor
        self.ewc_factor = ewc_factor
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def to(self, device):
        super().to(device)
        if self.fisher is not None:
            self.fisher = {k: v.to(device) for k, v in self.fisher.items()}
        return self

    def forward(self, src, pred, ref, params):
        #ref = rearrange(ref, 'b s -> (b s)')
        #pred = rearrange(pred, 'b s c -> (b s) c')
        loss = self.ce(pred, ref)

        ewc_loss = torch.tensor(0.0, device=pred.device)
        if self.ewc_factor > 0.0:
            for name, param in params:
                if name in self.fisher:
                    fisher_matrix = self.fisher[name]
                    old_param = attrgetter(name)(self.model)
                    ewc_loss += (fisher_matrix * (param - old_param.detach())**2).sum()
                    #print(name, ewc_loss.item())

        if self.distillation_factor > 0.0:
            teacher_pred = self.model(src)
            #teacher_pred = rearrange(teacher_pred, 'b s c -> (b s) c')
            student_log_probs = torch.log_softmax(pred, dim=-1)
            teacher_probs = torch.softmax(teacher_pred, dim=-1)
            distill_loss = self.kl(student_log_probs, teacher_probs)
        else:
            distill_loss = torch.tensor(0.0, device=pred.device)

        loss = loss \
            + (self.ewc_factor / 2) * ewc_loss \
            + self.distillation_factor * distill_loss
        # print(f"Primary Loss: {loss.item()}")
        # print(f"EWC Loss: {ewc_loss.item()}")
        # print(f"Distillation Loss: {distill_loss.item()}")
        # print(f"Combined Loss: {loss.item()}")
        return loss

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize VGG16 with 10 output classes (for MNIST)
        # We don't use pre-trained weights as we are training from scratch on MNIST
        self.vgg = vgg16(num_classes=20)
        
    def forward(self, pixel_values, labels=None):
        logits = self.vgg(pixel_values)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)
        return logits

class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = Compose([
                                Resize((224, 224)),
                                ToTensor(),
                                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}
    
class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, root, train=True):
            self.dataset = MNIST(root=root, train=train, download=True)
            self.transform = Compose([
                                    Resize((224, 224)),
                                    Grayscale(num_output_channels=3),
                                    ToTensor(),
                                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            return {"pixel_values": image, "labels": label}

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, train_size=800, val_size=100, test_size=100):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def setup(self, stage=None):
        self.train_dataset = CIFARDataset(root="./data", train=True)
        self.new_val_dataset = CIFARDataset(root="./data", train=False)
        self.old_val_dataset = MNISTDataset(root="./data", train=False)
        self.new_test_dataset = CIFARDataset(root="./data", train=False)
        self.old_test_dataset = MNISTDataset(root="./data", train=False)

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
                 teacher,
                 fim,
                 lr=0.001,
                 distillation_factor=0.0,
                 ewc_factor=0.0,
                 **kwargs):
        super(LitModel, self).__init__()
        self.model = model
        self.teacher = teacher
        self.criterion = CLLoss(model=self.teacher, 
                                fisher=fim, 
                                distillation_factor=distillation_factor, 
                                ewc_factor=ewc_factor)
        self.lr = lr
        self.distillation_factor = distillation_factor
        self.save_hyperparameters(ignore=['model', 'teacher', 'fim'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        src, ref = batch['pixel_values'], batch['labels']
        if self.teacher:
            with torch.no_grad():
                teacher_pred = self.teacher(src)
                teacher_pred = teacher_pred[:, -10:]  # Adjusting output for 10 classes
        else:
            teacher_pred = None
        pred = self(src)
        pred = pred[:, -10:]  # Adjusting output for 10 classes
        loss = self.criterion(src, pred, ref, self.model.named_parameters())
        self.log('train_loss', loss)
        return loss
    
    def shared_step(self, batch, dataloader_idx):
        src, ref = batch['pixel_values'], batch['labels']
        pred = self(src)
        pred = pred[:, -10:]  if dataloader_idx == 0 else pred[:, :10]  # Adjusting output for 10 classes
        loss = self.criterion.ce(pred, ref)
        acc = (pred.argmax(dim=-1) == ref).float().cpu().mean()

        return loss, acc

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.shared_step(batch, dataloader_idx)
        self.log(f'val_loss', loss)
        self.log(f'val_accuracy', acc)

    def test_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.shared_step(batch, dataloader_idx)
        self.log(f'test_loss', loss)
        self.log(f'test_accuracy', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    args = parse_args()
    if args.teacher:
        teacher_model = VGG()
        teacher_model.load_state_dict(load_file(args.teacher))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    
    # seed everthing
    pl.seed_everything(42)
    data_module = DataModule(batch_size=args.batch_size, train_size=args.size)

    model = VGG()
    if args.model:
        model.load_state_dict(load_file(args.model))
    model = LitModel(model=model, 
                        teacher=teacher_model if args.teacher else None,
                        distillation_factor=args.distillation,
                        ewc_factor=args.ewc,
                        fim=torch.load(args.fim) if args.fim else None,
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
                         accumulate_grad_batches=args.accum_steps,
                         logger=logger, 
                         callbacks=callbacks,
                         num_sanity_val_steps=0)
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    torch.save(model.model.state_dict(), os.path.join("logs", args.experiment, "final_model.pth"))

if __name__ == "__main__":
    main()