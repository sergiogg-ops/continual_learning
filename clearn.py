import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import lxml.etree as ET
from einops import rearrange
from sacrebleu import corpus_bleu
from configargparse import ArgParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_args():
    parser = ArgParser(description="Continual learning framework for translation models")
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Path to load model checkpoint from which to continue training')
    parser.add_argument('--src_train', type=str, required=True, help='Path to source training data')
    parser.add_argument('--tgt_train', type=str, required=True, help='Path to target training data')
    parser.add_argument('--src_val', type=str, required=True, help='Path to source validation data')
    parser.add_argument('--tgt_val', type=str, required=True, help='Path to target validation data')
    parser.add_argument('--distillation', type=float, default=0.0, help='Distillation factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of steps to train')
    parser.add_argument('--max_time', type=int, default=None, help='Maximum training time in seconds')
    parser.add_argument('--val_check_interval', default=1, help='Validation check interval')
    parser.add_argument('--experiment', type=str, default='clearn_experiment', help='Experiment name')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--teacher', type=str, default=None, help='Path to teacher model checkpoint for distillation')
    return parser.parse_args()

class LossFn(torch.nn.Module):
    def __init__(self, distillation_factor=0.0):
        super(LossFn, self).__init__()
        self.distillation_factor = distillation_factor
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, ref, teacher_pred=None):
        ref = rearrange(ref, 'b s -> (b s)')
        pred = rearrange(pred, 'b s c -> (b s) c')
        loss = self.ce(pred, ref)
        #print(f"Primary Loss: {loss.item()}")
        if teacher_pred is not None:
            teacher_pred = rearrange(teacher_pred, 'b s c -> (b s) c')
            student_log_probs = torch.log_softmax(pred, dim=-1)
            teacher_probs = torch.softmax(teacher_pred, dim=-1)
            distill_loss = self.kl(student_log_probs, teacher_probs)
            #print(f"Distillation Loss: {distill_loss.item()}")
            loss = (1 - self.distillation_factor) * loss + self.distillation_factor * distill_loss
            #print(f"Combined Loss: {loss.item()}")
        return loss

class Model(torch.nn.Sequential):
    def __init__(self, hidden_size=2):
        super(Model, self).__init__(
            nn.Linear(1,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
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
        self.x = torch.randn(size, 1) * 10000
        self.y = (self.x > 0).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TranslationData(torch.utils.data.Dataset):
    def __init__(self, src_path, tgt_path, tokenizer):
        self.src = self.parse_moses(src_path)
        self.tgt = self.parse_moses(tgt_path)

        self.tokenizer = tokenizer
    
    def parse_moses(self,file):
        '''
        Parse the Moses file and return the segments.
        Args:
            file: path to the Moses file
        Returns:
            segments: list of segments
        '''
        with open(file, 'r', encoding='utf-8') as f:
            segments = [line.strip() for line in f.readlines()]
        return segments
    
    def parse_xml(self,file):
        '''
        Parse the XML file and return the segments.
        Args:
            file: path to the XML file
        Returns:
            segments: list of segments
        '''
        try:
            tree = ET.parse(file,  ET.XMLParser(recover=True))
        except IOError:
            raise IOError(f"File {file} not found.")
        except SyntaxError:
            raise SyntaxError(f"File {file} is not well-formed.")
        root = tree.getroot()
        
        segments = []
        for doc in root.findall('DOC'):
            for seg in doc.findall('SEG'):
                text = seg.text.strip() if seg.text is not None else ''
                segments.append(text)
        return segments
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]
        return {
            'src': src_text,
            'tgt': tgt_text
        }

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    src_train_path,
                    tgt_train_path,
                    src_val_path,
                    tgt_val_path, 
                    tokenizer,
                    batch_size=32):
        super().__init__()
        self.train_paths = (src_train_path, tgt_train_path)
        self.val_paths = (src_val_path, tgt_val_path)
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_dataset = TranslationData(self.train_paths[0], self.train_paths[1], self.tokenizer)
        self.val_dataset = TranslationData(self.val_paths[0], self.val_paths[1], self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

class LitModel(pl.LightningModule):
    def __init__(self, 
                 model,
                 tokenizer,
                 teacher=None,
                 lr=0.001,
                 distillation_factor=0.0
                 ):
        super(LitModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.criterion = LossFn(distillation_factor=distillation_factor)
        self.lr = lr
        self.distillation_factor = distillation_factor
        self.preds = []
        self.refs = []
    
    def forward(self, model, batch):
        tok_inputs = self.tokenizer(batch['src'], text_target=batch['tgt'], return_tensors='pt', padding='max_length', truncation=True)
        output = model(**tok_inputs.to(self.device))
        return output.logits, tok_inputs['labels']

    def training_step(self, batch, batch_idx):
        if self.teacher:
            with torch.no_grad():
                teacher_pred, _ = self(self.teacher, batch)
        else:
            teacher_pred = None
        pred, labels = self(self.model, batch)
        loss = self.criterion(pred, labels, teacher_pred)
        self.log('train_loss', loss, batch_size=len(batch['src']))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred, labels = self(self.model, batch)
        loss = self.criterion(pred, labels)
        self.log(f'val_loss', loss, batch_size=len(batch['src']))

        tok_inputs = self.tokenizer(batch['src'],  return_tensors='pt', padding='max_length', truncation=True)
        pred = self.model.generate(**tok_inputs.to(self.device))

        pred = self.tokenizer.batch_decode(pred.detach(), skip_special_tokens=True)
        self.preds.extend(pred)
        self.refs.extend(batch['tgt'])
    
    def on_validation_epoch_end(self):
        bleu = corpus_bleu(self.preds, [self.refs]).score
        self.log('val_BLEU', bleu)
        self.preds = []
        self.refs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    args = parse_args()
    if args.teacher:
        teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    
    # seed everthing
    pl.seed_everything(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    data_module = DataModule(src_train_path=args.src_train,
                             tgt_train_path=args.tgt_train,
                             src_val_path=args.src_val,
                             tgt_val_path=args.tgt_val,
                             tokenizer=tokenizer,
                             batch_size=args.batch_size)
    model = LitModel(model=model, 
                        tokenizer=tokenizer,
                        teacher=teacher_model if args.teacher else None,
                        distillation_factor=args.distillation,
                        lr=args.lr)
    logger = pl.loggers.TensorBoardLogger(os.path.join("logs", args.experiment))
    callbacks = []
    if args.early_stop:
        early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_BLEU', 
                                                         patience=args.patience, 
                                                         mode='max',
                                                         min_delta=0)
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         max_steps=args.max_steps, 
                         max_time=args.max_time,
                         val_check_interval=float(args.val_check_interval),
                         logger=logger, 
                         callbacks=callbacks,
                         num_sanity_val_steps=0,)
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    torch.save(model.model.state_dict(), os.path.join("logs", args.experiment, "final_model.pth"))

if __name__ == "__main__":
    main()