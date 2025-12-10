"""
Continual Learning Training Script for Neural Machine Translation

This script implements continual learning techniques including:
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- Knowledge Distillation to preserve previous task performance
- PyTorch Lightning integration for streamlined training

Usage:
    python clearn.py --config config/preliminary.yaml
    python clearn.py --config config/ewc.yaml --model path/to/model --fim path/to/fisher.pth
"""

import os
import torch
import torch.nn as nn
import lxml.etree as ET
import pytorch_lightning as pl
from operator import attrgetter
from einops import rearrange
from sacrebleu import corpus_bleu
from safetensors.torch import load_file
from configargparse import ArgParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_args():
    parser = ArgParser(description="Continual Learning Training Script for NMT")
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--model', type=str, default=None, help='Path to load model checkpoint from')
    parser.add_argument('--src_train', type=str, required=True, help='Path to source training data')
    parser.add_argument('--tgt_train', type=str, required=True, help='Path to target training data')
    parser.add_argument('--size', type=int, default=-1, help='Number of samples to use from the train dataset (-1 for all)')
    parser.add_argument('--src_val', type=str, required=True, help='Path to source validation data')
    parser.add_argument('--tgt_val', type=str, required=True, help='Path to target validation data')
    parser.add_argument('--teacher', type=str, default=None, help='Path to teacher model checkpoint for distillation')
    parser.add_argument('--distillation', type=float, default=0.0, help='Distillation factor')
    parser.add_argument('--fim', type=str, default=None, help='Path to Fisher Information Matrix file for EWC')
    parser.add_argument('--ewc', type=float, default=0.0, help='EWC factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of steps to train')
    parser.add_argument('--max_time', type=int, default=None, help='Maximum training time in seconds')
    parser.add_argument('--val_check_interval', default=1, help='Validation check interval')
    parser.add_argument('--experiment', type=str, default='clearn_experiment', help='Experiment name')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

class CLLoss(torch.nn.Module):
    """
    Continual Learning Loss combining Cross-Entropy, EWC, and Knowledge Distillation.
    
    The total loss is computed as:
        L_total = L_CE + (λ_EWC / 2) * L_EWC + λ_distill * L_KD
    
    Where:
        - L_CE: Cross-entropy loss on current task
        - L_EWC: Elastic Weight Consolidation regularization
        - L_KD: Knowledge distillation (KL divergence from teacher)
    """
    
    def __init__(self, model, fisher, distillation_factor=0.0, ewc_factor=0.0):
        """
        Initialize the continual learning loss function.
        
        Args:
            model: Teacher model for knowledge distillation (can be None if distillation_factor=0)
            fisher: Dictionary mapping parameter names to Fisher Information Matrix values (can be None if ewc_factor=0)
            distillation_factor (float): Weight for knowledge distillation loss (λ_distill)
            ewc_factor (float): Weight for EWC regularization (λ_EWC)
        """
        super(CLLoss, self).__init__()
        self.model = model
        self.fisher = fisher
        self.distillation_factor = distillation_factor
        self.ewc_factor = ewc_factor
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def to(self, device):
        """
        Move the loss module and Fisher Information Matrix to the specified device.
        
        Args:
            device: Target device (e.g., 'cuda' or 'cpu')
            
        Returns:
            self: Returns the module instance for method chaining
        """
        super().to(device)
        if self.fisher is not None:
            self.fisher = {k: v.to(device) for k, v in self.fisher.items()}
        return self

    def forward(self, src, pred, ref, params):
        """
        Compute the combined continual learning loss.
        
        Args:
            src: Tokenized source inputs (BatchEncoding from tokenizer)
            pred: Model predictions (logits) with shape (batch, sequence, vocab_size)
            ref: Reference labels with shape (batch, sequence)
            params: Iterator of (name, parameter) tuples from model.named_parameters()
            
        Returns:
            torch.Tensor: Combined loss value
        """
        ref = rearrange(ref, 'b s -> (b s)')
        pred = rearrange(pred, 'b s c -> (b s) c')
        # Compute standard cross-entropy loss on current task
        loss = self.ce(pred, ref)

        # Compute EWC regularization loss if enabled
        ewc_loss = torch.tensor(0.0, device=pred.device)
        if self.ewc_factor > 0.0:
            for name, param in params:
                if name in self.fisher:
                    fisher_matrix = self.fisher[name]
                    old_param = attrgetter(name)(self.model)
                    ewc_loss += (fisher_matrix * (param - old_param.detach())**2).sum()
                    #print(name, ewc_loss.item())

        # Compute knowledge distillation loss if enabled
        if self.distillation_factor > 0.0:
            teacher_pred = self.model(**src.to(pred.device)).logits
            teacher_pred = rearrange(teacher_pred, 'b s c -> (b s) c')
            student_log_probs = torch.log_softmax(pred, dim=-1)
            teacher_probs = torch.softmax(teacher_pred, dim=-1)
            distill_loss = self.kl(student_log_probs, teacher_probs)
        else:
            distill_loss = torch.tensor(0.0, device=pred.device)

        # Combine all loss components
        loss = loss \
            + (self.ewc_factor / 2) * ewc_loss \
            + self.distillation_factor * distill_loss
        # print(f"Primary Loss: {loss.item()}")
        # print(f"EWC Loss: {ewc_loss.item()}")
        # print(f"Distillation Loss: {distill_loss.item()}")
        # print(f"Combined Loss: {loss.item()}")
        return loss

class TranslationData(torch.utils.data.Dataset):
    """
    PyTorch Dataset for parallel translation data.
    
    Supports two input formats:
        1. Moses format: Plain text files with one sentence per line
        2. XML format: TMX-style XML with DOC and SEG tags
    """
    
    def __init__(self, src_path, tgt_path, size=-1):
        """
        Initialize the translation dataset.
        
        Args:
            src_path (str): Path to source language file
            tgt_path (str): Path to target language file
            size (int): Number of samples to load (-1 for all)
        """
        self.src = self.parse_moses(src_path, size)
        self.tgt = self.parse_moses(tgt_path, size)
    
    def parse_moses(self,file,size):
        """
        Parse Moses format file and return the segments.
        
        Moses format is a simple plain text format with one sentence per line.
        
        Args:
            file (str): Path to the Moses format file
            size (int): Maximum number of lines to read (-1 for all)
            
        Returns:
            list: List of text segments (strings)
        """
        lines = []
        with open(file, 'r', encoding='utf-8') as f:
            count = size
            while count != 0:
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
                count -= 1
        return lines
    
    def parse_xml(self,file):
        """
        Parse XML file (TMX-style) and return the segments.
        
        Extracts text from <SEG> tags within <DOC> elements.
        Uses XML recovery mode to handle malformed XML.
        
        Args:
            file (str): Path to the XML file
            
        Returns:
            list: List of text segments (strings)
            
        Raises:
            IOError: If file is not found
            SyntaxError: If XML is not well-formed and cannot be recovered
        """
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
        """
        Get a single translation pair.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary with 'src' and 'tgt' keys containing source and target text
        """
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]
        return {
            'src': src_text,
            'tgt': tgt_text
        }

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for managing training and validation datasets.
    
    Handles data loading, dataset creation, and dataloader configuration.
    """
    
    def __init__(self, 
                    src_train_path,
                    tgt_train_path,
                    src_val_path,
                    tgt_val_path, 
                    tokenizer,
                    size=-1,
                    batch_size=32):
        """
        Initialize the DataModule.
        
        Args:
            src_train_path (str): Path to source training data
            tgt_train_path (str): Path to target training data
            src_val_path (str): Path to source validation data
            tgt_val_path (str): Path to target validation data
            tokenizer: HuggingFace tokenizer instance
            size (int): Number of samples to use (-1 for all)
            batch_size (int): Batch size for dataloaders
        """
        super().__init__()
        self.train_paths = (src_train_path, tgt_train_path)
        self.size = size
        self.val_paths = (src_val_path, tgt_val_path)
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage (str, optional): Current stage ('fit', 'validate', 'test', or 'predict')
        """
        self.train_dataset = TranslationData(self.train_paths[0], self.train_paths[1], self.size)
        self.val_dataset = TranslationData(self.val_paths[0], self.val_paths[1], self.size)

    def train_dataloader(self):
        """
        Create training dataloader with shuffling enabled.
        
        Returns:
            DataLoader: PyTorch DataLoader for training
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Create validation dataloader (no shuffling).
        
        Returns:
            DataLoader: PyTorch DataLoader for validation
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)


class LitModel(pl.LightningModule):
    """
    PyTorch Lightning Module for sequence-to-sequence translation with continual learning.
    
    Integrates:
        - Transformer-based seq2seq model
        - Continual learning loss (EWC + Knowledge Distillation)
        - BLEU score evaluation
        - TensorBoard logging
    """
    
    def __init__(self, 
                 model,
                 tokenizer,
                 teacher,
                 fim,
                 lr=0.001,
                 distillation_factor=0.0,
                 ewc_factor=0.0,
                 **kwargs):
        """
        Initialize the Lightning Module.
        
        Args:
            model: Student model (HuggingFace Seq2Seq model)
            tokenizer: HuggingFace tokenizer
            teacher: Teacher model for knowledge distillation (None if not using distillation)
            fim: Fisher Information Matrix dictionary (None if not using EWC)
            lr (float): Learning rate for optimizer
            distillation_factor (float): Weight for distillation loss (λ_distill)
            ewc_factor (float): Weight for EWC loss (λ_EWC)
            **kwargs: Additional hyperparameters to log
        """
        super(LitModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.criterion = CLLoss(model=self.teacher, 
                                fisher=fim, 
                                distillation_factor=distillation_factor, 
                                ewc_factor=ewc_factor)
        self.lr = lr
        self.distillation_factor = distillation_factor
        self.save_hyperparameters(ignore=['model', 'tokenizer', 'teacher', 'fim'])
        self.preds = []
        self.refs = []

    def forward(self, model, tok_inputs):
        """
        Forward pass through the model.
        
        Args:
            model: The seq2seq model to use
            tok_inputs: Tokenized inputs (BatchEncoding with input_ids, attention_mask, labels)
            
        Returns:
            tuple: (logits, labels) where logits have shape (batch, sequence, vocab_size)
        """
        output = model(**tok_inputs.to(self.device))
        return output.logits, tok_inputs['labels']

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.
        
        Args:
            batch (dict): Batch containing 'src' and 'tgt' text
            batch_idx (int): Index of the current batch
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        tok_inputs = self.tokenizer(batch['src'], text_target=batch['tgt'], return_tensors='pt', padding='max_length', truncation=True)
        pred, labels = self(self.model, tok_inputs)
        loss = self.criterion(tok_inputs, 
                              pred, 
                             labels, 
                             params=self.model.named_parameters())
        self.log('train_loss', loss, batch_size=len(batch['src']))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Execute one validation step.
        
        Computes:
            1. Cross-entropy loss (without EWC/distillation)
            2. Generated translations for BLEU score calculation
        
        Args:
            batch (dict): Batch containing 'src' and 'tgt' text
            batch_idx (int): Index of the current batch
            dataloader_idx (int): Index of the dataloader (for multiple validation sets)
        """
        tok_inputs = self.tokenizer(batch['src'], text_target=batch['tgt'], return_tensors='pt', padding='max_length', truncation=True)
        pred, labels = self(self.model, tok_inputs)
        pred = rearrange(pred, 'b s c -> (b s) c')
        labels = rearrange(labels, 'b s -> (b s)')
        loss = self.criterion.ce(pred, labels)
        self.log(f'val_loss', loss, batch_size=len(batch['src']))

        tok_inputs = self.tokenizer(batch['src'],  return_tensors='pt', padding='max_length', truncation=True)
        pred = self.model.generate(**tok_inputs.to(self.device))

        pred = self.tokenizer.batch_decode(pred.detach(), skip_special_tokens=True)
        self.preds.extend(pred)
        self.refs.extend(batch['tgt'])
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        
        Computes BLEU score from accumulated predictions and references,
        then clears the accumulation lists for the next epoch.
        """
        bleu = corpus_bleu(self.preds, [self.refs]).score
        self.log('val_BLEU', bleu)
        self.preds = []
        self.refs = []

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with configured learning rate
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def load_model(path):
    """
    Load a HuggingFace Seq2Seq model and tokenizer.
    
    Args:
        path (str): Path to model checkpoint or HuggingFace model name
        
    Returns:
        tuple: (model, tokenizer) - Model and tokenizer instances
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok

def main():
    """
    Main training function.
    
    Workflow:
        1. Parse arguments from command line and config files
        2. Load teacher model (if using distillation)
        3. Load Fisher Information Matrix (if using EWC)
        4. Initialize student model and data module
        5. Configure PyTorch Lightning trainer with callbacks
        6. Train the model
        7. Save final model checkpoint
    """
    args = parse_args()
    
    # Load teacher model if using knowledge distillation
    if args.teacher:
        teacher_model, _ = load_model(args.teacher)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    
    # Load Fisher Information Matrix if using EWC
    fim = torch.load(args.fim) if args.fim else None,
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    model, tokenizer = load_model(args.model)
    model.train()
    
    data_module = DataModule(src_train_path=args.src_train,
                             tgt_train_path=args.tgt_train,
                             src_val_path=args.src_val,
                             tgt_val_path=args.tgt_val,
                             tokenizer=tokenizer,
                             size=args.size,
                             batch_size=args.batch_size)
    
    # Wrap model in Lightning Module with continual learning components
    model = LitModel(model=model, 
                        tokenizer=tokenizer,
                        teacher=teacher_model if args.teacher else None,
                        distillation_factor=args.distillation,
                        fim=fim,
                        ewc_factor=args.ewc,
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
    #trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    torch.save(model.model.state_dict(), os.path.join("logs", args.experiment, "final_model.pth"))

if __name__ == "__main__":
    main()