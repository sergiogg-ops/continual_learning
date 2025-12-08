import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file
from clearn import MNISTDataset, VGG
from torch.utils.data import DataLoader, Dataset
from configargparse import ArgParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = ArgParser(description="Compute Fisher Information Matrix for a given model and dataset")
    #arser.add_argument('src_path', type=str, help='Path to the source text file')
    #arser.add_argument('tgt_path', type=str, help='Path to the target text file')
    parser.add_argument('output', type=str, help='Path to save the computed Fisher Information Matrix')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-fr-en', help='Pretrained model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    return parser.parse_args()

class MosesDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer):
        self.src, self.tgt = [], []
        with open(src_path, 'r') as file:
            self.src = file.readlines()
        with open(tgt_path, 'r') as file:
            self.tgt = file.readlines()
        assert len(self.src) == len(self.tgt), f"Source and target files must have the same number of lines but got {len(self.src)} and {len(self.tgt)}"
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        tok_src = self.tokenizer(self.src[idx].strip(), return_tensors='pt', padding='max_length')
        tok_tgt = self.tokenizer(self.tgt[idx].strip(), return_tensors='pt', padding='max_length')
        return {
            'src': self.src[idx].strip(),
            'tgt': self.tgt[idx].strip(),
            'input_ids': tok_src['input_ids'].squeeze(0),
            'attention_mask': tok_src['attention_mask'].squeeze(0),
            'labels': tok_tgt['input_ids'].squeeze(0)
        }

def get_fisher_diagonal(model, data_loader):
    """
    Computes the diagonal of the Empirical Fisher Information Matrix.
    
    Args:
        model (nn.Module): The model to compute FIM for.
        data_loader (DataLoader): DataLoader for the task data.
        
    Returns:
        dict: A dictionary mapping parameter names to their Fisher importance (squared gradient average).
    """
    # 2. Initialize a dictionary to accumulate the squared gradients
    fisher_diag = {}
    for name, param in model.named_parameters():
        fisher_diag[name] = torch.zeros_like(param.data)

    model.eval()
    
    # 3. Iterate over the dataset to accumulate squared gradients
    num_samples = 0
    for batch in tqdm(data_loader, desc="Computing FIM diagonal", unit="batch",total=len(data_loader)):
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        model.zero_grad()
        num_samples += inputs.size(0)

        outputs = model(inputs)

        # Note: The gradient must be calculated for *each* sample,
        # so we calculate the loss per sample and sum it.
        loss_vector = F.cross_entropy(outputs, labels, reduction='none')
        
        # We calculate the gradient of the mean loss for the batch
        # For classification, this is a standard and effective approximation.
        mean_loss = loss_vector.mean()
        mean_loss.backward()

        # 4. Accumulate the squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Add the squared gradient of the batch mean loss.
                # The total accumulation will be averaged later.
                fisher_diag[name] += param.grad.data.pow(2)
                
    # 5. Final Averaging
    for name in fisher_diag:
        fisher_diag[name] /= num_samples

    return fisher_diag

def main():
    args = parse_args()
    
    # Load model and tokenizer
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VGG()
    model.load_state_dict(load_file(args.model))
    model.to(device)
    
    # Prepare dataset and dataloader
    num_workers = os.cpu_count()-1 if os.cpu_count() > 1 else 1
    dataset = MNISTDataset(root="./data", train=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Compute Fisher Information Matrix diagonal
    fisher_diag = get_fisher_diagonal(model, data_loader)
    
    # Save the Fisher Information Matrix diagonal
    torch.save(fisher_diag, args.output)
    print(f"Fisher Information Matrix diagonal saved to {args.output}")

if __name__ == "__main__":
    main()