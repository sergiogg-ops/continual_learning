import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from clearn import TranslationData, load_model, LitModel
from torch.utils.data import DataLoader
from configargparse import ArgParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = ArgParser(description="Compute Fisher Information Matrix for a given model and dataset")
    parser.add_argument('src_path', type=str, help='Path to the source text file')
    parser.add_argument('tgt_path', type=str, help='Path to the target text file')
    parser.add_argument('output', type=str, help='Path to save the computed Fisher Information Matrix')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-fr-en', help='Pretrained model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--size', type=int, default=-1, help='Number of samples to use from the dataset (-1 for all)')
    return parser.parse_args()

def get_fisher_diagonal(model, tokenizer, data_loader):
    """
    Computes the diagonal of the Empirical Fisher Information Matrix.
    
    Args:
        model (nn.Module): The model to compute FIM for.
        tokenizer (PreTrainedTokenizerFast): Tokenizer corresponding to the model.
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
        inputs = tokenizer(batch['src'], text_target=batch['tgt'], return_tensors='pt', padding='max_length', truncation=True).to(device)
        model.zero_grad()
        num_samples += inputs['input_ids'].size(0)

        outputs, labels = model(model.model,inputs)

        # Note: The gradient must be calculated for *each* sample,
        # so we calculate the loss per sample and sum it.
        outputs = rearrange(outputs, 'b s c -> (b s) c')
        labels = rearrange(labels, 'b s -> (b s)')
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
    model, tokenizer = load_model(args.model)
    model = LitModel(model=model,
                     tokenizer=tokenizer,
                     teacher=None,
                     fim=None,
                     distillation_factor=0.0,
                     ewc_factor=0.0
                     ).to(device)
    
    # Prepare dataset and dataloader
    num_workers = os.cpu_count()-1 if os.cpu_count() > 1 else 1
    dataset = TranslationData(args.src_path, args.tgt_path, args.size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Compute Fisher Information Matrix diagonal
    fisher_diag = get_fisher_diagonal(model, tokenizer, data_loader)
    
    # Save the Fisher Information Matrix diagonal
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    torch.save(fisher_diag, args.output)
    print(f"Fisher Information Matrix diagonal saved to {args.output}")

if __name__ == "__main__":
    main()