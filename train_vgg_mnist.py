import numpy as np
from clearn import VGG, MNISTDataset
from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

def train():
    # 1. Preprocessing

    print("Loading MNIST dataset via torchvision...")
    train_dataset = MNISTDataset(root="./data", train=True)
    eval_dataset = MNISTDataset(root="./data", train=False)
    
    # 3. Model Configuration
    print("Initializing VGG model...")
    model = VGG()

    # 4. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        #output_dir="./vgg_mnist_results",
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, # Save only the last checkpoint
        logging_steps=100,
        learning_rate=1e-4,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    # 6. Save final model
    print("Saving final model...")
    trainer.save_model("./vgg_mnist")
    print("Training completed and model saved to ./vgg_mnist")

if __name__ == "__main__":
    train()
