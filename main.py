import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from utils.trainutils import count_parameters_layerwise, save_checkpoint
from modeling.model import MoEModel
from modeling.model_config import ModelConfig
from cut_cross_entropy import linear_cross_entropy

torch.set_float32_matmul_precision('high')

class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx]}

def load_and_preprocess_data(max_length=255):
    dataset = load_dataset("skeskinen/TinyStories-hf", split="train[:5%]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_input_ids = []
    for text in tqdm(dataset["text"]):
        text = text.strip()
        if not text:
            continue
        tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
        if len(tokens) >= max_length:
            continue
        sequence = tokens
        if sequence.size(0) < max_length:
            padding = torch.full((max_length - sequence.size(0),), tokenizer.eos_token_id, dtype=torch.long)
            sequence = torch.cat([sequence, padding])
        all_input_ids.append(sequence)

    input_ids_tensor = torch.stack(all_input_ids)
    return TextDataset(input_ids_tensor), tokenizer

# Auxillary loss free routing: https://arxiv.org/abs/2408.15664
# TLDR is that this does a direct update (not via backward) to the expert biases which will tip towards underfilled experts.
def auxillary_loss_free_update(model, all_topk_indices, update_rate):
    with torch.no_grad():
        for layer_idx, topk_idx in enumerate(all_topk_indices):
            expert_counts = torch.bincount(
                topk_idx.flatten(),
                minlength=model.layers[layer_idx].mlp.gate.n_routed_experts
            )
            avg_count = expert_counts.float().mean()
            for expert_idx, count in enumerate(expert_counts):
                error = avg_count - count.float()
                model.layers[layer_idx].mlp.gate.expert_biases[expert_idx] += update_rate * torch.sign(error)

def train(model, train_dataset, tokenizer, num_epochs=10, batch_size=64, learning_rate=1e-4, update_rate=1e-4):
    device = torch.device("cuda")
    model.to(device)

    # optimizer = DistributedShampoo(
    #     model.parameters(),
    #     lr=learning_rate,
    #     betas=(0.9, 0.999),
    #     epsilon=1e-4,
    #     grafting_config=AdamPreconditionerConfig(
    #         beta2=0.999,
    #         epsilon=1e-4,
    #     ),
    # )

    rmsnorm_weightdecayed = {'params':[p for name, p in model.named_parameters() if 'rmsnorm.weight' in name], 'weight_decay': 1e-4}
    others = {'params':[p for name, p in model.named_parameters() if 'rmsnorm.weight' not in name]}
    optimizer = torch.optim.Adam([rmsnorm_weightdecayed, others], lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    scaler = torch.amp.GradScaler("cuda")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings, all_topk_indices  = model.headless_forward(input_ids)
                classifier = model.get_classifier_weights()
                loss = linear_cross_entropy(embeddings, classifier, input_ids, shift=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            auxillary_loss_free_update(model, all_topk_indices, update_rate)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.safetensors"
        save_checkpoint(model, optimizer, str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    config = ModelConfig()
    train_dataset, tokenizer = load_and_preprocess_data()

    model = MoEModel(config)

    count_parameters_layerwise(model)
    torch.compile(model, mode="reduce-overhead")
    train(model, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
