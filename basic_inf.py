import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from modeling.model import MoEModel
from modeling.model_config import ModelConfig
from utils.trainutils import load_checkpoint

class MoEInference:
    def __init__(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = ModelConfig()
        self.model = MoEModel(self.config)

        load_checkpoint(self.model, None, model_path)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            for _ in range(max_length):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type=='cuda'):
                    outputs, _ = self.model(input_ids)

                next_token_logits = outputs[:, -1, :].float()
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                print(next_token)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    inference = MoEInference(model_path="./checkpoints/checkpoint_epoch_10.safetensors")
    result = inference.generate("The fast bear couldn't bear")
    print(result)

if __name__ == "__main__":
    main()
