import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from safetensors.torch import save_file, load_file
from pathlib import Path
import time

def count_parameters_layerwise(model):
    # Layerwise params, turn this into a util function.
    total_params = 0
    layer_params = {}
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
            
        param_count = parameter.numel()
        layer_params[name] = param_count
        total_params += param_count
    
    print(f"\nModel Parameter Summary:")
    print("-" * 60)
    for name, count in layer_params.items():
        print(f"{name}: {count:,} parameters")
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params:,}\n")
    
    return total_params

def save_checkpoint(model, optimizer, filename="checkpoint.safetensors"):
    # Get the original uncompiled model if it's compiled -- compiled model has weird keys
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()
    
    save_file(model_state, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.safetensors"):
    model_state = load_file(filename)
    
    # Get the original uncompiled model if it's compiled -- compiled model has weird keys
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
        
class TBLogger:
    def __init__(self, log_dir='logs/current_run', flush_secs=10, enable_grad_logging=True):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        self.enable_grad_logging = enable_grad_logging
        self.start_time = time.time()
        
    def log(self, metrics, step=None, model=None, prefix='', grad_checking=False):
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            
            if isinstance(value, (int, float)):
                self.writer.add_scalar(full_name, value, step)
            elif isinstance(value, torch.Tensor):
                self.writer.add_scalar(full_name, value.item(), step)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if all(isinstance(x, (int, float)) for x in value):
                    self.writer.add_histogram(full_name, torch.tensor(value), step)
                
        if self.enable_grad_logging and model is not None:
            self._log_gradients(model, step, grad_checking)
            
    def _log_gradients(self, model, step, grad_checking):
        total_norm = 0.0
        for name, param in model.named_parameters():
            if grad_checking and param.grad is not None:
                # Check for inf/nan in gradients
                if torch.isnan(param.grad).any():
                    print(f"Warning: Found nan in gradients for layer: {name}")
                    continue
                if torch.isinf(param.grad).any():
                    print(f"Warning: Found inf in gradients for layer: {name}")
                    continue
                    
                param_norm = param.grad.detach().data.norm(2)
                self.writer.add_scalar(f"gradients/{name}_norm", param_norm, step)
                total_norm += param_norm.item() ** 2
        
        # Only compute total norm if we haven't encountered inf/nan
        if total_norm > 0:  # This means we had valid gradients
            total_norm = total_norm ** 0.5
            self.writer.add_scalar("gradients/total_norm", total_norm, step)
    
    def close(self):
        self.writer.close()