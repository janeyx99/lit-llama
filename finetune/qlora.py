"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time

import numpy as np
import torch
from tqdm import tqdm
import random

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig, QloraMLP, QloraConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

from transformer_nuggets.utils import save_memory_snapshot


instruction_tuning = True
eval_interval = 10
save_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 5e-5
batch_size = 128
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 256  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 8
lora_dropout = 0.05
warmup_iters = 100
device = torch.device("cuda")

def swap_for_qlora(model: torch.nn.Module) -> None:
    print("Swapping for Qlora...")
    for name, child in tqdm(model.named_modules(), total = len(list(model.named_modules()))):
        if name.endswith("mlp"):
            w1 = child.c_fc1.weight
            w2 = child.c_fc2.weight
            w3 = child.c_proj.weight
            qlora_config = QloraConfig(lora_r, lora_alpha, lora_dropout)
            setattr(model, name, QloraMLP(w1.to(torch.bfloat16), w2.to(torch.bfloat16), w3.to(torch.bfloat16), qlora_config))

def swap_for_qlora_jank(model: torch.nn.Module) -> None:
    print("Swapping for Qlora...")
    for module in tqdm(model.transformer.h):
        current_mlp = module.mlp
        w1 = current_mlp.c_fc1.weight
        w2 = current_mlp.c_fc2.weight
        w3 = current_mlp.c_proj.weight
        qlora_config = QloraConfig(lora_r, lora_alpha, lora_dropout)
        new_mod = QloraMLP(w1.to(torch.bfloat16), w2.to(torch.bfloat16), w3.to(torch.bfloat16), qlora_config)
        module.mlp = new_mod
        del current_mlp

def main(
    data_dir: Path = Path("data/alpaca"), 
    pretrained_path: Path = Path("checkpoints/lit-llama/llama-2-7b/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    out_dir: Path = Path("out/lora/alpaca"),
    compile: bool = False,
):  
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    os.makedirs(out_dir, exist_ok=True)
    train_data, val_data = load_datasets(data_dir=data_dir)
    config = LLaMAConfig.from_name(str(pretrained_path))
    config.block_size = max_seq_length
    
    print("Loading model...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    print("Checkpoint loaded")
    with torch.device('meta'):
        model = LLaMA(config)

    model.load_state_dict(checkpoint, strict=True, assign=True)
    model.to(device)
    print("Loaded!")

    # Qlora Module swapping
    swap_for_qlora_jank(model)
    mark_only_lora_as_trainable(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"The number of trainable parameters: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    if compile:
        model = torch.compile(model)
    train(model, optimizer, train_data, val_data, tokenizer_path, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    torch.save(checkpoint, out_dir / "lit-llama-qlora.pth")


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    progress_bar = tqdm(total=max_iters)
    # Initial validation
    val_loss = validate(model, val_data, tokenizer_path)
    print(f"step {0}: val loss {val_loss:.4f}")
    model.train()
    for iter_num in range(max_iters):
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(train_data)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)

        loss = loss_fn(logits, targets)

        # Scale the loss by grad_accumulation iters
        (loss/gradient_accumulation_iters).backward()
        
        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(model, val_data, tokenizer_path)
                print(f"step {iter_num}: val loss {val_loss:.4f}")

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                torch.save(checkpoint, out_dir / f"iter-{iter_num:06d}-ckpt.pth")

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            # tqdm.write(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            # progress_bar.set_description(f"Iter {iter_num}: Loss {loss.item():.4f}, Time: {dt*1000:.2f}ms")
            progress_bar.set_postfix_str(f"Iter {iter_num}: Loss {loss.item():.4f}, Time: {dt*1000:.2f}ms")
        progress_bar.update(1)

@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def validate(model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, tokenizer_path)
    print(instruction)
    print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = x.pin_memory().to(device), y.pin_memory().to(device)
    # x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse import CLI

    CLI(main)
