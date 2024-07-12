from unsloth import FastLanguageModel
import torch
import wandb
import os

wandb.login(key=os.getenv('WANDB_API_KEY'))

max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/Qwen2-0.5b-bnb-4bit",           # Qwen2 2x faster!
    "unsloth/Qwen2-1.5b-bnb-4bit",
    "unsloth/Qwen2-7b-bnb-4bit",
    "unsloth/Qwen2-72b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # Try more models at https://huggingface.co/unsloth!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-0.5B", # Reminder we support ANY Hugging Face model!
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

dataset_name = "ruslanmv/ai-medical-chatbot"
#Importing the dataset
dataset = load_dataset(dataset_name, split="all")
# Get the size of the dataset
dataset_size = len(dataset)
print(f"Size of the dataset: {dataset_size} samples")
def check_length(row):
    # This function checks if the tokenized length is within the max length allowed
    input_content = tokenizer.encode(row["Patient"] + ' ' + row["Doctor"],
                                        add_special_tokens=True,
                                        truncation=False,
                                        return_length=True,
                                        max_length=None)
    return len(input_content) <= max_seq_length - 20 # extra 20 tokens for the chat template

# Filter the dataset to exclude entries that are too long
dataset = dataset.filter(check_length)
print(f"Size of the dataset after filtering: {len(dataset)} samples")
print (dataset)

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]}]
    tokenized_output = tokenizer.apply_chat_template(
                            row_json,
                            tokenize=True,
                            add_generation_prompt=False,
                            padding="max_length",
                            max_length=max_seq_length,
                            truncation=True,
                    )
    input_ids = torch.tensor(tokenized_output, dtype=torch.int)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )

dataset = dataset.map(
    format_chat_template,
    # batched = True,
)
print("Column names in the dataset:", dataset.column_names)

dataset = dataset.train_test_split(test_size=0.1)

run = wandb.init(
    project='Fine-tune Qwen2 0.5B on Medical Dataset',
    job_type="training",
    anonymous="allow"
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    packing = True,
    # dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        num_train_epochs = 5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 8,
        evaluation_strategy = "epoch",

        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 20,
        #max_steps = 120,

        learning_rate = 3e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = ["wandb"],
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")