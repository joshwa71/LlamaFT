import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description="Finetune Llama7B using Hugging Face or local dataset")
parser.add_argument("--HF", type=bool, default=True, help="True for Hugging Face dataset, False for local dataset")
parser.add_argument("--dataset-dir", type=str, default="", help="Name of the Hugging Face dataset or local directory")
parser.add_argument("--save-steps", type=int, default=25, help="Interval of steps to save model checkpoints")
parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep")
parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for training")
parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs")
args = parser.parse_args()

# Dataset
if args.HF:
    dataset_name = args.dataset_dir if args.dataset_dir else "mlabonne/guanaco-llama2-1k"
    training_data = load_dataset(dataset_name, split="train")
else:
    # Assuming the dataset is in a local directory and loaded as a PyTorch Dataset
    # Replace this line with your local dataset loading logic
    training_data = torch.load(args.dataset_dir)

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
refined_model = "llama-2-7b-mlabonne-finetune"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./model",
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    logging_steps=25,
    learning_rate=args.learning_rate,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
