import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel
from trl import GRPOConfig, GRPOTrainer
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-grpo-gsm8k"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Dataset Preparation
def format_gsm8k(example):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant. Provide your reasoning inside <think>...</think> tags, followed by the final answer inside <answer>...</answer> tags. Example: <think>2+2 is 4</think>\n<answer>4</answer>"},
            {"role": "user", "content": example["question"]}
        ],
        "answer": example["answer"].split("#### ")[-1].strip()
    }

# Force a smaller subset so it finishes faster
dataset = load_dataset("gsm8k", "main", split="train")
dataset = dataset.shuffle(seed=42).select(range(300))
dataset = dataset.map(format_gsm8k, remove_columns=["question", "answer"])

# Custom Programmatic Reward Functions (Fixed for TRL Data Structures)
def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """Assigns a reward if the model output strictly contains the XML tag format sequential order."""
    rewards = []
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[-1].get("content", "").strip()
        elif isinstance(completion, dict):
            text = completion.get("content", "").strip()
        else:
            text = str(completion).strip()

        if re.search(pattern, text, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    """Extracts the number inside <answer>...</answer> and compares it to the ground truth."""
    rewards = []

    n_gens = len(completions) // len(answer)

    for i, completion in enumerate(completions):
        gold_answer = answer[i // n_gens]

        if isinstance(completion, list) and len(completion) > 0:
            text = completion[-1].get("content", "").strip()
        elif isinstance(completion, dict):
            text = completion.get("content", "").strip()
        else:
            text = str(completion).strip()

        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)

        if match:
            predicted_text = match.group(1).strip()
            predicted_text = predicted_text.replace(",", "").replace("$", "")
            gold_answer = gold_answer.replace(",", "").replace("$", "")

            if predicted_text == gold_answer:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# Memory Optimization: LoRA Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# GRPO Hyperparameters
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,                  # Slightly higher LR for smaller dataset
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,

    # GRPO Specific parameters
    num_generations=2,
    max_completion_length=128,

    # Colab Memory Optimization
    optim="adamw_8bit",
    gradient_checkpointing=True,

    # Logging & Saving
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    report_to="none"
)

# Initialize Trainer
trainer = GRPOTrainer(
    model=MODEL_ID,
    reward_funcs=[format_reward_fn, accuracy_reward_fn],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Start Training
if __name__ == "__main__":
    print("Bootstrapping GRPO Optimization Loop:")
    trainer.train()

trainer.save_model("./qwen-grpo-gsm8k")
tokenizer.save_pretrained("./qwen-grpo-gsm8k")
print("Aligned LoRA adapters successfully saved")


print("Loading base model and attaching trained adapters")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./qwen-grpo-gsm8k")
tokenizer = AutoTokenizer.from_pretrained("./qwen-grpo-gsm8k")

# A brand new math problem to test reasoning
messages = [
    {"role": "system", "content": "You are a helpful assistant. Provide your reasoning inside <think>...</think> tags, followed by the final answer inside <answer>...</answer> tags."},
    {"role": "user", "content": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.6)

prompt_length = inputs["input_ids"].shape[1]
raw_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

print("\nTRAINED MODEL OUTPUT:\n" + "="*30 + f"\n{raw_output}\n" + "="*30)
