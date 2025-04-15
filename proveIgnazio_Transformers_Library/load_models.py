import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TorchAoConfig
from transformers import TrainingArguments, Trainer
from dotenv import dotenv_values 


# --- Configuration ---
# Load environment variables from the .env file located in the specified path
config = dotenv_values(".env")

# --- Load Models ---
# Get the model name/path from the loaded configuration
MODEL = config["T5_BASE_223M"]

# Define the quantization configuration using TorchAoConfig for int4 weight-only quantization
quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

# Load the pre-trained Seq2Seq language model (T5)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL, # Model identifier
    torch_dtype=torch.bfloat16, # Use bfloat16 for mixed-precision inference
    device_map="auto", # Automatically map model layers to available devices (CPU/GPU)
    # quantization_config=quantization_config # Apply the defined quantization configuration
)

# Load the tokenizer associated with the specific T5 model variant being used
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# --- Load Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results", # Directory to save the model and training outputs
    per_device_train_batch_size=2, # Batch size for training on each device
    per_device_eval_batch_size=2, # Batch size for evaluation on each device
    gradient_accumulation_steps=1, # Number of steps to accumulate gradients before updating weights
    num_train_epochs=10, # Total number of training epochs
    logging_dir="./logs", # Directory for storing logs
    logging_steps=10, # Log every 10 steps
    save_steps=500, # Save the model every 500 steps
    evaluation_strategy="steps", # Evaluate the model every 'eval_steps'
    eval_steps=500, # Evaluate every 500 steps
    seed=42, # Random seed for reproducibility
    fp16=True, # Use mixed precision training (if supported by the hardware)
    load_best_model_at_end=True, # Load the best model at the end of training
    metric_for_best_model="eval_loss", # Metric to determine the best model
    greater_is_better=False, # Whether a higher metric value is better
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)