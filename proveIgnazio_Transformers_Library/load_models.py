import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TorchAoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from dotenv import dotenv_values 
from datasets import load_from_disk
from metrics_utils import compute_metrics
from peft import LoraConfig, TaskType, get_peft_model



# --- Configuration ---
# Load environment variables from the .env file located in the specified path
config = dotenv_values("proveIgnazio_Transformers_Library/.env")

# --- Load Models ---
# Get the model name/path from the loaded configuration
MODEL_NAME = 'T5_SMALL_60M'
MODEL = config["T5_SMALL_60M"]
PREPARED_DATASET = config.get("TOKENIZED_DATASET", config["PREPARED_DATASET"])
SAVED_MODEL_PATH = config["SAVED_MODEL_PATH"]

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
    output_dir="proveIgnazio_Transformers_Library/results", # Directory to save the model and training outputs
    per_device_train_batch_size=2, # Batch size for training on each device
    per_device_eval_batch_size=2, # Batch size for evaluation on each device
    gradient_accumulation_steps=5, # Number of steps to accumulate gradients before updating weights
    num_train_epochs=10, # Total number of training epochs
    logging_dir=f"proveIgnazio_Transformers_Library/tensorboard_logs/{MODEL_NAME}", # Directory for storing logs
    logging_steps=100, # Log every 10 steps
    save_steps=500, # Save the model every 500 steps
    evaluation_strategy="steps", # Evaluate the model every 'eval_steps'
    do_eval=True, # Perform evaluation during training
    eval_steps=500, # Evaluate every 500 steps
    seed=42, # Random seed for reproducibility
    load_best_model_at_end=True, # Load the best model at the end of training
    metric_for_best_model="eval_loss", # Metric to determine the best model
    greater_is_better=False, # Whether a higher metric value is better
    learning_rate=5e-5,  # Learning rate
    warmup_steps=500, # Number of warmup steps for learning rate scheduler
    eval_accumulation_steps=500, # Number of steps to accumulate gradients during evaluation
    eval_on_start=True, # Evaluate at the start of training
)

# Load the dataset
dataset = load_from_disk(PREPARED_DATASET)
print(f"Loaded dataset from: {PREPARED_DATASET}")
print(f"Dataset structure: {dataset}")

# Check if we're using a tokenized dataset or if we need to tokenize on-the-fly
is_tokenized = all(col in dataset["train"].column_names for col in ["input_ids", "attention_mask", "labels"])


# --- LoRA Configuration ---
# Define the LoRA configuration for model adaptation
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1, # dropout of LoRA layers
    # target_modules=["q", "k", "v", "o"] # Specify the target modules for LoRA adaptation
)

# Apply LoRA using get_peft_model
model = get_peft_model(model, lora_config)


# Create a data collator for padding sequences in batches efficiently
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length"
)

# Initialize the Trainer with the appropriate configuration
if is_tokenized:
    print("Using pre-tokenized dataset")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
else:
    print("Dataset not tokenized. Tokenizing on-the-fly...")
    # Define preprocessing function for on-the-fly tokenization
    def preprocess_function(examples):
        prefix = "answer the question: "
        inputs = [prefix + q for q in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["answer"], max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize the dataset on-the-fly if needed
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer", "id", "relevant_passage_ids"])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,    
    )

# --- Training ---
# Start the training process
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")


# Save the trained model
trainer.save_model(SAVED_MODEL_PATH+'_'+MODEL) # Save the model to the specified directory
# Save the tokenizer
tokenizer.save_pretrained(SAVED_MODEL_PATH+'_'+MODEL+'_'+'Tokenizer') # Save the tokenizer to the same directory