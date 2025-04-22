# --- Script Workflow ---
# This script orchestrates the fine-tuning of a Seq2Seq language model (specifically FLAN-T5)
# for a Question Answering task using the Hugging Face Transformers and PEFT libraries.
# The key steps involved are:
# 1.  **Import Libraries:** Import necessary modules from `torch`, `transformers`, `datasets`, `dotenv`, `peft`, `os`, and `numpy`.
# 2.  **Environment Setup:** Configure PyTorch MPS fallback settings for macOS GPU usage.
# 3.  **Configuration Loading:** Load environment variables (like model paths, dataset paths) from a `.env` file.
# 4.  **Model Definition:** Specify the pre-trained model name (e.g., "FLAN_T5_SMALL_77M") and retrieve its path from the config. Define paths for the dataset and where the fine-tuned model will be saved.
# 5.  **Quantization (Optional):** Define a quantization configuration (currently commented out) for potential model size reduction and performance improvement.
# 6.  **Model Loading:** Load the specified pre-trained Seq2Seq model (e.g., FLAN-T5) using `AutoModelForSeq2SeqLM`, potentially applying quantization and mapping it to the appropriate device (CPU in this case).
# 7.  **LoRA Configuration:** Define a Low-Rank Adaptation (LoRA) configuration using `LoraConfig` to enable efficient fine-tuning by adapting only a small number of parameters.
# 8.  **PEFT Model Creation:** Apply the LoRA configuration to the base model using `get_peft_model` from the PEFT library.
# 9.  **Tokenizer Loading:** Load the tokenizer corresponding to the chosen pre-trained model using `AutoTokenizer`.
# 10. **Training Arguments:** Define training hyperparameters (batch size, epochs, learning rate, logging/saving frequency, evaluation strategy, etc.) using `Seq2SeqTrainingArguments`.
# 11. **Dataset Loading:** Load the dataset, potentially pre-tokenized, from the disk path specified in the configuration.
# 12. **Dataset Type Check:** Determine if the loaded dataset is already tokenized or requires on-the-fly tokenization.
# 13. **Data Collator:** Initialize a `DataCollatorForSeq2Seq` to handle dynamic padding of input and label sequences within each batch.
# 14. **Metrics Computation Function:** Define a wrapper function (`compute_metrics_for_trainer`) that prepares predictions and labels and calls the actual metric calculation logic (imported from `metrics_utils`). This is necessary for Seq2Seq tasks during evaluation.
# 15. **Trainer Initialization:** Instantiate a `Seq2SeqTrainer` by providing the model, training arguments, datasets (train and validation), data collator, metrics function, and tokenizer.
# 16. **On-the-Fly Tokenization (Conditional):** If the dataset is not pre-tokenized, define a `preprocess_function` and apply it using `.map()` to tokenize the data dynamically. Re-initialize the Trainer with the tokenized datasets.
# 17. **Training Execution:** Start the fine-tuning process by calling `trainer.train()`. Includes basic error handling.
# 18. **Model Saving:** Save the fine-tuned model's weights (specifically the adapted LoRA weights) and configuration using `trainer.save_model()`.
# 19. **Tokenizer Saving:** Save the tokenizer associated with the model using `tokenizer.save_pretrained()` for easy reloading later.
# --- End Script Workflow ---

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TorchAoConfig,
    AutoModelForCausalLM,
)
from transformers import TrainingArguments, Trainer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from dotenv import dotenv_values
from datasets import load_from_disk
from metrics_utils import compute_metrics_base
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model
import os
import numpy as np

# Set environment variable for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- Configuration ---
# Load environment variables from the .env file located in the specified path
config = dotenv_values(".env")

# --- Load Models ---
# Get the model name/path from the loaded configuration
MODEL_NAME = "FLAN_T5_SMALL_77M"
MODEL = config["FLAN_T5_SMALL_77M"]
PREPARED_DATASET = config.get("TOKENIZED_DATASET", config["PREPARED_DATASET"])
SAVED_MODEL_PATH = config["SAVED_MODEL_PATH"]

# Define the quantization configuration using TorchAoConfig for int8 weight-only quantization
quantization_config = TorchAoConfig("int8_weight_only")

# Load the pre-trained Seq2Seq language model (T5)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,  # Model identifier
    torch_dtype=torch.bfloat16,  # Use bfloat16 for mixed-precision inference
    # torch_dtype=torch.float32,  # Use float32 for mixed-precision inference
    device_map="cpu",  # Map the model to CPU (or "auto" for automatic mapping)
    # NOTE: non è la quantizzazione il problema, riabilitarla
    # quantization_config=quantization_config,  # Apply the defined quantization configuration
)


# --- LoRA Configuration ---
# Define the LoRA configuration for model adaptation
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # type of task to train on
    inference_mode=False,  # set to False for training
    r=2,  # dimension of the smaller matrices
    lora_alpha=40,  # scaling factor
    lora_dropout=0.5,  # dropout of LoRA layers
    # target_modules=["q", "k", "v", "o"] # Specify the target modules for LoRA adaptation
)

# Apply LoRA using get_peft_model
model = get_peft_model(model, lora_config)

# model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory during training

# Load the tokenizer associated with the specific T5 model variant being used
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# --- Load Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="proveIgnazio_Transformers_Library/results",  # Directory to save the model and training outputs
    per_device_train_batch_size=5,  # Batch size for training on each device
    per_device_eval_batch_size=2,  # Batch size for evaluation on each device
    gradient_accumulation_steps=1,  # Number of steps to accumulate gradients before updating weights
    num_train_epochs=10,  # Total number of training epochs
    logging_dir=f"proveIgnazio_Transformers_Library/tensorboard_logs/{MODEL_NAME}",  # Directory for storing logs
    logging_steps=300,  # Log every 10 steps
    save_steps=500,  # Save the model every 500 steps
    eval_strategy="steps",  # Evaluate the model every 'eval_steps'
    do_eval=True,  # Perform evaluation during training
    eval_steps=500,  # Evaluate every 500 steps
    seed=42,  # Random seed for reproducibility
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Metric to determine the best model
    greater_is_better=False,  # Whether a higher metric value is better
    learning_rate=5e-5,  # Learning rate
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    dataloader_num_workers=0,  # Number of subprocesses to use for data loading
    eval_accumulation_steps=50,  # Muove ogni batch subito in CPU, evitando di creare buffer grandi
    eval_on_start=True,  # Evaluate at the start of training
    predict_with_generate=True,  # Added: Necessary for Seq2Seq metrics
)

# Load the dataset
dataset = load_from_disk(PREPARED_DATASET)
print(f"Loaded dataset from: {PREPARED_DATASET}")
print(f"Dataset structure: {dataset}")

# Check if we're using a tokenized dataset or if we need to tokenize on-the-fly
is_tokenized = all(
    col in dataset["train"].column_names
    for col in ["input_ids", "attention_mask", "labels"]
)


# Create a data collator for padding sequences in batches efficiently
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    # padding="max_length",
    padding="longest",
)


def compute_metrics_for_trainer(eval_preds):
    preds, labels = eval_preds
    print(
        f"compute_metrics_for_trainer - Original preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'N/A')}"
    )
    print(
        f"compute_metrics_for_trainer - Original labels type: {type(labels)}, shape: {getattr(labels, 'shape', 'N/A')}"
    )
    # Print some example values if they are numpy arrays
    if hasattr(preds, "shape"):
        print(
            f"compute_metrics_for_trainer - preds sample: {preds[0][:20] if len(preds) > 0 else 'empty'}"
        )  # Print first 20 tokens of first prediction
    if hasattr(labels, "shape"):
        print(
            f"compute_metrics_for_trainer - labels sample: {labels[0][:20] if len(labels) > 0 else 'empty'}"
        )  # Print first 20 tokens of first label

    if isinstance(preds, tuple):
        print("compute_metrics_for_trainer - preds is a tuple, taking first element")
        preds = preds[0]
        print(
            f"compute_metrics_for_trainer - Updated preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'N/A')}"
        )
        if hasattr(preds, "shape"):
            print(
                f"compute_metrics_for_trainer - preds sample after tuple extraction: {preds[0][:20] if len(preds) > 0 else 'empty'}"
            )

    # Ensure labels are handled correctly (e.g., removing padding for metrics)
    # The -100 replacement happens in compute_metrics_base, which is fine

    # Call the base function
    return compute_metrics_base(
        preds, labels, tokenizer, metric_rouge, metric_bleu, metric_meteor
    )  # Pass tokenizer and metrics


# Initialize the Trainer with the appropriate configuration
if is_tokenized:
    print("Using pre-tokenized dataset")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"].select(range(100)),
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_trainer,  # Use the wrapper function
        tokenizer=tokenizer,  # Pass the tokenizer to the Trainer
    )
else:
    print("Dataset not tokenized. Tokenizing on-the-fly...")

    # Define preprocessing function for on-the-fly tokenization
    def preprocess_function(examples):
        prefix = "answer the question: "
        inputs = [prefix + q for q in examples["question"]]

        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            # padding="max_length"
        )

        labels = tokenizer(
            text_target=examples["answer"],
            max_length=512,
            truncation=True,
            # padding="max_length",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize the dataset on-the-fly if needed
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["question", "answer", "id", "relevant_passage_ids"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_trainer,  # Use the wrapper function
        tokenizer=tokenizer,  # Pass the tokenizer to the Trainer
    )

# --- Training ---
# Start the training process
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")


# Save the trained model
trainer.save_model(
    SAVED_MODEL_PATH + "_" + MODEL_NAME  # Use MODEL_NAME for consistency
)  # Save the model to the specified directory
# Save the tokenizer
tokenizer.save_pretrained(
    SAVED_MODEL_PATH
    + "_"
    + MODEL_NAME
    + "_"
    + "Tokenizer"  # Use MODEL_NAME for consistency
)  # Save the tokenizer to the same directory
