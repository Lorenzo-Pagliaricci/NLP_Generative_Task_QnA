from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os

def preprocess_dataset(dataset_name, model_name, output_dir):
    """
    Preprocess the dataset by tokenizing the input and target columns.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face Hub.
        model_name (str): The name of the model to load the tokenizer for.
        output_dir (str): The directory to save the processed dataset.

    Returns:
        None
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, 'question-answer-passages')

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the preprocessing function
    def preprocess_function(examples):
        prefix = "answer the question: "
        inputs = [prefix + q for q in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # Tokenize the target (answers)
        labels = tokenizer(text_target=examples["answer"], max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split the train dataset into train and validation subsets
    train_validation_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_validation_split["train"]
    validation_dataset = train_validation_split["test"]
    test_dataset = dataset["test"]

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # Apply the preprocessing function
    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

    # Remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(["question", "answer", "id", "relevant_passage_ids"])

    # Save the processed dataset to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenized_datasets.save_to_disk(output_dir)

    print(f"Processed dataset saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    DATASET_NAME = "enelpol/rag-mini-bioasq"
    MODEL_NAME = "google/t5-base"
    OUTPUT_DIR = "data/tokenized_data"

    preprocess_dataset(DATASET_NAME, MODEL_NAME, OUTPUT_DIR)