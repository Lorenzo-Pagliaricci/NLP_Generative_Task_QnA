# This file defines utility functions for computing evaluation metrics for NLP generative tasks.
#
# Workflow:
# 1. Initialize evaluation metrics (BLEU, ROUGE, METEOR) using the `evaluate` library.
# 2. Define the `compute_metrics_base` function which takes predictions, labels, tokenizer, and metric objects as input.
# 3. Inside `compute_metrics_base`:
#    - Print input shapes and value ranges for debugging.
#    - Replace padding token IDs (-100) in labels with the actual padding token ID from the tokenizer.
#    - Decode predicted token IDs and label token IDs into text strings.
#    - Compute ROUGE, BLEU, and METEOR scores using the decoded text.
#    - Handle potential errors during decoding or metric computation.
#    - Return a dictionary containing the computed metric scores.

import evaluate
import numpy as np  # Added import for numpy

try:
    # Initialize the metric objects globally for efficiency
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
except Exception as e:
    print(f"Error loading metrics: {e}")


# Inside metrics_utils.py


def compute_metrics_base(
    predictions, labels, tokenizer, metric_rouge, metric_bleu, metric_meteor
):
    """
    Computes ROUGE, BLEU, and METEOR metrics for generated predictions against reference labels.

    Args:
        predictions: NumPy array of predicted token IDs.
        labels: NumPy array of ground truth token IDs.
        tokenizer: The tokenizer used for decoding.
        metric_rouge: Initialized ROUGE metric object.
        metric_bleu: Initialized BLEU metric object.
        metric_meteor: Initialized METEOR metric object.

    Returns:
        A dictionary containing the computed metric scores.
    """
    print(f"\ncompute_metrics_base - Received predictions shape: {predictions.shape}")
    print(f"compute_metrics_base - Received labels shape: {labels.shape}")
    # Print min/max values to check token ID ranges
    print(
        f"compute_metrics_base - predictions min/max: {np.min(predictions)}, {np.max(predictions)}"
    )
    print(
        f"compute_metrics_base - labels min/max before replace: {np.min(labels)}, {np.max(labels)}"
    )

    # Replace -100 (ignore index) in labels with the tokenizer's pad token ID for correct decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    print(
        f"compute_metrics_base - labels min/max after replace: {np.min(labels)}, {np.max(labels)}"
    )

    # Decode token IDs to strings
    try:
        # Decode predictions, skipping special tokens (like padding, eos, bos)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Decode labels, skipping special tokens
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(
            f"compute_metrics_base - Decoded preds sample: {decoded_preds[0] if decoded_preds else 'empty'}"
        )
        print(
            f"compute_metrics_base - Decoded labels sample: {decoded_labels[0] if decoded_labels else 'empty'}"
        )
    except Exception as e:
        print(f"ERROR during decoding: {e}")
        # Return an error indicator if decoding fails
        return {"error": "decoding failed"}

    # Compute metrics using the decoded strings
    results = {}
    try:
        # Compute ROUGE scores
        rouge_output = metric_rouge.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        # Add individual ROUGE scores (rouge1, rouge2, rougeL, rougeLsum) to results
        results.update({key: value for key, value in rouge_output.items()})
        print("compute_metrics_base - ROUGE computed")
    except Exception as e:
        print(f"ERROR computing ROUGE: {e}")
        results["rouge_error"] = str(e)  # Store error message if computation fails

    try:
        # Compute BLEU score
        bleu_output = metric_bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        # Add BLEU score components to results
        results.update({key: value for key, value in bleu_output.items()})
        print("compute_metrics_base - BLEU computed")
    except Exception as e:
        print(f"ERROR computing BLEU: {e}")
        results["bleu_error"] = str(e)  # Store error message

    try:
        # Compute METEOR score
        meteor_output = metric_meteor.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        # Add METEOR score to results
        results.update({key: value for key, value in meteor_output.items()})
        print("compute_metrics_base - METEOR computed")
    except Exception as e:
        print(f"ERROR computing METEOR: {e}")
        results["meteor_error"] = str(e)  # Store error message

    # Print final computed metrics for inspection
    print(f"compute_metrics_base - Final results: {results}")

    return results
