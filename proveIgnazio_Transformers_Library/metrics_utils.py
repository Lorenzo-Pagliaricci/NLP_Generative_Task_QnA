import evaluate

try:
    # Initialize the metric
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
except Exception as e:
    print(f"Error loading metrics: {e}")


# Inside metrics_utils.py


def compute_metrics_base(
    predictions, labels, tokenizer, metric_rouge, metric_bleu, metric_meteor
):
    print(f"\ncompute_metrics_base - Received predictions shape: {predictions.shape}")
    print(f"compute_metrics_base - Received labels shape: {labels.shape}")
    # Print min/max values to check ranges
    print(
        f"compute_metrics_base - predictions min/max: {np.min(predictions)}, {np.max(predictions)}"
    )
    print(
        f"compute_metrics_base - labels min/max before replace: {np.min(labels)}, {np.max(labels)}"
    )

    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    print(
        f"compute_metrics_base - labels min/max after replace: {np.min(labels)}, {np.max(labels)}"
    )

    # Decode
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(
            f"compute_metrics_base - Decoded preds sample: {decoded_preds[0] if decoded_preds else 'empty'}"
        )
        print(
            f"compute_metrics_base - Decoded labels sample: {decoded_labels[0] if decoded_labels else 'empty'}"
        )
    except Exception as e:
        print(f"ERROR during decoding: {e}")
        # You might want to return dummy metrics here or re-raise
        return {"error": "decoding failed"}

    # Compute metrics (add try-except for each)
    results = {}
    try:
        rouge_output = metric_rouge.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        results.update(
            {key: value for key, value in rouge_output.items()}
        )  # Flatten ROUGE scores
        print("compute_metrics_base - ROUGE computed")
    except Exception as e:
        print(f"ERROR computing ROUGE: {e}")
        results["rouge_error"] = str(e)

    try:
        # BLEU
        bleu_output = metric_bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        results.update(
            {key: value for key, value in bleu_output.items()}
        )  # Flatten BLEU scores
        print("compute_metrics_base - BLEU computed")
    except Exception as e:
        print(f"ERROR computing BLEU: {e}")
        results["bleu_error"] = str(e)

    try:
        # METEOR
        meteor_output = metric_meteor.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        results.update(
            {key: value for key, value in meteor_output.items()}
        )  # Flatten METEOR scores
        print("compute_metrics_base - METEOR computed")
    except Exception as e:
        print(f"ERROR computing METEOR: {e}")
        results["meteor_error"] = str(e)

    # Print final results
    print(f"compute_metrics_base - Final results: {results}")

    return results
