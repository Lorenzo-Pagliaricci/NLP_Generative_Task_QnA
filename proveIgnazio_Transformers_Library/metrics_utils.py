import evaluate

try:
    # Initialize the metric
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
except Exception as e:
    print(f"Error loading metrics: {e}")

 

def compute_metrics(predictions, references):
    """
    Compute BLEU, ROUGE, and METEOR scores for the given predictions and references.

    Args:
        predictions (list): List of predicted strings.
        references (list): List of reference strings.

    Returns:
        dict: Dictionary containing BLEU, ROUGE, and METEOR scores.
    """
    # Ensure that predictions and references are in the correct format
    if isinstance(predictions, list) and isinstance(references, list):
        predictions = [pred.strip() for pred in predictions]
        references = [[ref.strip()] for ref in references]  # Wrap each reference in a list

    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # Compute ROUGE score
    rouge_score = rouge.compute(predictions=predictions, references=references)

    # Compute METEOR score
    meteor_score = meteor.compute(predictions=predictions, references=references)

    return {
        "bleu": bleu_score["bleu"],
        "rouge": rouge_score,
        "meteor": meteor_score["meteor"],
    }