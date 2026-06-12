# evaluation.py
import os
import logging

# Suppress console logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    model_path = 'bert_model'
    tokenizer_path = 'bert_tokenizer'

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Error: Saved model folders not found. Please run model_training.py first.")
        return

    print("Loading local standard BERT classifier and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Evaluation validation set
    test_reviews = [
        "I absolutely loved this movie, the cinematography was amazing!",
        "The story made no sense and the acting was atrocious.",
        "It was an average film, some good moments but mostly boring.",
        "A true cinematic masterpiece with spectacular performances.",
        "I regret spending money on this garbage.",
        "Brilliant screenplay coupled with top notch direction."
    ]
    # 1 = Positive, 0 = Negative
    y_test = [1, 0, 0, 1, 0, 1]
    
    print("\nEvaluating text sequences...")
    predicted_classes = []
    
    with torch.no_grad():
        for review in test_reviews:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
            predicted_classes.append(np.argmax(probs))

    # 1. Detailed Classification Report
    target_names = ['Negative', 'Positive']
    print("\nDetailed Sequence Classification Metrics:")
    print(classification_report(y_test, predicted_classes, target_names=target_names))
    
    # 2. Save Confusion Matrix Plot
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, predicted_classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Greens, ax=ax)
    
    plot_filename = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Confusion matrix image saved as '{plot_filename}'.")

if __name__ == '__main__':
    main()