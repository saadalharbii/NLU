# Model Card – Solution C (Transformer-based NLI)

## Model Overview

This model addresses the Natural Language Inference (NLI) task as a binary classification problem. Given a premise and a hypothesis, the model predicts whether the hypothesis is entailed by the premise (label 1) or not (label 0).

The solution is based on a pretrained transformer model, **DistilRoBERTa**, which is fine-tuned on the provided NLI dataset. This approach falls under Category C (deep learning methods) as defined in the coursework.

Compared to traditional feature-based methods, this model leverages contextual embeddings to better capture semantic relationships between sentence pairs.

---

## Model Details

- **Base model:** `distilroberta-base`
- **Task:** Binary sequence pair classification (NLI)
- **Framework:** Hugging Face Transformers
- **Tokenisation:** WordPiece/BPE tokenizer from DistilRoBERTa
- **Max sequence length:** 256 tokens

Each input consists of a pair of sentences (premise and hypothesis), which are jointly encoded and passed through the transformer model.

---

## Training Procedure

The model was fine-tuned on the provided training data using the Hugging Face `Trainer` API.

### Training configuration:
- **Epochs:** 2  
- **Batch size:** 16  
- **Learning rate:** 2e-5  
- **Weight decay:** 0.01  
- **Warmup:** 10% of training steps  
- **Optimizer:** AdamW (default in Trainer)

The model was evaluated on the development set after each epoch, and the best checkpoint was selected based on **macro-F1 score**.

After hyperparameter selection, the final model was retrained on the combined **training + development data** before generating predictions on the test set.

---

## Evaluation

Performance was evaluated on the development set using multiple metrics:

- **Accuracy:** ~0.83  
- **Macro Precision:** ~0.83  
- **Macro Recall:** ~0.83  
- **Macro F1:** ~0.83  
- **Matthews Correlation Coefficient (MCC):** ~0.66  

These results represent a substantial improvement over a traditional TF-IDF-based baseline (~0.58 macro-F1), demonstrating the benefit of contextual representations.

---

## Error Analysis

Although the model performs well overall, some types of errors remain:

- Cases requiring **implicit reasoning** or external knowledge  
- Examples where premise and hypothesis share vocabulary but differ in meaning  
- Subtle contradictions or paraphrases that are difficult to distinguish  
- Ambiguous or noisy input text  

These errors suggest that while the model captures context effectively, it is not perfect at deeper reasoning or world knowledge.

---

## Limitations

- The model is relatively large (~300MB), which may limit portability  
- It requires GPU acceleration for efficient training  
- Performance may degrade on domains very different from the training data  
- Like most transformer models, it may struggle with rare edge cases or ambiguous phrasing  

---

## Reproducibility

The model can be reproduced using the provided training notebook.  
A separate demo notebook is included to load the saved model and generate predictions without retraining.

The following resources are required:
- Pretrained model weights (`solution_C_model.zip`)
- Input CSV file with `premise` and `hypothesis` columns

---

## Ethical Considerations

This model is trained on general language data and may reflect biases present in the underlying datasets. Predictions should therefore be interpreted with caution, especially in sensitive applications.

---

## Summary

This solution demonstrates that transformer-based models significantly outperform traditional approaches for NLI tasks. By fine-tuning DistilRoBERTa, the model achieves strong performance while remaining feasible to train within the constraints of the coursework.