# Model Card: Non-Transformer NLI BiLSTM

## Model Details

- **Model name:** Non-Transformer NLI BiLSTM
- **Track / task:** Natural Language Inference (NLI)
- **Approach category:** B (deep learning without transformers)
- **Group number:** 56
- **Authors:** Shane, Maryam, Saad
- **Repository:** https://gitlab.cs.man.ac.uk/x37259sg/nli56.git
- **Primary code file:** `nli.py`
- **Contact:** shane.gonsalves@student.manchester.ac.uk

## Model Summary

This model performs Natural Language Inference (NLI) on premise–hypothesis pairs.
Given a `premise` and a `hypothesis`, it predicts the task label for the pair.

This implementation is a **shared BiLSTM classifier** written in PyTorch.

## Intended Use

This model is intended for:
- coursework experimentation and evaluation on the COMP34812 NLI shared task,
- reproducible generation of predictions for the provided train/dev/test split,
- comparison with other NLI approaches from a different coursework category.

It is **not** intended for high-stakes or safety-critical use.

## Training Data

The model is designed for the provided coursework data only.
It expects CSV files with:
- `premise`
- `hypothesis`
- `label` (for training/evaluation)

For inference, the input CSV should contain:
- `premise`
- `hypothesis`

The test data follows the same format as the trial data, except that the `label` column is absent.

## Preprocessing

The preprocessing pipeline in `nli.py` is:

- lowercasing of text,
- regex tokenisation using a simple token pattern,
- vocabulary creation from the **training data only**,
- rare token filtering using `min_freq`,
- mapping unseen tokens to `<unk>`,
- truncation to `max_len`,
- dynamic padding within each batch.

## Model Architecture

### Encoder
The model uses a **shared bidirectional LSTM encoder** for both the premise and the hypothesis.

The encoder consists of:
- an embedding layer,
- dropout on embeddings,
- a bidirectional LSTM,
- mean pooling over time,
- max pooling over time.

### Sentence Pair Features
Let:
- `u` = encoded premise representation
- `v` = encoded hypothesis representation

The classifier builds the pair representation:
- `u`
- `v`
- `|u - v|`
- `u * v`

These are concatenated and passed to a small MLP classifier.

### Classifier
The classifier consists of:
- linear layer,
- ReLU,
- dropout,
- final linear layer to output class logits.

## Training Procedure

Training is implemented in `train_model(...)` in `nli.py`.

### Optimisation
- **Loss:** Cross-entropy loss
- **Optimiser:** AdamW
- **Gradient clipping:** max norm = 1.0
- **Model selection:** best checkpoint selected by dev macro-F1
- **Early stopping:** stops after `patience` non-improving epochs

### Final Hyperparameters
- **Batch size:** 64
- **Epochs:** 15
- **Learning rate:** 0.001
- **Weight decay:** 0.0001
- **Embedding dimension:** 200
- **Hidden dimension:** 128
- **MLP dimension:** 256
- **Number of BiLSTM layers:** 1
- **Dropout:** 0.3
- **Max sequence length:** 128
- **Min token frequency:** 2
- **Max vocabulary size:** 30000
- **Patience:** 3
- **Random seed:** 42

## Evaluation

### Metrics
The code reports:
- **Macro-F1**
- **Accuracy**

### Best Development Results
- **Best dev macro-F1:** 0.6898
- **Best dev accuracy:** 0.6909
- **Best epoch:** 3.0000

## Inference

Inference is implemented in `run_inference(...)` in `nli.py`.

The model:
1. loads the saved checkpoint,
2. restores the saved vocabulary and label mapping,
3. reads an input CSV containing `premise` and `hypothesis`,
4. predicts one label per row,
5. writes a CSV with a single `label` column.

## Files

- `nli.py` — model, training, evaluation, and inference code
- `nli_hparam_search.py` — optional hyperparameter-search / run-selection utility
- checkpoint file — saved best model
- demo notebook — inference-mode notebook for the coursework
- this model card

## Experimental Utilities

A separate utility script, `nli_hparam_search.py`, was used to run multiple training configurations, compare runs by dev macro-F1, and copy the best checkpoint and logs. It is an experiment-management script rather than a separate model.

## Strengths

- Simple and reproducible implementation
- Explicitly satisfies the coursework’s non-transformer deep learning category
- Uses a shared encoder, reducing parameter duplication
- Uses both mean pooling and max pooling to capture sentence-level information
- Includes early stopping and checkpoint saving for reproducibility

## Limitations

- May struggle on examples requiring fine-grained token alignment or deeper semantic reasoning.
- Uses simple regex tokenisation and train-from-scratch word embeddings rather than pretrained language models.

## Ethical Considerations

This model is for coursework and experimentation only.
It may reflect dataset biases or annotation artefacts.
Its outputs should not be interpreted as reliable real-world reasoning beyond the provided task.

## Reproducibility

To reproduce the final system, provide:
- the final code,
- the saved checkpoint,
- this model card,
- the demo notebook,
- the README with instructions and any required attribution.

### Example Commands

Train:
```bash
python nli.py train --train_csv train.csv --dev_csv dev.csv --output_dir runs/nli_bilstm
```

Evaluate:
```bash
python nli.py eval --checkpoint runs/nli_bilstm/best_model.pt --dev_csv dev.csv
```

Predict:
```bash
python nli.py predict --checkpoint runs/nli_bilstm/best_model.pt --input_csv test.csv --output_csv preds.csv
```
