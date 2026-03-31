# COMP34812 NLU Coursework – Group 56

## Overview

This submission addresses the **Natural Language Inference (NLI)** track of the COMP34812 shared task.

The goal of the task is to determine whether a hypothesis is entailed by a given premise. Our group developed two solutions from different coursework categories:

- **Solution B** – a non-transformer deep learning approach based on a shared **BiLSTM** sentence-pair classifier
- **Solution C** – a transformer-based approach using **DistilRoBERTa**

This matches the coursework requirement to submit **two different solutions from two different categories**. :contentReference[oaicite:0]{index=0}

## Repository

## The full project code is also available on GitHub:

https://github.com/saadalharbii/NLU

## Solution Summary

### Solution B – Non-Transformer BiLSTM

Solution B is a shared **BiLSTM** model implemented in PyTorch. It uses:

- token embeddings learned from the training data
- a shared bidirectional LSTM encoder for premise and hypothesis
- pooled sentence representations
- pairwise interaction features built from the two encoded sentences

The final classifier is a small MLP on top of the pair representation.

### Solution C – DistilRoBERTa

Solution C fine-tunes **`distilroberta-base`** for binary sequence-pair classification. It takes the premise and hypothesis jointly as input and predicts whether the hypothesis is entailed by the premise.

This was our strongest-performing model on the development set.

---

## Running the Code

The coursework requires runnable demo code in notebook form, separate from the training/building code. :contentReference[oaicite:1]{index=1}

### Solution B

Use `nli_demo_code.ipynb`.

Expected inputs:

- the saved Solution B checkpoint/model file
- an input CSV containing:
  - `premise`
  - `hypothesis`

Expected output:

- a CSV with one column named `prediction`

### Solution C

Use `demo_notebook_C.ipynb`.

Expected inputs:

- `solution_C_model.zip`
- an input CSV containing:
  - `premise`
  - `hypothesis`

Expected output:

- a CSV with one column named `prediction`

---

## Model Files

Large model files are hosted externally rather than included in the submission zip. The coursework specification states that resources larger than 10MB should not be uploaded directly to Canvas and should instead be stored on the cloud with links provided in the README. :contentReference[oaicite:2]{index=2}

### Solution C model

**Google Drive link:**  
https://drive.google.com/drive/folders/12K2-yE7EpRcKzZxLiACt4MVTUe8W7YYw?usp=drive_link

### Solution B model

**Google Drive link:**  
https://drive.google.com/drive/folders/12K2-yE7EpRcKzZxLiACt4MVTUe8W7YYw?usp=drive_link

---

## Development Results

### Solution B

Best development performance reported by the selected BiLSTM configuration:

- **Macro-F1:** 0.6898
- **Accuracy:** 0.6909

Note: All of the hyperparam searches performaned the same or worse than just the defualt values using nli.py as standalone.

### Solution C

Development performance of the fine-tuned DistilRoBERTa model:

- **Macro-F1:** approximately 0.83
- **Accuracy:** approximately 0.83
- **MCC:** approximately 0.66

Overall, Solution C substantially outperformed the non-transformer approach, showing the advantage of contextual transformer representations for NLI.

---

## File Format Notes

Prediction files follow the coursework format:

- one column only
- column name: `prediction`
- one prediction per row, in the same order as the test input

This matches the required predictions format described in the coursework materials. :contentReference[oaicite:3]{index=3}

---

## Reproducibility Notes

The submission is organised so that:

- training/building code is separated from demo code
- demo notebooks can be used to generate predictions for a provided input file
- model cards describe the configuration, evaluation, and limitations of each solution

This follows the coursework guidance on organisation, reproducibility, and documentation. :contentReference[oaicite:4]{index=4}

---

## Use of Generative AI Tools

Generative AI tools were used to assist with:

- refining documentation and written explanations
- improving notebook structure
- debugging code

All final implementation choices, testing, result checking, and submission decisions were made by the group.

---

## Group Members

- Shane Gonsalves
- Maryam Azzam
- Saad Alharbi

---

## Final Note

This submission contains two different NLI systems from different coursework categories:

- a non-transformer deep learning model (Solution B)
- a transformer-based model (Solution C)

Both were developed using only the provided coursework data, in line with the shared task’s closed setting.
