# Vaccinate or Not Vaccinate: Vaccine Sentiment Classification

## Overview
This project builds a binary NLP classifier that predicts whether a tweet expresses **pro-vaccination** (`1`) or **non-support/other** (`0`) sentiment. The solution is trained on labeled social media text and produces a competition-style submission file for unseen tweets.

The core modeling approach uses **DistilBERT** fine-tuning through `simpletransformers`, combining transfer learning with lightweight training settings suitable for CPU-based experimentation.

## Problem Statement
Public discourse on vaccines is highly polarized and fast-moving. Automatically identifying supportive vs. non-supportive vaccine narratives can help:
- monitor public-health communication trends,
- prioritize misinformation review,
- support downstream moderation or analytics systems.

## Repository Structure
- `Train.csv`: Labeled training data (`tweet_id`, `safe_text`, `label`, `agreement`)
- `Test.csv`: Unlabeled inference set (`tweet_id`, `safe_text`)
- `submission_vaccination.csv`: Predicted labels for test data (`tweet_id`, `label`)
- `Vaccinate_or_Not_Vaccinate.ipynb`: End-to-end training, validation, and inference notebook

## Data Analysis
### Dataset schema
- **Train columns**: `tweet_id`, `safe_text`, `label`, `agreement`
- **Test columns**: `tweet_id`, `safe_text`
- **Submission columns**: `tweet_id`, `label`

### Size snapshot
- Training rows: **10,001**
- Test rows: **5,177**
- Submission rows: **5,177**

### Label semantics (as used in notebook)
- `1`: pro-vaccine
- `0`: non-support/neutral class for the binary task
- `-1`: filtered out before training/evaluation in the notebook pipeline

### Data quality notes
- Notebook profiling output indicates `tweet_id` has about **9,999 unique values** in train, suggesting a small number of duplicate IDs.
- Social text includes URL/user placeholders and mixed tone, slang, sarcasm, and strong sentiment, which makes this a realistic noisy NLP setting.

## Modeling Approach
### Model
- Backbone: `distilbert-base-uncased`
- Wrapper: `simpletransformers.classification.ClassificationModel`
- Task: Binary text classification (`num_labels=2`)

### Training configuration
- Epochs: `3`
- Learning rate: `2e-5`
- Max sequence length: `128`
- Train batch size: `16`
- Eval batch size: `16`
- Mixed precision: disabled (`fp16=False`)
- Device: CPU (`use_cuda=False`)

### Validation strategy
- Random holdout split: `train_test_split(..., test_size=0.2, random_state=42)`
- Filtering rule: rows with `label == -1` removed for both train and validation

## Results
From the notebook evaluation output:
- `MCC`: **0.6240**
- `AUROC`: **0.8932**
- `AUPRC`: **0.8636**
- `Eval loss`: **0.4398**
- Confusion terms: `TP=649`, `TN=782`, `FP=161`, `FN=168`

These numbers indicate solid separability for a compact baseline and confirm DistilBERT is a strong fit for this task.

## Reproducibility
### 1) Install dependencies
```bash
pip install transformers==4.36.2 simpletransformers==0.64.3 tokenizers==0.15.2
```

### 2) Place data files in runtime working directory
The notebook currently uses Colab-style paths:
- reads from `/content/Train.csv`, `/content/Test.csv`
- writes to `/content/submission.csv`

If running locally, update paths to relative local files or mount the same structure.

### 3) Run notebook cells in order
1. Install packages
2. Load datasets and split
3. Configure model args
4. Train on filtered labels (`label != -1`)
5. Evaluate on filtered validation set
6. Predict test labels and export submission

## Engineering Assessment
### Strengths
- Clear end-to-end baseline from preprocessing to export.
- Sensible transformer choice for noisy short-text data.
- Useful validation metrics beyond accuracy (MCC, AUROC, AUPRC).

## Example Portfolio Framing
This project demonstrates:
- practical transformer fine-tuning for social NLP,
- handling ambiguous labels and noisy user-generated text,
- metric-driven model evaluation,
- delivery of competition-ready predictions.


## Author
Nisy Adjei Acheampong  
ML/AI Engineer
