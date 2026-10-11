# Semantic Feature Tagging with RoBERTa

This repository contains the code for an AIML339 research project investigating **multi-label semantic feature tagging of English nouns and proper nouns**. It constructs a labelled dataset using Universal Dependencies (UD) English Web Treebank (EWT), WordNet word-sense disambiguation (WSD), supplementary LLM annotation, and a semantic-feature inheritance hierarchy. A fine-tuned **RoBERTa-large** model predicts semantic features at token level. A **GLiNER2.5** approach is evaluated as a comparison baseline.


## 1. Repository structure

The repository contains a `notebooks/` directory.

| Stage | Notebook or script |
|---|---|
| Dataset construction / WordNet WSD / LLM annotation | AIML339 Selectional Restriction Tagger.ipynb |
| RoBERTa-large fine-tuning and repeated evaluation | AIML339 RoBERTa fine tune.ipynb |
| GLiNER2.5 baseline | AIML GliNER2.5 classifier.ipynb |

Trained weights, caches and large processed datasets are excluded from the repository.

## 2. Requirements and installation

- **Python:** 3.9.21-compatible environment recommended;
- **GPU:** CUDA-capable NVIDIA GPU recommended for RoBERTa-large and GLiNER2.5. CPU execution is possible for some preparation steps but model training will be substantially slower.
- **External resources:** UD English EWT, NLTK WordNet, pretrained `FacebookAI/roberta-large`, and the LLM used for WSD/supplementary labelling.

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a **CUDA-compatible PyTorch build** using the instructions at https://pytorch.org/get-started/locally/ and then install the project's other dependencies.

```bash
python -m pip install transformers datasets accelerate scikit-learn \
    numpy pandas scipy nltk spacy conllu tqdm matplotlib \
    jupyter ipykernel gliner2[local] openai iterative-stratification
python -m spacy download en_core_web_sm
```

Download WordNet data:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```



## 3. Data preparation

### 3.1 Obtain the source corpus

Download the **UD English EWT** treebank from its official repository:

https://github.com/UniversalDependencies/UD_English-EWT

Keep its original training and test partitions distinct. The data-processing notebooks expect paths to the relevant `.conllu` files; set those paths in the notebooks before running them.

### 3.2 Generate semantic annotations

Run the data-preparation notebook: **AIML339 Selectional Restriction Tagger.ipynb**

## 4. Train RoBERTa-large

Open **AIML339 RoBERTa fine tune.ipynb** and run the first cell after adjusting the input and output directories. The training implementation:


- Creates a **sentence-level** training/validation split from the original EWT training partition; keeps the original EWT test partition held out until final evaluation.
-- Split seed: 42
-- Training set: 8,885 sentences, 34,895 labelled tokens: 25,277 nouns, 9,618 proper nouns
-- Validation set: 2313 sentences, 8,733 labelled tokens: 6,356 nouns, 2,377 proper nouns
-- Test set: 1759 sentences, 5,704 labelled tokens: 3,712 nouns, 1,992 proper nouns

- Initialises the pretrained `FacebookAI/roberta-large` backbone.
- Builds token-aligned, multi-label targets for eligible `NOUN`/`PROPN` tokens.
- Trains the **independent semantic features** and restores inherited labels via deterministic hierarchy expansion.
- Uses validation metrics to select checkpoints and calibrates **per-feature decision thresholds on validation data only**.
- Repeats training with the fixed list of 30 seeds below and aggregates the test results.

```python
RUN_SEEDS = [
    11, 23, 37, 53, 71, 89, 107, 127, 149, 173,
    197, 223, 251, 281, 313, 347, 383, 421, 461, 503,
    547, 593, 641, 691, 743, 797, 853, 911, 971, 1031,
]
```

The reusable inference checkpoint is expected under a path such as `../models/roberta_semantic_features/reference_model`, with `semantic_features.json` and `prediction_thresholds.json` alongside it.
### Training configuration for the main reported experiment

| Parameter | Main experiment |
|---|---|
| Base checkpoint | `FacebookAI/roberta-large` |
| Task | Multi-label token classification (`NOUN` / `PROPN`) |
| Training targets | Independent feature labels, followed by deterministic hierarchy expansion |
| Random seeds | The 30 values listed above |
| Split | Sentence-level; validation derived from EWT train; EWT test held out |
| Learning rate | 4.0e-05 |
| Per-device training batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Epochs | 5 |
| Warm-up and weight decay | 278 and 0.01 |
| Threshold calibration | Per-feature, validation-only |
| Disabled features (if any) | "State", "State_of_Matter", "Shape", "Rigidity" |


## 5. Evaluation

### 5.1 RoBERTa

The first cell in **AIML339 RoBERTa fine tune.ipynb** also reports the evaluation metrics.

- Evaluates each run's validation-selected checkpoint on the held-out EWT test set.
- Reports the **0.5 threshold** results as a diagnostic and the **validation-calibrated threshold** results as the primary evaluation.
- Restores hierarchy inherited features before evaluating the *complete-hierarchy* metric; retain *independent-feature* metrics as a separate informational tool.
- Computes micro and macro precision, recall and F1, with per-feature breakdowns.
- Aggregates scores across the same 30 seeds, reporting **mean ± standard deviation** and computes sentence-cluster bootstrap confidence intervals where reported.


### 5.2 GLiNER2.5 baseline

Open **AIML GliNER2.5 classifier.ipynb** and run the baseline with the **same held-out test records, eligible token criteria, feature inventory and hierarchy** used for RoBERTa. The baseline uses validation data for any threshold calibration.

### 5.3 Interpreting the reported metrics

| Metric | Interpretation |
|---|---|
| Micro F1 | Aggregates true positives, false positives and false negatives across features |
| Macro F1 | Unweighted mean of per-feature F1 scores; gives uncommon features equal importance |
| Mean ± SD | Variation between training runs with different random seeds |
| Sentence-cluster bootstrap CI | Sampling uncertainty estimated by resampling sentences, keeping token dependence within each sentence |

## 6. Matching the final report results

| Report result | System | Experiment file | Output figure |
|---|---|---|---|
| Main 30-seed performance | RoBERTa-large | **AIML339 RoBERTa fine tune.ipynb** | REPEATED-RUN TEST SUMMARY table |
| Per-feature performance | RoBERTa-large | **AIML339 RoBERTa fine tune.ipynb** | REPEATED-RUN TEST PERFORMANCE BY COMPLETE FEATURE (AFTER INHERITANCE) table |
| Baseline comparison | GLiNER2.5 | **AIML GliNER2.5 classifier.ipynb** | TEST PERFORMANCE WITH VALIDATION-OPTIMISED THRESHOLDS — COMPLETE HIERARCHY table and TEST PERFORMANCE BY COMPLETE FEATURE (AFTER INHERITANCE) table |

## 7. Data availability and licensing

UD English EWT, WordNet and the pretrained model are externally maintained resources with their own licences or distribution terms. They are not included in this repository.

## 8. Acknowledgements

- [Universal Dependencies — English EWT](https://github.com/UniversalDependencies/UD_English-EWT)
- [Princeton WordNet](https://wordnet.princeton.edu/)
- [RoBERTa / Hugging Face Transformers](https://huggingface.co/FacebookAI/roberta-large)
- [GLiNER](https://github.com/urchade/GLiNER) — confirm the exact GLiNER2.5 implementation used

See the accompanying AIML339 final report for methodological discussion, experimental findings and full academic references.
