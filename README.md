# ml-fake-news-detection

# Fake News Detection with Logistic Regression

A notebook-based machine learning project that classifies news articles as **real** or **fake** using two simple text-derived features: article character length and word count. It explores the dataset, visualizes article characteristics, and evaluates a logistic regression classifier with stratified 10-fold cross-validation.

> **Note:** This is an educational baseline. Its reported average accuracy is 62.82%, and it uses only article length and word count—not the article's vocabulary or meaning. Treat predictions as experimental, not as a reliable fact-check.

## Project objective

Train and evaluate a binary classifier on labeled news articles:

| Label | Value |
|---|---:|
| Real | 0 |
| Fake | 1 |

The notebook also examines class balance, text-length distributions, subject categories, and the relationship between character length, word count, and label.

## Dataset

The notebook expects two CSV files in the working directory:

- `True.csv` — real articles
- `Fake.csv` — fake articles

Both files need a `text` column. The visualizations also use a `subject` column. The notebook loads the CSVs with Python's CSV engine and skips malformed rows (`on_bad_lines='skip'`). It assigns labels, combines the files, and calculates `text_length` and `word_count` from each article's `text`.

In the notebook's recorded run, after malformed rows were skipped, the combined dataset contained **44,898 articles**: **23,481 fake** and **21,417 real**. Counts may differ if your CSV files differ.

The notebook does not document the dataset's source or a download URL. Add the dataset files yourself; do not commit them if their license or size makes that unsuitable.

## Method

1. Read `True.csv` and `Fake.csv`, skipping malformed rows.
2. Assign `0` to real articles and `1` to fake articles, then concatenate the data.
3. Derive `text_length` (character count) and `word_count` (whitespace-separated token count) from `text`.
4. Standardize the two numeric features with `StandardScaler`.
5. Evaluate `LogisticRegression(max_iter=1000, random_state=42)` using shuffled, stratified 10-fold cross-validation with `random_state=42`.
6. Report accuracy, precision, recall, and F1 for each fold and their unweighted mean; generate per-fold confusion matrices and metric plots.

No text cleaning, tokenization, or text-vectorization is performed. The classifier receives only `text_length` and `word_count`.

### Evaluation caveat

In the notebook, `StandardScaler` is fit once on the complete feature matrix before the cross-validation folds are created. This lets each fold's test features influence the scaling parameters, so the evaluation has preprocessing leakage. For a stricter estimate, put the scaler and classifier in a scikit-learn `Pipeline` and pass that pipeline to cross-validation so scaling is fit on each training fold only. The metrics below reproduce the notebook's recorded run and should be read with this caveat.

## Recorded results

Mean of the 10 folds:

| Metric | Score |
|---|---:|
| Accuracy | 62.82% |
| Precision | 64.67% |
| Recall | 63.74% |
| F1-score | 64.20% |

These are the notebook's saved outputs, not a promise of results on another dataset or rerun. The notebook does not save a trained final model for inference.

## Libraries

The notebook imports:

- Python standard library: `os`, `warnings`
- `pandas`, `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn` (`LogisticRegression`, `StratifiedKFold`, `StandardScaler`, and classification metrics)
- Jupyter Notebook or JupyterLab to run the `.ipynb`

## Project structure

The notebook confirms the input filenames and creates a `charts/` directory, but does not save chart files in its recorded outputs. A compatible layout is:

```text
.
├── rmFinal (1).ipynb
├── True.csv
├── Fake.csv
└── charts/                 # created by the notebook for chart outputs
```

Rename the notebook to a simpler filename such as `fake_news_detection.ipynb` if desired; update the command below to match.

## Setup and usage

1. Install Python and Jupyter, then install the notebook libraries:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Place `True.csv` and `Fake.csv` beside the notebook.
3. Start Jupyter from that directory:

   ```bash
   jupyter notebook
   ```

4. Open `rmFinal (1).ipynb` and run the cells from top to bottom.

The notebook displays exploratory charts and cross-validation metrics inline. It does not provide a command-line interface, prediction function, model export, or automated setup of the dataset.

## Reproducibility

The notebook sets random seeds for its sample, cross-validation split, and logistic regression (`42`). The saved outputs correspond to the data available when that notebook was run. Skipped malformed records, library versions, and different CSV contents can change the results.
