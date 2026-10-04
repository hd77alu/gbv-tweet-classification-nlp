# Gender-Based Violence Tweet Classification

A research project comparing five machine learning approaches for classifying tweets of gender-based violence (GBV).

Developed for **Formative Assignment 2: Research-Informed Sequential Models for NLP and Language Technologies** at the African Leadership University.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd77alu/gbv-tweet-classification-nlp/blob/main/notebook/gbv_tweet_classification.ipynb)

## Project Overview

This project investigates how effectively sequential models classify informal social media text, particularly when some violence categories have very few training examples.

We compare two TF-IDF baselines with SimpleRNN, LSTM, and BERTweet. The investigation includes exploratory data analysis, preprocessing, class weighting, pretrained representations, validation metrics, and qualitative analysis of test predictions.

**Research question:** Do models that learn from token sequences classify the five categories more effectively than lexical baselines, especially for rare classes?

## Dataset

The data comes from the [Zindi Gender-Based Violence Tweet Classification Challenge 2025](https://zindi.world/competitions/gender-based-violence-tweet-classification-challenge-2025/data).

- `dataset/Train.csv`: 39,650 labeled tweets.
- `dataset/Test.csv`: 15,581 tweets without publicly available labels.
- Original training columns: `Tweet_ID`, `tweet`, and `type`.
- Original test columns: `Tweet_ID` and `tweet`.

### Class Distribution After Cleaning

|  Label ID | Class                          | Labeled tweets |    Share |
| --------: | ------------------------------ | -------------: | -------: |
|         0 | `Harmful_Traditional_practice` |            187 |    0.48% |
|         1 | `Physical_violence`            |          5,556 |   14.18% |
|         2 | `economic_violence`            |            215 |    0.55% |
|         3 | `emotional_violence`           |            648 |    1.65% |
|         4 | `sexual_violence`              |         32,587 |   83.14% |
| **Total** |                                |     **39,193** | **100%** |

The strong imbalance makes accuracy alone insufficient: good performance on sexual violence can conceal poor recognition of rare categories.

## Exploratory Analysis and Preprocessing

The notebook examines class frequencies, duplicate tweets, and word-count and character-count distributions overall and by class.

The preparation workflow:

1. Removes eight exact duplicate tweets.
2. Decodes HTML entities, normalizes Unicode, and standardizes whitespace.
3. Audits missing values, blank text, duplicate IDs, and conflicting labels among the remaining texts.
4. Removes 449 additional duplicate rows identified after cleaning.
5. Encodes the five class labels.
6. Creates a stratified 80/20 training–validation split using seed 42.

| Split      |   Rows | Prepared columns                |
| ---------- | -----: | ------------------------------- |
| Training   | 31,354 | `Tweet_ID`, `tweet`, `label_id` |
| Validation |  7,839 | `Tweet_ID`, `tweet`, `label_id` |
| Test       | 15,581 | `Tweet_ID`, `tweet`             |

The cleaned text is stored in the `tweet` column. Learned vocabularies are fitted on the training split only.

## Models and Experiments

| Approach                         | Representation and purpose                                                                                                   |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **TF-IDF + Linear SVM**          | Word unigrams and bigrams with a class-weighted linear classifier and sigmoid probability calibration.                       |
| **TF-IDF + Logistic Regression** | A probabilistic lexical baseline using the same TF-IDF settings and balanced class weights.                                  |
| **SimpleRNN**                    | Trainable word embeddings followed by a recurrent layer; tested with and without class weighting.                            |
| **LSTM**                         | A gated recurrent model; experiments include a baseline, regularization, GloVe initialization, and bidirectional processing. |
| **BERTweet**                     | Fine-tuning `vinai/bertweet-base`, a Transformer pretrained on English tweets.                                               |

SimpleRNN and LSTM use **TensorFlow/Keras**. BERTweet uses **PyTorch and Hugging Face Transformers**. The lexical baselines use **scikit-learn**.

The LSTM variants are experiments within the LSTM approach, rather than additional model families.

BERTweet inputs are limited to 128 tokens. Training token lengths have a median of 54 and a 99th percentile of 77; approximately 0.01% exceed the limit.

## Validation Results

The following values come from saved notebook outputs. **These are validation results, not test scores.**

| Model / configuration          | Validation accuracy | Validation macro F1 |
| ------------------------------ | ------------------: | ------------------: |
| TF-IDF + calibrated Linear SVM |              99.91% |              0.9979 |
| TF-IDF + Logistic Regression   |              96.07% |              0.7216 |
| SimpleRNN — unweighted         |              98.24% |              0.7196 |
| SimpleRNN — weighted           |              98.67% |              0.7501 |
| LSTM — baseline                |              99.55% |              0.9443 |
| LSTM — regularized             |              98.98% |              0.9393 |
| LSTM — GloVe embeddings        |              99.46% |              0.9609 |
| Bidirectional LSTM             |              99.45% |              0.9529 |
| BERTweet — baseline            |              99.99% |              0.9992 |

**Macro F1** is the main comparison metric because it gives each class equal importance. The notebook also reports per-class metrics, balanced accuracy, weighted F1, and multiclass ROC-AUC where available, alongside confusion matrices and learning curves.

### Main Findings

- BERTweet achieved the highest recorded validation macro F1, with one error among 7,839 validation examples.
- The TF-IDF + SVM baseline was also highly competitive. These results do not establish that complex sequence models are necessary for strong performance on this split.
- Class weighting increased SimpleRNN macro F1 from 0.7196 to 0.7501, but economic violence F1 decreased from 0.44 to 0.36.
- GloVe initialization produced the highest recorded macro F1 among the LSTM experiments.
- Training budgets, representations, and checkpoint-selection rules differ. BERTweet selects its best epoch by validation macro F1; SimpleRNN and the regularized LSTM variants select by validation loss. The baseline LSTM's recorded evaluation uses its final epoch.

## Test Predictions and Limitations

The test set is unlabeled, so its prediction distributions are descriptive. Test accuracy, macro F1, and a test confusion matrix cannot be calculated from these files.

Qualitative inspection identified concerning predictions:

- BERTweet and SimpleRNN classified some stabbing-only tweets as sexual violence.
- SimpleRNN classified a tweet explicitly mentioning forced marriage, child marriage, and FGM as sexual violence.
- Some questionable predictions had very high confidence.

Additional limitations include:

- Very small validation samples for harmful traditional practice (37) and economic violence (43).
- A random stratified split that may retain related topics or near-duplicate wording across partitions.
- One recorded run per configuration in the comparison above.
- Ambiguous, mixed-category, humorous, fictional, or potentially out-of-scope tweets.
- Conflict checks performed after the initial exact-duplicate removal.

Further work should include annotation review, checks for near-duplicate overlap, repeated seeds, and evaluation on independently labeled data.

## Repository Structure

- `dataset/` — original training and test CSV files.
- `notebook/gbv_tweet_classification.ipynb` — EDA, preprocessing, experiments, results, and interpretation.
- `models/` — saved model artifacts.
- `README.md` — project overview and usage guidance.

## Running the Notebook

The notebook is designed for **Google Colab**.

1. Open it using the Colab button above.
2. Upload `Train.csv` and `Test.csv` to the runtime.
3. Select a GPU runtime for BERTweet; the recorded run used a Tesla T4.
4. Run the imports, data investigation, and preprocessing cells before the model sections.
5. Allow downloads of BERTweet and GloVe resources when running those experiments.
6. Run Google Drive backup cells only if you want persistent copies of runtime outputs.

Dependencies include pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorFlow, PyTorch, Transformers, emoji, and joblib. The notebook includes an installation cell for Transformers and emoji.

## Saved Models

| File                                    | Description                                               |
| --------------------------------------- | --------------------------------------------------------- |
| `models/tfidf_svm_pipeline.joblib`      | TF-IDF preprocessing and calibrated SVM pipeline.         |
| `models/tfidf_lr_pipeline.joblib`       | TF-IDF preprocessing and Logistic Regression pipeline.    |
| `models/Simple RNN_weighted.keras`      | Weighted SimpleRNN with its text-vectorization layer.     |
| `models/glove_lstm_model.keras`         | GloVe-initialized LSTM with its text-vectorization layer. |
| `models/tweet_transformer_model.joblib` | BERTweet model tracked through Git LFS.                   |

Use `joblib.load(...)` for the scikit-learn pipelines and `tf.keras.models.load_model(...)` for the Keras models.

The BERTweet artifact is approximately 540 MB. Retrieve its contents using Git LFS rather than treating the small pointer file as the model:

```bash
git clone https://github.com/hd77alu/gbv-tweet-classification-nlp.git
cd gbv-tweet-classification-nlp
git lfs install
git lfs pull
```

BERTweet inference also requires the matching tokenizer and compatible PyTorch/Transformers packages.

**Artifact status:** `models/simpleRNN_baseline.keras` currently contains only a blank placeholder and cannot be loaded as a trained model. Retrain the baseline or use a separately saved copy.

## Responsible Use

This is an academic research prototype. It should not be used to make automated decisions about victims, incidents, or individuals.

Tweets may contain sensitive descriptions of violence. Predictions require human interpretation, and the five-label setup does not provide an explicit “not GBV” or “uncertain” category.

Consult the source challenge's terms before reusing or redistributing the dataset.

## References

- [Zindi: Gender-Based Violence Tweet Classification Challenge 2025](https://zindi.world/competitions/gender-based-violence-tweet-classification-challenge-2025/data).
- Elman, J. L. (1990). [Finding structure in time](https://doi.org/10.1207/s15516709cog1402_1).
- Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory](https://doi.org/10.1162/neco.1997.9.8.1735).
- Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global vectors for word representation](https://aclanthology.org/D14-1162/).
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/).
- [BERTweet pretrained checkpoint](https://huggingface.co/vinai/bertweet-base).

Further related work and methodological discussion are included in the notebook.

## Authors

This project was conducted as a collaborative research study by:

- **Shalom Amaliza**
- **Yvette Uwimpaye**
- **Gaddiel Irakoze**
- **Hamed Alfatih**
