
# Question-Answer Text Classification Pipeline

## Overview

This repository contains a comprehensive Natural Language Processing (NLP) pipeline designed to classify Question-Answer (QA) text into 10 distinct categories. The project systematically evaluates multiple text representations (TF-IDF, CBOW, Skip-gram) and deep learning architectures (DNNs, RNNs, LSTMs, GRUs) to identify the optimal model for multi-class text classification.

## Dataset

The dataset consists of over 150,000 QA pairs, split into training and testing sets. The data is uniformly distributed across 10 classes to prevent class imbalance issues.

* **Training Set:** 93,333 samples
* **Test Set:** 59,999 samples
* **Classes (≈10% distribution each):** Business & Finance, Computers & Internet, Education & Reference, Entertainment & Music, Family & Relationships, Health, Politics & Government, Science & Mathematics, Society & Culture, Sports.

## Preprocessing & Feature Engineering

An A/B test was conducted to determine the optimal cleaning strategy.

* **Light Cleaning:** HTML unescaping, tag removal, and lowercasing (Baseline Accuracy: 65.43%).
* **Heavy Cleaning (Selected):** HTML unescaping, punctuation/number removal, lemmatization (WordNet), and stopword removal. This approach reduced noise and yielded superior baseline performance (Accuracy: 66.10%).

### Word Representations

To capture semantic relationships, the pipeline generated multiple text vectorizations:

1. **CountVectorizer (Bag of Words):** Max features = 5,000.
2. **TF-IDF:** Max features = 5,000.
3. **Custom Word Embeddings (Gensim):**
* Continuous Bag of Words (CBOW)
* Skip-gram (Selected for deep learning models due to better rare-word handling).
* Vector size = 100, trained on 283,941 unique tokens.



## Model Architectures & Experiments

The project explores a progression of models, from traditional machine learning to advanced recurrent neural networks. Early stopping and learning rate schedulers (`ReduceLROnPlateau`) were utilized to prevent overfitting.

* **Baseline:** Logistic Regression (Tuned `C=1`).
* **Deep Neural Networks (DNN):** Tested with both TF-IDF inputs and Skip-gram embeddings. Explored variations with Batch Normalization, LeakyReLU, Max Pooling, and L2 Regularization.
* **Recurrent Neural Networks:**
* SimpleRNN and Bidirectional SimpleRNN.
* LSTM and Bidirectional LSTM (Optimized using CuDNN fast paths).
* GRU and Bidirectional GRU (SpatialDropout1D applied to embeddings).



## Results

The Bidirectional GRU achieved the highest accuracy, slightly outperforming the Bi-LSTM. Bidirectional recurrent networks consistently outperformed their unidirectional counterparts by capturing both past and future context in the text sequences.

| Model Architecture | Text Representation | Test Accuracy |
| --- | --- | --- |
| **Bidirectional GRU** | Skip-gram (Fine-tuned) | **71.44%** |
| Bidirectional LSTM | Skip-gram (Fine-tuned) | 71.22% |
| Unidirectional GRU | Skip-gram (Static) | 70.27% |
| Unidirectional LSTM | Skip-gram (Static) | 69.30% |
| Deep Neural Network | Skip-gram (Fine-tuned) | 68.91% |
| Logistic Regression | TF-IDF | 67.47% |
| Deep Neural Network | TF-IDF | 67.10% |
| Bidirectional SimpleRNN | Skip-gram (Fine-tuned) | 59.47% |
| Unidirectional SimpleRNN | Skip-gram (Static) | 52.77% |

## Key Insights

* **Embeddings over TF-IDF:** Sequence-based models utilizing fine-tuned Skip-gram embeddings outperformed static frequency-based TF-IDF matrices.
* **Bidirectionality is Crucial:** The context provided by reading the QA text in both directions significantly boosted the performance of GRU and LSTM models.
* **Vanishing Gradients in SimpleRNN:** The SimpleRNN models struggled significantly (52-59%), demonstrating the necessity of gating mechanisms (LSTM/GRU) for long text sequences (max length = 200 tokens).
