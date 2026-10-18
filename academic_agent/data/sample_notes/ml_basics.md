# Machine Learning - Basics

Machine learning lets computers learn patterns from data instead of being explicitly programmed.

## Types of learning
Supervised learning uses labelled examples (classification and regression). Unsupervised learning finds structure in unlabelled data (clustering, dimensionality reduction). Reinforcement learning trains an agent through rewards and penalties.

## Overfitting and underfitting
Overfitting means the model memorises training data and performs poorly on new data (high variance). Underfitting means the model is too simple to capture the pattern (high bias). Remedies for overfitting include more data, regularisation (L1/L2), dropout, early stopping and cross-validation.

## Evaluation metrics
Accuracy is the fraction of correct predictions. Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 score is the harmonic mean of precision and recall. For imbalanced classes accuracy is misleading, so precision, recall and F1 are preferred.

## Retrieval-Augmented Generation (RAG)
RAG improves large language model answers by retrieving relevant documents from an external knowledge base and adding them to the prompt, which reduces hallucination and allows up-to-date, source-cited answers.
