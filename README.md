# Telecom Customer Churn – Predictive Modeling (ANN)

This repository contains the predictive modeling deliverables for a telecommunications customer churn project. The focus of Stage 3 is on developing, training, and evaluating an Artificial Neural Network (ANN) model to predict customer churn, and packaging all outputs as a structured deliverable.

## Repository Structure

```text
Predictive_Modeling/
│
├── ANN_Architecture_Details.pdf       # Document describing full ANN architecture
│
├── Trained_ANN_Model/
│   ├── model_training_script.py       # End-to-end script to train and evaluate the ANN
│   ├── churn_ann_model.keras          # Saved trained ANN model
│   ├── churn_ann_test_predictions.csv # Test set predictions (y_true, y_pred, y_proba)
│   └── README_Model_Usage.txt         # How to load and use the model
│
└── Visualisations/
    ├── training_loss_curve.png        # Training vs validation loss
    ├── training_auc_curve.png         # Training vs validation ROC-AUC
    ├── roc_curve.png                  # ROC curve on test set
    └── confusion_matrix.png           # Confusion matrix for test predictions
