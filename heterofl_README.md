# HETROFL - Heterogeneous Federated Learning Implementation

This implementation uses knowledge distillation to aggregate multiple heterogeneous machine learning models (Random Forest, XGBoost, and CatBoost) trained on different data partitions for intrusion detection.

## Overview

Traditional federated learning requires all models to have the same architecture to directly average weights. This implementation addresses the challenge of combining heterogeneous models (different architectures) using knowledge distillation:

1. Each model was trained locally on a different data partition
2. A public dataset (unseen during training) is used to get soft predictions from each model
3. These predictions are aggregated using weighted averaging
4. A global model is trained using the aggregated soft labels
5. The global model captures the collective knowledge from all heterogeneous models

## Files

- `heterofl_aggregation.py`: The main script that performs the HETROFL aggregation
- `heterofl_models/`: Directory containing the final global model
- `heterofl_plots/`: Directory containing evaluation plots

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- tqdm

## Usage

1. Ensure all local models are trained and saved in their respective directories:
   - `Random_forest/random_forest_model.pkl`
   - `xgboost/xgboost_model.pkl`
   - `catboost/ensemble_model.pkl`

2. Run the HETROFL aggregation:
   ```
   python heterofl_aggregation.py
   ```

3. The script will:
   - Load all trained models
   - Generate soft predictions on a public dataset
   - Aggregate predictions using weighted averaging
   - Train a global model (logistic regression or random forest)
   - Evaluate and save the global model

## Configuration

You can adjust the following parameters in `heterofl_aggregation.py`:

- `SAMPLE_SIZE`: Number of samples to use from the public dataset (default: 100,000)
- `GLOBAL_MODEL_TYPE`: Type of global model to train ('logistic_regression' or 'random_forest')
- Model weights in `MODEL_DIRS`: Adjust weights for each model in the ensemble based on performance

## Extending

To add more heterogeneous models:

1. Add model information to the `MODEL_DIRS` dictionary
2. Ensure the model has a compatible `predict_proba()` method
3. Make sure all models use the same label encoding scheme

## Performance

The global model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC (where applicable)

Evaluation results are saved as plots in the `heterofl_plots/` directory. 


Local Models (RF, XGB, CatB) 
        ↓
Public Dataset (real/synthetic)
        ↓
Predictions → Aggregation (Soft labels)
        ↓
Global Model (Trained from soft labels)
        ↓
Repeat Rounds → Finetune → Evaluate → Save
