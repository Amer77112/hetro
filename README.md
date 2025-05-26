# HeteroFL: Heterogeneous Federated Learning System

This system implements a Heterogeneous Federated Learning (HeteroFL) approach that combines three different machine learning models (CatBoost, Random Forest, and XGBoost) using knowledge distillation.

## System Architecture

The HeteroFL system consists of the following components:

1. **Local Models**: Three pre-trained models (CatBoost, Random Forest, XGBoost) that are updated through federated learning
2. **Knowledge Distillation**: A technique where models learn from the ensemble's aggregated predictions
3. **Federated Learning**: An iterative process that updates local models based on the ensemble knowledge

## How It Works

1. **Initialization**:
   - Loads pre-trained models from the three directories (catboost, Random_forest, xgboost)
   - Prepares training data using a common encoding/scaling approach

2. **Federated Learning Rounds**:
   - For each round:
     - Aggregates predictions from all local models (ensemble prediction)
     - Updates each local model using knowledge distillation
     - Evaluates the performance of each local model and the ensemble
     - Saves updated models

3. **Knowledge Distillation**:
   - Uses the ensemble's soft predictions as "teacher knowledge"
   - Local models (students) learn from both hard labels and the ensemble's soft predictions
   - Balances learning from hard ground truth and soft ensemble predictions

## Features

- **Model Heterogeneity**: Supports different model architectures working together
- **Efficient Updates**: Uses smaller training iterations for model updates
- **Performance Tracking**: Tracks and visualizes model performance across rounds
- **Memory Efficiency**: Optimized for handling large datasets with memory constraints

## Usage

To run the HeteroFL system:

```bash
python heterofl_system.py
```

## Configuration

You can modify the following parameters in `heterofl_system.py`:

- `SAMPLE_SIZE`: Number of samples to use for training
- `FEDERATED_ROUNDS`: Number of federated learning rounds
- `TEMPERATURE`: Temperature parameter for knowledge distillation
- `ALPHA`: Balance between hard and soft targets

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- CatBoost
- XGBoost
- pandas
- numpy
- matplotlib

## Output

The system generates:
- Updated model files in the `heterofl_models` directory
- Performance visualization in the `heterofl_plots` directory
- Detailed logs in `heterofl_log.log` 