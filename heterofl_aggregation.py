import os
import pandas as pd
import numpy as np
import pickle
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import gc
import uuid

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, 'heterofl_models')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'heterofl_plots')
DEBUG_DIR = os.path.join(SCRIPT_DIR, 'heterofl_debug')
GLOBAL_MODEL_FILENAME = 'heterofl_global_model.pkl'

# Federated Learning / Aggregation Rounds
NUM_ROUNDS = 5 # Define the number of aggregation rounds

# Dataset preference
DATASET_PREFERENCE = 'auto' # Options: 'auto' (try real, then synthetic), 'real' (require real), 'synthetic' (use only synthetic)

# Feature matching options
USE_SYNTHETIC_DATA = True
SYNTHETIC_DATA_SIZE = 50000
SKIP_ZERO_COLS_REMOVAL = True

# Finetuning options
FINETUNE_LOCAL_MODELS = True
FINETUNE_GLOBAL_MODEL = True
FINETUNE_DATA_SIZE = 1000000
FINETUNE_EPOCHS = 3

# Model weights
MODEL_WEIGHTS = {
    'random_forest': 1.0,
    'xgboost': 1.0,
    'catboost': 1.0
}

# Model directories and filenames
MODEL_DIRS = {
    'random_forest': {
        'dir': os.path.join(SCRIPT_DIR, 'Random_forest'),
        'model': 'random_forest_model.pkl',
        'encoder': 'label_encoder.pkl',
        'scaler': 'scaler.pkl',
        'weight': 1.0
    },
    'xgboost': {
        'dir': os.path.join(SCRIPT_DIR, 'xgboost'),
        'model': 'xgboost_model.pkl',
        'encoder': 'label_encoder.pkl',
        'scaler': 'scaler.pkl',
        'weight': 1.0
    },
    'catboost': {
        'dir': os.path.join(SCRIPT_DIR, 'catboost'),
        'model': 'ensemble_model.pkl',
        'encoder': 'label_encoder.pkl',
        'scaler': 'scaler.pkl',
        'weight': 1.0
    }
}

# Dataset for knowledge distillation
DATASET_PATH = os.path.join(SCRIPT_DIR, 'data', 'NF-ToN-IoT-v3-cleaned.csv')
SAMPLE_SIZE = 1000000
TARGET_COLUMN = 'Attack'
COLUMNS_TO_DROP = ['Label']

# Global model parameters
GLOBAL_MODEL_TYPE = 'random_forest'
GLOBAL_MODEL_PARAMS = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 2000,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced_subsample',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1
    }
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'heterofl_aggregation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_model_and_artifacts(model_info):
    """Load a trained model and its associated artifacts."""
    model_path = os.path.join(model_info['dir'], model_info['model'])
    encoder_path = os.path.join(model_info['dir'], model_info['encoder'])
    scaler_path = os.path.join(model_info['dir'], model_info['scaler'])
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from: {model_path}")
        
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        logger.info(f"Loaded label encoder from: {encoder_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from: {scaler_path}")
        
        return model, encoder, scaler
    except Exception as e:
        logger.error(f"Error loading model and artifacts: {e}")
        return None, None, None

def load_public_dataset(dataset_path, sample_size):
    """Load and prepare the public dataset for knowledge distillation."""
    logger.info(f"Loading public dataset from: {dataset_path}")
    try:
        # Check if file exists
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found at {dataset_path}")
            if USE_SYNTHETIC_DATA and DATASET_PREFERENCE != 'real':
                logger.info("Falling back to synthetic dataset")
                return None, None
            else:
                logger.error("Synthetic data fallback disabled or dataset preference set to 'real'. Exiting...")
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
        # Read a sample of the dataset
        df = pd.read_csv(dataset_path, nrows=sample_size)
        logger.info(f"Loaded public dataset with shape: {df.shape}")
        
        # Drop columns that weren't used during training
        for col in COLUMNS_TO_DROP:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Dropped column: {col}")
        
        if TARGET_COLUMN not in df.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in the public dataset")
            return None, None
        
        # Split features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        logger.info(f"Public dataset features shape: {X.shape}")
        logger.info(f"Class distribution in public dataset:")
        for cls, count in y.value_counts().items():
            logger.info(f"  {cls}: {count:,} ({count/len(y)*100:.2f}%)")
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading public dataset: {e}")
        if USE_SYNTHETIC_DATA and DATASET_PREFERENCE != 'real':
            logger.info("Falling back to synthetic dataset due to error")
            return None, None
        else:
            logger.error("Synthetic data fallback disabled or dataset preference set to 'real'. Exiting...")
            raise

def save_debug_info(models_data, X_public):
    """Save debug information about feature matching"""
    create_dir_if_not_exists(DEBUG_DIR)
    
    # Save the public dataset feature names
    with open(os.path.join(DEBUG_DIR, 'public_dataset_features.txt'), 'w') as f:
        f.write("Public Dataset Features:\n")
        for i, feature in enumerate(X_public.columns):
            f.write(f"{i+1}. {feature}\n")
    
    # Save each model's expected features
    for model_name, model_info in models_data.items():
        model = model_info['model']
        if hasattr(model, 'feature_names_in_'):
            with open(os.path.join(DEBUG_DIR, f'{model_name}_features.txt'), 'w') as f:
                f.write(f"{model_name.upper()} Model Expected Features:\n")
                for i, feature in enumerate(model.feature_names_in_):
                    f.write(f"{i+1}. {feature}\n")
        else:
            logger.warning(f"{model_name} model does not have feature_names_in_ attribute")
    
    logger.info(f"Saved feature debugging information to {DEBUG_DIR}")

def preprocess_public_dataset(X_public, y_public):
    """Apply additional preprocessing to the public dataset if needed"""
    logger.info("Applying additional preprocessing to the public dataset")
    
    # Type conversion
    for col in X_public.select_dtypes(include=['float64']).columns:
        X_public[col] = X_public[col].astype('float32')
    for col in X_public.select_dtypes(include=['int64']).columns:
        X_public[col] = X_public[col].astype('int32')
    
    # Check for and remove any columns with all zeros or NaNs
    if not SKIP_ZERO_COLS_REMOVAL:
        zero_cols = [col for col in X_public.columns if X_public[col].sum() == 0]
        if zero_cols:
            logger.warning(f"Removing columns with all zeros: {zero_cols}")
            X_public = X_public.drop(columns=zero_cols)
    
    # Fill any remaining NaNs
    if X_public.isnull().sum().sum() > 0:
        logger.warning(f"Filling {X_public.isnull().sum().sum()} NaN values with 0")
        X_public = X_public.fillna(0)
    
    return X_public, y_public

def fix_feature_names(X, expected_features):
    """Fix feature names to match expected features by closest string matching."""
    logger.info("Attempting to fix feature names by string similarity")
    
    def find_closest_match(feature, all_features):
        best_match = None
        best_score = 0
        
        for f in all_features:
            score = sum(c1 == c2 for c1, c2 in zip(feature.lower(), f.lower())) / max(len(feature), len(f))
            if score > best_score:
                best_score = score
                best_match = f
        
        return best_match if best_score > 0.6 else None
    
    X_fixed = pd.DataFrame()
    feature_mapping = {}
    for expected in expected_features:
        if expected in X.columns:
            feature_mapping[expected] = expected
        else:
            closest = find_closest_match(expected, X.columns)
            if closest:
                feature_mapping[expected] = closest
                logger.info(f"Mapped '{expected}' to '{closest}'")
            else:
                feature_mapping[expected] = None
                logger.warning(f"No match found for '{expected}', will use zeros")
    
    for expected in expected_features:
        if feature_mapping[expected]:
            X_fixed[expected] = X[feature_mapping[expected]]
        else:
            X_fixed[expected] = 0.0
    
    return X_fixed

def align_features(model, X_public, model_name):
    """Ensure the features in X_public match what the model expects."""
    try:
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
            logger.info(f"{model_name} model expects {len(model_features)} features")
            
            missing_features = [f for f in model_features if f not in X_public.columns]
            if missing_features:
                logger.warning(f"Features required by {model_name} but missing from data: {len(missing_features)} features")
                if len(missing_features) > 0 and len(missing_features) <= 5:
                    logger.warning(f"Missing features: {missing_features}")
                
            extra_features = [f for f in X_public.columns if f not in model_features]
            if extra_features:
                logger.warning(f"Features in data but not used by {model_name}: {len(extra_features)} features")
                if len(extra_features) > 0 and len(extra_features) <= 5:
                    logger.warning(f"Extra features: {extra_features}")
                
            if len(missing_features) == 0:
                logger.info(f"Perfect feature match for {model_name}, reordering columns")
                X_aligned = X_public[model_features]
                return X_aligned
            elif len(missing_features) <= 5:
                logger.info(f"Adding {len(missing_features)} missing features with zeros for {model_name}")
                X_aligned = pd.DataFrame()
                for feature in model_features:
                    if feature in X_public.columns:
                        X_aligned[feature] = X_public[feature]
                    else:
                        X_aligned[feature] = 0.0
                return X_aligned
            else:
                logger.warning(f"Too many missing features ({len(missing_features)}) for {model_name}, attempting fuzzy matching")
                X_aligned = fix_feature_names(X_public, model_features)
                return X_aligned
        
        logger.warning(f"{model_name} does not have feature_names_in_ attribute. Using all features.")
        return X_public
    
    except Exception as e:
        logger.error(f"Error aligning features for {model_name}: {e}")
        return None

def get_model_predictions(models_data, X_public):
    """Get predictions from all models on the public dataset."""
    predictions = {}
    
    for model_name, model_info in models_data.items():
        model, encoder, scaler = model_info['model'], model_info['encoder'], model_info['scaler']
        weight = MODEL_WEIGHTS.get(model_name, 1.0)
        
        logger.info(f"Getting predictions from {model_name} model with weight {weight}...")
        
        try:
            X_scaled = X_public.copy()
            X_scaled_values = scaler.transform(X_scaled)
            X_scaled = pd.DataFrame(X_scaled_values, columns=X_scaled.columns)
            
            X_aligned = align_features(model, X_scaled, model_name)
            
            if X_aligned is None:
                logger.error(f"Failed to align features for {model_name} model. Skipping.")
                continue
            
            logger.info(f"{model_name} model will use {len(X_aligned.columns)} features")
            
            if model_name == 'xgboost':
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_aligned)
                    elif hasattr(model, 'predict'):
                        margins = model.predict(X_aligned, output_margin=True)
                        if isinstance(margins, np.ndarray) and len(margins.shape) == 1:
                            probs = 1.0 / (1.0 + np.exp(-margins))
                            y_prob = np.vstack([1-probs, probs]).T
                        else:
                            exp_margins = np.exp(margins - np.max(margins, axis=1, keepdims=True))
                            y_prob = exp_margins / np.sum(exp_margins, axis=1, keepdims=True)
                    else:
                        logger.error(f"{model_name} model has no suitable prediction method")
                        continue
                except Exception as e:
                    logger.error(f"Error getting XGBoost predictions: {e}")
                    continue
            else:
                y_prob = model.predict_proba(X_aligned)
            
            predictions[model_name] = {
                'probabilities': y_prob,
                'weight': weight,
                'classes': encoder.classes_
            }
            
            logger.info(f"Got predictions from {model_name}, shape: {y_prob.shape}")
            
            pred_classes = np.argmax(y_prob, axis=1)
            class_counts = np.bincount(pred_classes, minlength=y_prob.shape[1])
            logger.info(f"{model_name} prediction distribution: {dict(enumerate(class_counts))}")
            
        except Exception as e:
            logger.error(f"Error getting predictions from {model_name}: {e}")
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_.tolist() if hasattr(model.feature_names_in_, 'tolist') else model.feature_names_in_
                logger.error(f"Model feature names required ({len(model_features)}): {model_features}")
                logger.error(f"Dataset columns provided ({len(X_public.columns)}): {X_public.columns.tolist()}")
                
                create_dir_if_not_exists(DEBUG_DIR)
                with open(os.path.join(DEBUG_DIR, f'{model_name}_feature_comparison.txt'), 'w') as f:
                    f.write(f"=== {model_name.upper()} FEATURE COMPARISON ===\n\n")
                    f.write(f"Model requires {len(model_features)} features:\n")
                    for i, feature in enumerate(model_features):
                        f.write(f"{i+1}. {feature}\n")
                    
                    f.write(f"\nDataset provides {len(X_public.columns)} features:\n")
                    for i, feature in enumerate(X_public.columns):
                        f.write(f"{i+1}. {feature}\n")
                    
                    f.write("\nFeature differences:\n")
                    in_model_not_data = [f for f in model_features if f not in X_public.columns]
                    in_data_not_model = [f for f in X_public.columns if f not in model_features]
                    
                    f.write(f"Features required by model but missing from data ({len(in_model_not_data)}):\n")
                    for i, feature in enumerate(in_model_not_data):
                        f.write(f"{i+1}. {feature}\n")
                    
                    f.write(f"\nFeatures in data but not used by model ({len(in_data_not_model)}):\n")
                    for i, feature in enumerate(in_data_not_model):
                        f.write(f"{i+1}. {feature}\n")
    
    return predictions

def aggregate_predictions(predictions):
    """Aggregate predictions from all models using weighted averaging."""
    if not predictions:
        logger.error("No predictions to aggregate")
        return None, None, None
    
    first_model = list(predictions.keys())[0]
    classes = predictions[first_model]['classes']
    n_samples = predictions[first_model]['probabilities'].shape[0]
    n_classes = len(classes)
    
    logger.info(f"Aggregating predictions for {n_samples} samples across {n_classes} classes")
    
    aggregated = np.zeros((n_samples, n_classes))
    total_weight = 0
    
    for model_name, model_data in predictions.items():
        weight = model_data['weight']
        probas = model_data['probabilities']
        aggregated += weight * probas
        total_weight += weight
    
    if total_weight > 0:
        aggregated /= total_weight
    
    hard_labels = np.argmax(aggregated, axis=1)
    class_counts = np.bincount(hard_labels, minlength=n_classes)
    logger.info(f"Aggregated prediction distribution: {dict(enumerate(class_counts))}")
    
    return aggregated, hard_labels, classes

def balance_predictions(soft_labels):
    """Balance the predictions to ensure all classes are represented."""
    n_samples, n_classes = soft_labels.shape
    hard_labels = np.argmax(soft_labels, axis=1)
    
    class_counts = np.bincount(hard_labels, minlength=n_classes)
    logger.info(f"Original class distribution: {dict(enumerate(class_counts))}")
    
    min_samples_per_class = max(50, n_samples // (n_classes * 5))
    balanced_soft_labels = soft_labels.copy()
    
    for cls in range(n_classes):
        if class_counts[cls] < min_samples_per_class:
            samples_to_add = min_samples_per_class - class_counts[cls]
            logger.info(f"Adding {samples_to_add} synthetic samples for class {cls}")
            
            majority_indices = np.where(hard_labels == np.argmax(class_counts))[0]
            
            if len(majority_indices) >= samples_to_add:
                indices_to_convert = np.random.choice(majority_indices, samples_to_add, replace=False)
            else:
                indices_to_convert = np.random.choice(n_samples, samples_to_add, replace=False)
            
            for idx in indices_to_convert:
                balanced_soft_labels[idx] = np.zeros(n_classes)
                balanced_soft_labels[idx, cls] = 1.0
    
    new_hard_labels = np.argmax(balanced_soft_labels, axis=1)
    new_class_counts = np.bincount(new_hard_labels, minlength=n_classes)
    logger.info(f"Balanced class distribution: {dict(enumerate(new_class_counts))}")
    
    return balanced_soft_labels

def train_global_model(X_public, soft_labels, model_type, model_params):
    """Train a global model using soft labels from knowledge distillation."""
    logger.info(f"Training global {model_type} model...")
    
    balanced_soft_labels = balance_predictions(soft_labels)
    hard_labels = np.argmax(balanced_soft_labels, axis=1)
    
    unique_classes = np.unique(hard_labels)
    logger.info(f"Number of unique classes in aggregated predictions: {len(unique_classes)}")
    logger.info(f"Class distribution: {pd.Series(hard_labels).value_counts().to_dict()}")
    
    if len(unique_classes) < 2:
        logger.warning("Only one class found in predictions. Adding synthetic classes for training.")
        n_classes = balanced_soft_labels.shape[1]
        for cls in range(n_classes):
            if cls not in unique_classes:
                n_synthetic = min(500, X_public.shape[0] // 10)
                indices = np.random.choice(X_public.shape[0], n_synthetic, replace=False)
                for idx in indices:
                    balanced_soft_labels[idx] = np.zeros(n_classes)
                    balanced_soft_labels[idx, cls] = 1.0
        
        hard_labels = np.argmax(balanced_soft_labels, axis=1)
        logger.info(f"After adding synthetic classes: {len(np.unique(hard_labels))} classes")
        logger.info(f"New class distribution: {pd.Series(hard_labels).value_counts().to_dict()}")
    
    if model_type == 'logistic_regression':
        global_model = LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        global_model = RandomForestClassifier(**model_params)
    else:
        logger.error(f"Unsupported global model type: {model_type}")
        return None
    
    try:
        class_weights = compute_sample_weight(class_weight='balanced', y=hard_labels)
        logger.info(f"Using sample weights to balance training")
        global_model.fit(X_public, hard_labels, sample_weight=class_weights)
        logger.info(f"Global model trained successfully")
        return global_model
    except Exception as e:
        logger.error(f"Error training global model: {e}")
        try:
            logger.info(f"Trying training without sample weights")
            global_model.fit(X_public, hard_labels)
            logger.info(f"Global model trained successfully without sample weights")
            return global_model
        except Exception as e2:
            logger.error(f"Error training global model without sample weights: {e2}")
            return None

def evaluate_global_model(global_model, X_test, y_test, classes):
    """Evaluate the global model on a test dataset."""
    logger.info("Evaluating global model performance...")
    
    y_pred = global_model.predict(X_test)
    y_prob = global_model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    if len(np.unique(y_test)) == 2:
        metrics['auc'] = roc_auc_score(y_test, y_prob[:, 1])
    else:
        try:
            y_test_bin = np.zeros((len(y_test), len(classes)))
            for i, c in enumerate(y_test):
                y_test_bin[i, c] = 1
            metrics['auc'] = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("GLOBAL MODEL PERFORMANCE METRICS")
    logger.info("="*50)
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    logger.info("="*50)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm

def plot_and_save_results(metrics, cm, classes):
    """Generate and save plots for evaluation results."""
    create_dir_if_not_exists(PLOTS_DIR)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Global Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(PLOTS_DIR, 'global_confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to: {confusion_matrix_path}")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim([0, 1.0])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Global Model Performance Metrics')
    plt.tight_layout()
    
    metrics_path = os.path.join(PLOTS_DIR, 'global_metrics.png')
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    logger.info(f"Saved metrics plot to: {metrics_path}")

def save_global_model(global_model, encoder):
    """Save the global model and label encoder."""
    create_dir_if_not_exists(SAVE_DIR)
    
    global_model_path = os.path.join(SAVE_DIR, GLOBAL_MODEL_FILENAME)
    try:
        with open(global_model_path, 'wb') as f:
            pickle.dump(global_model, f)
        logger.info(f"Global model saved to: {global_model_path}")
    except Exception as e:
        logger.error(f"Error saving global model: {e}")
    
    encoder_path = os.path.join(SAVE_DIR, 'global_label_encoder.pkl')
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        logger.info(f"Label encoder saved to: {encoder_path}")
    except Exception as e:
        logger.error(f"Error saving label encoder: {e}")

def create_synthetic_dataset(models_data, sample_size=SYNTHETIC_DATA_SIZE):
    """Create a synthetic dataset based on the feature requirements of all models."""
    logger.info(f"Creating synthetic dataset with {sample_size} samples")
    
    all_features = set()
    model_with_features = None
    for model_name, model_info in models_data.items():
        if hasattr(model_info['model'], 'feature_names_in_'):
            model_with_features = (model_name, model_info['model'])
            all_features.update(model_info['model'].feature_names_in_)
    
    if not all_features:
        logger.error("No model with feature_names_in_ attribute found")
        return None, None
    
    features = list(all_features)
    logger.info(f"Using {len(features)} unique features from all models: {features}")
    
    np.random.seed(42)
    X_synthetic = pd.DataFrame(np.random.randn(sample_size, len(features)), columns=features)
    
    scaler = StandardScaler()
    X_synthetic = pd.DataFrame(scaler.fit_transform(X_synthetic), columns=features)
    
    num_classes = len(models_data[list(models_data.keys())[0]]['encoder'].classes_)
    samples_per_class = sample_size // num_classes
    balanced_classes = []
    
    for cls in range(num_classes):
        balanced_classes.extend([cls] * samples_per_class)
    
    remaining = sample_size - len(balanced_classes)
    if remaining > 0:
        balanced_classes.extend(np.random.randint(0, num_classes, size=remaining))
    
    np.random.shuffle(balanced_classes)
    y_synthetic = pd.Series(balanced_classes)
    
    logger.info(f"Created synthetic dataset with {X_synthetic.shape[1]} features and {X_synthetic.shape[0]} samples")
    logger.info(f"Synthetic target distribution: {y_synthetic.value_counts().to_dict()}")
    
    create_dir_if_not_exists(DEBUG_DIR)
    sample_df = pd.DataFrame(X_synthetic.head(100))
    sample_df['target'] = y_synthetic.head(100)
    sample_df.to_csv(os.path.join(DEBUG_DIR, 'synthetic_dataset_sample.csv'), index=False)
    
    return X_synthetic, y_synthetic

def finetune_local_model(model, model_name, X_data, y_data):
    """Finetune a local model on a small dataset before aggregation."""
    logger.info(f"Finetuning {model_name} model on {len(X_data)} samples...")
    
    try:
        if model_name == 'random_forest':
            if hasattr(model, 'warm_start'):
                original_n_estimators = model.n_estimators
                model.warm_start = True
                
                for epoch in range(FINETUNE_EPOCHS):
                    model.n_estimators += 10
                    model.fit(X_data, y_data)
                    logger.info(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS}: {model.n_estimators} trees")
                
                logger.info(f"Finetuned {model_name} from {original_n_estimators} to {model.n_estimators} trees")
            else:
                model.fit(X_data, y_data)
                
        elif model_name == 'xgboost':
            if hasattr(model, 'get_booster') and hasattr(model, 'fit'):
                try:
                    import xgboost as xgb
                    dtrain = xgb.DMatrix(X_data, label=y_data)
                    for epoch in range(FINETUNE_EPOCHS):
                        model.get_booster().update(dtrain, epoch)
                        logger.info(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS} completed")
                except ImportError:
                    model.fit(X_data, y_data)
            else:
                model.fit(X_data, y_data)
                
        elif model_name == 'catboost':
            if hasattr(model, 'fit'):
                for epoch in range(FINETUNE_EPOCHS):
                    model.fit(X_data, y_data, verbose=False)
                    logger.info(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS} completed")
        else:
            if hasattr(model, 'fit'):
                model.fit(X_data, y_data)
        
        logger.info(f"Successfully finetuned {model_name} model")
        return model
    
    except Exception as e:
        logger.error(f"Error finetuning {model_name} model: {e}")
        logger.info(f"Using original {model_name} model without finetuning")
        return model

def finetune_global_model(global_model, model_type, X_data, y_data):
    """Finetune the global model on real data after knowledge distillation."""
    logger.info(f"Finetuning global {model_type} model on {len(X_data)} samples...")
    
    try:
        class_weights = compute_sample_weight(class_weight='balanced', y=y_data)
        
        if model_type == 'random_forest' and hasattr(global_model, 'warm_start'):
            original_n_estimators = global_model.n_estimators
            global_model.warm_start = True
            
            for epoch in range(FINETUNE_EPOCHS):
                global_model.n_estimators += 20
                global_model.fit(X_data, y_data, sample_weight=class_weights)
                logger.info(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS}: {global_model.n_estimators} trees")
            
            logger.info(f"Finetuned global model from {original_n_estimators} to {global_model.n_estimators} trees")
        else:
            for epoch in range(FINETUNE_EPOCHS):
                global_model.fit(X_data, y_data, sample_weight=class_weights)
                logger.info(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS} completed")
        
        logger.info(f"Successfully finetuned global {model_type} model")
        return global_model
    
    except Exception as e:
        logger.error(f"Error finetuning global model: {e}")
        logger.info(f"Using global model without finetuning")
        return global_model

def create_finetuning_dataset(dataset_path, size=FINETUNE_DATA_SIZE):
    """Create a small dataset for finetuning models from real data."""
    logger.info(f"Creating finetuning dataset of {size} samples from real data")
    
    try:
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found at {dataset_path}")
            if USE_SYNTHETIC_DATA and DATASET_PREFERENCE != 'real':
                logger.info("Will use synthetic dataset for finetuning")
                return None, None
            else:
                logger.error("Synthetic data fallback disabled or dataset preference set to 'real'. Exiting...")
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
        skip_rows = np.random.randint(1, 100000)
        df = pd.read_csv(dataset_path, skiprows=skip_rows, nrows=size)
        
        if TARGET_COLUMN not in df.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in the dataset")
            return None, None
        
        for col in COLUMNS_TO_DROP:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        logger.info(f"Created finetuning dataset with {X.shape[0]} samples")
        logger.info(f"Class distribution in finetuning dataset:")
        for cls, count in y.value_counts().items():
            logger.info(f"  {cls}: {count:,} ({count/len(y)*100:.2f}%)")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error creating finetuning dataset: {e}")
        if USE_SYNTHETIC_DATA and DATASET_PREFERENCE != 'real':
            logger.info("Will use synthetic dataset for finetuning")
            return None, None
        else:
            logger.error("Synthetic data fallback disabled or dataset preference set to 'real'. Exiting...")
            raise

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("HETROFL Aggregation using Knowledge Distillation (Multi-Round)")
    logger.info("=" * 80)
    
    overall_start_time = time.time()
    
    try:
        # Step 1: Create necessary directories
        create_dir_if_not_exists(SAVE_DIR)
        create_dir_if_not_exists(PLOTS_DIR)
        create_dir_if_not_exists(DEBUG_DIR)
        
        # Step 2: Load all trained models and their artifacts
        logger.info("\n--- Loading Trained Models ---")
        models_data = {}
        
        for model_name, model_info in MODEL_DIRS.items():
            model, encoder, scaler = load_model_and_artifacts(model_info)
            if model is not None:
                models_data[model_name] = {
                    'model': model,
                    'encoder': encoder,
                    'scaler': scaler
                }
        
        if not models_data:
            logger.error("No models were loaded successfully. Exiting...")
            exit(1)
        
        # Step 3: Load public dataset or create synthetic dataset for Knowledge Distillation
        logger.info("\n--- Loading Public Dataset for Knowledge Distillation ---")
        X_public, y_public = None, None
        if DATASET_PREFERENCE != 'synthetic':
            X_public, y_public = load_public_dataset(DATASET_PATH, SAMPLE_SIZE)
        
        if X_public is None and USE_SYNTHETIC_DATA:
            logger.info("Using synthetic dataset for knowledge distillation")
            X_public, y_public = create_synthetic_dataset(models_data)
        
        if X_public is None:
            logger.error("Failed to load or create dataset. Exiting...")
            exit(1)
        
        # Step 3.5: Load finetuning dataset if needed (used potentially for final global model finetuning or evaluation)
        if FINETUNE_LOCAL_MODELS or FINETUNE_GLOBAL_MODEL:
            logger.info("\n--- Loading Finetuning Dataset ---")
            X_finetune, y_finetune = create_finetuning_dataset(DATASET_PATH)
            
            if X_finetune is None and DATASET_PREFERENCE != 'synthetic':
                logger.warning("Failed to load finetuning dataset. Disabling finetuning.")
                FINETUNE_LOCAL_MODELS = False
                FINETUNE_GLOBAL_MODEL = False
            elif X_finetune is None:
                logger.info("Using synthetic dataset for finetuning")
                X_finetune, y_finetune = create_synthetic_dataset(models_data, FINETUNE_DATA_SIZE)
            
            if X_finetune is not None:
                X_finetune, y_finetune = preprocess_public_dataset(X_finetune, y_finetune)
        
        save_debug_info(models_data, X_public)
        X_public, y_public = preprocess_public_dataset(X_public, y_public)
        
        # Initialize global model before rounds
        logger.info("\n--- Initializing Global Model ---")
        # For heterogeneous FL with KD, the global model often starts simple or is trained once initially
        # Here, we'll initialize it and train it in the first 'round' conceptually.
        global_model = None # Global model will be trained/updated in rounds

        metrics, confusion_mat = None, None # Initialize metrics for final evaluation

        # Step 4: Iterate through aggregation rounds
        for round_num in range(NUM_ROUNDS):
            logger.info(f"\n--- Starting Aggregation Round {round_num + 1}/{NUM_ROUNDS} ---")
            round_start_time = time.time()

            # Get predictions from all local models on the public dataset
            logger.info("--- Getting Predictions from All Local Models ---")
            predictions = get_model_predictions(models_data, X_public)

            if not predictions:
                logger.error(f"No valid predictions obtained in round {round_num + 1}. Skipping round.")
                continue # Skip to the next round if no predictions

            # Aggregate predictions using weighted averaging
            logger.info("--- Aggregating Predictions (Soft Labels) ---")
            aggregation_result = aggregate_predictions(predictions)
            
            if aggregation_result[0] is None:
                logger.error(f"Failed to aggregate predictions in round {round_num + 1}. Skipping round.")
                continue # Skip to the next round
            
            soft_labels, hard_labels, classes = aggregation_result

            # Train or Update Global Model using aggregated soft labels
            logger.info("--- Updating Global Model ---")
            if global_model is None: # First round: train the initial global model
                global_model = train_global_model(
                    X_public,
                    soft_labels,
                    GLOBAL_MODEL_TYPE,
                    GLOBAL_MODEL_PARAMS[GLOBAL_MODEL_TYPE]
                )
                if global_model is not None:
                     logger.info("Initial global model trained.")
                else:
                    logger.error("Failed to train initial global model. Exiting.")
                    exit(1)
            else: # Subsequent rounds: finetune or update the existing global model
                 # Here, we finetune the global model on the public dataset using the aggregated hard labels from this round
                 # A more sophisticated approach might involve different update strategies
                logger.info(f"Finetuning global model in round {round_num + 1}")
                try:
                    # Using aggregated hard labels for finetuning
                    class_weights = compute_sample_weight(class_weight='balanced', y=hard_labels)
                    # Perform a limited fit/finetune step
                    global_model.fit(X_public, hard_labels, sample_weight=class_weights)
                    logger.info(f"Global model finetuned successfully in round {round_num + 1}")
                except Exception as e:
                    logger.error(f"Error finetuning global model in round {round_num + 1}: {e}")
                    logger.warning("Proceeding with previous global model version.")

            round_end_time = time.time()
            logger.info(f"--- Round {round_num + 1} completed in {round_end_time - round_start_time:.2f} seconds ---")

        # After all rounds, evaluate the final global model
        logger.info("\n--- Evaluating Final Global Model ---")
        # Using the public dataset with aggregated hard labels for evaluation after all rounds
        if global_model is not None and 'classes' in locals() and soft_labels is not None:
            y_eval_indices = np.argmax(soft_labels, axis=1) # Use the hard labels from the last aggregation
            metrics, confusion_mat = evaluate_global_model(
                global_model,
                X_public, # Evaluate on the public dataset
                y_eval_indices, # Using aggregated labels as ground truth for evaluation
                classes
            )
        else:
            logger.error("Global model not available for final evaluation.")
        
        # Step 6.5: Optional: Finetune global model on the real finetuning dataset if enabled
        # This step is separate from the aggregation rounds and uses the real data if loaded
        if FINETUNE_GLOBAL_MODEL and 'X_finetune' in locals() and X_finetune is not None and global_model is not None:
            logger.info("\n--- Final Finetuning Global Model on Real Data ---")
            # Ensure feature alignment and handle label encoding for real data finetuning
            if 'classes' in locals() and len(classes) > 0 and isinstance(classes[0], str):
                 # Assuming the global model was trained on indexed labels, encode real labels
                 # Need an encoder that maps real labels to indices. Let's reuse the encoder from one of the local models.
                 # Assuming all local model encoders map to the same set of indices.
                 if models_data:
                     first_model_encoder = list(models_data.values())[0]['encoder']
                     try:
                         y_finetune_encoded = first_model_encoder.transform(y_finetune)
                     except Exception as e:
                         logger.error(f"Error encoding finetuning labels for final finetuning: {e}. Skipping final finetuning.")
                         y_finetune_encoded = None
                 else:
                     logger.error("No local models loaded to get encoder for final finetuning. Skipping.")
                     y_finetune_encoded = None
            else:
                 # Assuming labels are already numerical indices
                 y_finetune_encoded = y_finetune.values

            if y_finetune_encoded is not None:
                # Align features for finetuning data with global model expected features
                if hasattr(global_model, 'feature_names_in_'):
                    X_finetune_aligned = align_features(global_model, X_finetune, "Global Model (Finetune)")
                else:
                     logger.warning("Global model does not have feature_names_in_. Using finetuning data as is.")
                     X_finetune_aligned = X_finetune

                if X_finetune_aligned is not None:
                     global_model = finetune_global_model(
                         global_model,
                         GLOBAL_MODEL_TYPE,
                         X_finetune_aligned,
                         y_finetune_encoded
                     )
                else:
                     logger.error("Failed to align features for final finetuning. Skipping.")
            
        # Step 7: Plot and save results (using the metrics from the final evaluation)
        if metrics is not None and confusion_mat is not None and 'classes' in locals():
            logger.info("\n--- Generating and Saving Plots ---")
            plot_and_save_results(metrics, confusion_mat, classes)
        else:
            logger.warning("Metrics or Confusion Matrix not available for plotting.")
        
        # Step 8: Save global model
        if global_model is not None and models_data:
            logger.info("\n--- Saving Global Model ---")
            first_model = list(models_data.keys())[0] # Assuming all local models use compatible encoders
            encoder = models_data[first_model]['encoder']
            save_global_model(global_model, encoder) # Save the final global model
        else:
            logger.warning("Global model or local model data not available for saving.")
        
        # Cleanup
        gc.collect()
        
        # Completion message
        total_time = time.time() - overall_start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"HETROFL Aggregation (Multi-Round) completed in {minutes} minutes and {seconds} seconds")
        if global_model is not None:
             logger.info(f"Final Global model saved to: {os.path.join(SAVE_DIR, GLOBAL_MODEL_FILENAME)}")
        if metrics is not None:
            logger.info("Final Performance Summary:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during HETROFL aggregation: {e}")
        import traceback
        traceback.print_exc()