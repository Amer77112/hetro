import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle
import psutil
import gc
import logging
from uuid import uuid4
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import catboost as cb
import lightgbm as lgb
import dask.dataframe as dd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILENAME = 'catboost_dataset.csv'
DATASET_PATH = os.path.join(SCRIPT_DIR, DATASET_FILENAME)
DATASET_PATH_ABSOLUTE = r'C:\Users\VICTUS\Desktop\XX_XX\catboost\catboost_dataset.csv'

TARGET_COLUMN = 'Attack'
COLUMNS_TO_DROP = ['Label']
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_DIR = SCRIPT_DIR
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
MODEL_FILENAME = 'ensemble_model.pkl'
ENCODER_FILENAME = 'label_encoder.pkl'
SCALER_FILENAME = 'scaler.pkl'

# --- Enable or disable features ---
FAST_MODE = True
SAMPLE_DATA = True
SAMPLE_SIZE = 3000000

# Reduced further to handle low memory
SKIP_CV = True
MINIMAL_PLOTS = True
MIN_ROUNDS = 50

# --- Model Parameters ---
def check_catboost_gpu():
    """Check if CatBoost can use GPU by attempting to initialize a GPU model."""
    try:
        cb.CatBoostClassifier(task_type='GPU', iterations=1).fit([[0]], [0])
        return 'GPU'
    except Exception:
        return 'CPU'

CAT_PARAMS = {
    'loss_function': 'MultiClass',
    'iterations': MIN_ROUNDS,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'random_seed': RANDOM_STATE,
    'bagging_temperature': 1,
    'rsm': 0.8,
    'thread_count': -1,
    'verbose': 100,
    'bootstrap_type': 'Bayesian',  # Changed to Bayesian to support bagging_temperature
    'od_type': 'Iter',
    'od_wait': 10,
    'task_type': check_catboost_gpu()
}

LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_iterations': MIN_ROUNDS,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 15,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu' if lgb.__version__ >= '3.0.0' else 'cpu'
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SAVE_DIR, 'training_log.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def check_system_resources():
    """Check available memory and CPU resources."""
    mem = psutil.virtual_memory()
    available_mem_mb = mem.available / (1024 * 1024)
    cpu_count = psutil.cpu_count()
    logger.info(f"Available memory: {available_mem_mb:.2f} MB")
    logger.info(f"Available CPU cores: {cpu_count}")
    if available_mem_mb < 2000:
        logger.warning("Low memory available. Consider reducing SAMPLE_SIZE or closing other applications.")
    return available_mem_mb, cpu_count

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def optimize_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory usage."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def load_data():
    logger.info("\n--- Loading Dataset ---")
    filepath = DATASET_PATH if os.path.exists(DATASET_PATH) else DATASET_PATH_ABSOLUTE
    if not os.path.exists(filepath):
        logger.error(f"Dataset file not found at {filepath}")
        return None

    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"Dataset file size: {file_size_mb:.2f} MB")
        
        # Check system resources before loading
        available_mem_mb, _ = check_system_resources()
        if file_size_mb > available_mem_mb * 0.5:
            logger.warning("Dataset size exceeds 50% of available memory. Using Dask for loading.")
        
        # Use Dask for large datasets
        if file_size_mb > 500:
            ddf = dd.read_csv(filepath, blocksize='64MB')
            if SAMPLE_DATA:
                ddf = ddf.sample(frac=SAMPLE_SIZE / ddf.shape[0].compute(), random_state=RANDOM_STATE)
            df = ddf.compute()
        else:
            df = pd.read_csv(filepath)
            if SAMPLE_DATA and len(df) > SAMPLE_SIZE:
                df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        
        df = optimize_dtypes(df)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        logger.info(f"\nData Types:\n{df.dtypes}")
        logger.info(f"\nMissing values: {df.isnull().sum().sum():,}")
        if TARGET_COLUMN in df.columns:
            logger.info("\nClass Distribution:")
            for cls, count in df[TARGET_COLUMN].value_counts().items():
                logger.info(f"  {cls}: {count:,} ({count/len(df)*100:.2f}%)")
        logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")
        gc.collect()
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    logger.info("\n--- Preprocessing Data ---")
    start_time = time.time()
    
    # Check system resources
    check_system_resources()
    
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped column: {col}")
    
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found!")
        return None, None, None, None, None, None

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Imputation
    try:
        logger.info("Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        logger.info("Imputation completed")
    except Exception as e:
        logger.error(f"Error during imputation: {e}")
        return None, None, None, None, None, None
    
    # Scaling
    try:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        logger.info("Scaling completed")
    except Exception as e:
        logger.error(f"Error during scaling: {e}")
        return None, None, None, None, None, None
    
    # Encode target
    try:
        logger.info("Encoding target...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        logger.info(f"Encoded {len(le.classes_)} target classes: {list(le.classes_)}")
    except Exception as e:
        logger.error(f"Error during target encoding: {e}")
        return None, None, None, None, None, None
    
    # Use all features instead of feature selection
    logger.info(f"Using all {X.shape[1]} features for training")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    logger.info(f"Training set: {X_train.shape[0]:,} samples")
    logger.info(f"Test set: {X_test.shape[0]:,} samples")
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Memory usage after preprocessing: {get_memory_usage():.2f} MB")
    gc.collect()
    return X_train, X_test, y_train, y_test, le, scaler

def train_ensemble(X_train, y_train, num_classes):
    logger.info("\n--- Training Ensemble Model (CatBoost + LightGBM) ---")
    start_time = time.time()
    
    # Adjust parameters for binary/multiclass
    if num_classes == 2:
        CAT_PARAMS['loss_function'] = 'Logloss'
        LGBM_PARAMS['objective'] = 'binary'
    else:
        CAT_PARAMS['loss_function'] = 'MultiClass'
        LGBM_PARAMS['objective'] = 'multiclass'
        LGBM_PARAMS['num_class'] = num_classes
    
    # Initialize models
    logger.info(f"CatBoost parameters for training: {CAT_PARAMS}")
    cat_model = cb.CatBoostClassifier(**CAT_PARAMS)
    lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    # Create ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('catboost', cat_model),
            ('lightgbm', lgb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Train ensemble with progress
    logger.info("Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")
    gc.collect()
    return ensemble_model

def evaluate_model(model, X_test, y_test, le):
    logger.info("\n--- Evaluating Model Performance ---")
    start_time = time.time()
    
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    if len(le.classes_) == 2:
        metrics['auc'] = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc_scores = []
        for i in range(len(le.classes_)):
            auc_scores.append(roc_auc_score((y_test == i).astype(int), y_prob[:, i]))
        metrics['auc_ovr_mean'] = np.mean(auc_scores)
    
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE METRICS SUMMARY")
    logger.info("="*50)
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    logger.info("="*50)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Memory usage after evaluation: {get_memory_usage():.2f} MB")
    return y_pred, y_prob, cm, metrics

def plot_and_save_results(X_train, X_test, y_test, y_pred, y_prob, cm, metrics, model, le):
    logger.info("\n--- Generating and Saving Plots ---")
    plot_start_time = time.time()
    
    create_dir_if_not_exists(PLOTS_DIR)
    plots_created = []
    
    # Confusion Matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_file = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=150)
        plt.close()
        plots_created.append(cm_file)
        logger.info(f"Saved confusion matrix to: {cm_file}")
    except Exception as e:
        logger.error(f"Error creating confusion matrix plot: {e}")
    
    # Feature Importance
    try:
        cat_model = model.named_estimators_['catboost']
        importances = cat_model.get_feature_importance()
        indices = np.argsort(importances)[::-1]
        num_features = min(10, len(X_train.columns))
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances (CatBoost)')
        plt.bar(range(num_features), importances[indices[:num_features]])
        plt.xticks(range(num_features), [X_train.columns[i] for i in indices[:num_features]], rotation=90)
        plt.tight_layout()
        
        fi_file = os.path.join(PLOTS_DIR, 'feature_importance.png')
        plt.savefig(fi_file, dpi=150)
        plt.close()
        plots_created.append(fi_file)
        logger.info(f"Saved feature importance plot to: {fi_file}")
        
        # Save top features to text file
        top_features_file = os.path.join(SAVE_DIR, 'top_features.txt')
        with open(top_features_file, 'w') as f:
            f.write("Top features by importance:\n")
            for i in range(min(10, len(X_train.columns))):
                f.write(f"{i+1}. {X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}\n")
        logger.info(f"Saved top features list to: {top_features_file}")
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
    
    # Metrics Summary
    try:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        
        plt.figure(figsize=(10, 6))
        plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
        plt.ylim([0, 1.0])
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.title('Performance Metrics Summary')
        plt.tight_layout()
        
        metrics_file = os.path.join(PLOTS_DIR, 'metrics_summary.png')
        plt.savefig(metrics_file, dpi=150)
        plt.close()
        plots_created.append(metrics_file)
        logger.info(f"Saved metrics summary plot to: {metrics_file}")
    except Exception as e:
        logger.error(f"Error creating metrics summary plot: {e}")
    
    logger.info(f"Created {len(plots_created)} plots in {time.time() - plot_start_time:.2f} seconds")
    logger.info(f"Memory usage after plotting: {get_memory_usage():.2f} MB")
    gc.collect()

def save_model_and_encoder(model, encoder, scaler):
    logger.info("\n--- Saving Model and Encoder ---")
    
    model_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to: {model_path} (Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    encoder_path = os.path.join(SAVE_DIR, ENCODER_FILENAME)
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        logger.info("Label encoder saved successfully")
    except Exception as e:
        logger.error(f"Error saving label encoder: {e}")
    
    scaler_path = os.path.join(SAVE_DIR, SCALER_FILENAME)
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Scaler saved successfully")
    except Exception as e:
        logger.error(f"Error saving scaler: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Optimized Ensemble (CatBoost + LightGBM) Training Script")
    logger.info(f"Target dataset: {DATASET_FILENAME}")
    logger.info("=" * 80)
    
    overall_start_time = time.time()
    
    try:
        df = load_data()
        if df is not None:
            X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(df)
            
            if X_train is not None:
                num_classes = len(label_encoder.classes_)
                ensemble_model = train_ensemble(X_train, y_train, num_classes)
                y_pred, y_prob, confusion_mat, eval_metrics = evaluate_model(ensemble_model, X_test, y_test, label_encoder)
                plot_and_save_results(X_train, X_test, y_test, y_pred, y_prob, confusion_mat, eval_metrics, ensemble_model, label_encoder)
                save_model_and_encoder(ensemble_model, label_encoder, scaler)
                
                total_time = time.time() - overall_start_time
                minutes = int(total_time // 60)
                seconds = int(total_time % 60)
                
                logger.info("\n" + "=" * 80)
                logger.info(f"Script completed in {minutes} minutes and {seconds} seconds")
                logger.info(f"Model saved to: {os.path.join(SAVE_DIR, MODEL_FILENAME)}")
                logger.info("Performance Summary:")
                for metric, value in eval_metrics.items():
                    logger.info(f"  {metric.upper()}: {value:.4f}")
                logger.info("=" * 80)
            else:
                logger.error("\nScript aborted due to preprocessing errors.")
        else:
            logger.error("\nScript aborted due to data loading errors.")
    except Exception as e:
        logger.error(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"\nFinal memory usage: {get_memory_usage():.2f} MB")