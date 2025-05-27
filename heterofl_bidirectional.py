"""
Bidirectional Heterogeneous Federated Learning System
Implements enhanced federated learning with feedback loops where the global model
sends fine-tuning signals back to local models after each aggregation round.
"""

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
from typing import Dict, Any, Tuple, List, Optional

# Import our custom utilities
from utils.model_manager import ModelStateManager, IncrementalLearningManager
from utils.metrics_tracker import MetricsTracker
from utils.websocket_handler import websocket_handler, fl_event_emitter

# Import functions from the original aggregation script
from heterofl_aggregation import (
    create_dir_if_not_exists, load_model_and_artifacts, load_public_dataset,
    preprocess_public_dataset, align_features, get_model_predictions,
    aggregate_predictions, balance_predictions, train_global_model,
    evaluate_global_model, plot_and_save_results, save_global_model,
    create_synthetic_dataset, create_finetuning_dataset
)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, 'heterofl_models')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'heterofl_plots')
DEBUG_DIR = os.path.join(SCRIPT_DIR, 'heterofl_debug')

# Bidirectional FL specific configuration
BIDIRECTIONAL_CONFIG = {
    'num_rounds': 5,
    'pseudo_label_confidence_threshold': 0.7,
    'local_update_learning_rate': 0.1,
    'local_update_epochs': 3,
    'feedback_sample_size': 10000,
    'knowledge_transfer_weight': 0.3,
    'enable_model_checkpoints': True,
    'checkpoint_frequency': 2,
    'enable_metrics_tracking': True
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'heterofl_bidirectional.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BidirectionalHeteroFL:
    """
    Enhanced Heterogeneous Federated Learning system with bidirectional communication.
    Implements feedback loops where global model knowledge is sent back to local models.
    """
    
    def __init__(self, config: Dict[str, Any], model_dirs: Dict[str, Dict], 
                 global_model_config: Dict[str, Any]):
        """
        Initialize the bidirectional federated learning system.
        
        Args:
            config: Configuration dictionary for bidirectional FL
            model_dirs: Dictionary containing local model directories and info
            global_model_config: Configuration for global model
        """
        self.config = config
        self.model_dirs = model_dirs
        self.global_model_config = global_model_config
        
        # Initialize managers
        self.state_manager = ModelStateManager(SCRIPT_DIR)
        self.metrics_tracker = MetricsTracker(SCRIPT_DIR)
        self.incremental_manager = IncrementalLearningManager()
        
        # Initialize state
        self.local_models = {}
        self.local_artifacts = {}
        self.global_model = None
        self.global_encoder = None
        self.global_scaler = None
        self.current_round = 0
        
        # Data storage
        self.public_dataset = None
        self.public_labels = None
        self.feedback_dataset = None
        self.feedback_labels = None
        
        logger.info("Initialized BidirectionalHeteroFL system")
        
        # Initialize event emitter for real-time updates
        self.event_emitter = fl_event_emitter
    
    def load_local_models(self) -> bool:
        """Load all local models and their artifacts."""
        logger.info("Loading local models and artifacts...")
        
        for model_name, model_info in self.model_dirs.items():
            model, encoder, scaler = load_model_and_artifacts(model_info)
            if model is not None:
                self.local_models[model_name] = model
                self.local_artifacts[model_name] = {
                    'encoder': encoder,
                    'scaler': scaler,
                    'weight': model_info.get('weight', 1.0)
                }
                logger.info(f"Successfully loaded {model_name}")
            else:
                logger.warning(f"Failed to load {model_name}")
        
        if not self.local_models:
            logger.error("No local models were loaded successfully")
            return False
        
        logger.info(f"Loaded {len(self.local_models)} local models: {list(self.local_models.keys())}")
        
        # Emit models loaded event
        self.event_emitter.on_models_loaded(list(self.local_models.keys()))
        
        return True
    
    def prepare_datasets(self, dataset_path: str, sample_size: int, 
                        use_synthetic: bool = True) -> bool:
        """Prepare public dataset and feedback dataset for training."""
        logger.info("Preparing datasets for bidirectional FL...")
        
        # Load public dataset for knowledge distillation
        self.public_dataset, self.public_labels = load_public_dataset(dataset_path, sample_size)
        
        if self.public_dataset is None and use_synthetic:
            logger.info("Creating synthetic dataset for knowledge distillation")
            models_data = {name: {'model': model, 'encoder': self.local_artifacts[name]['encoder']} 
                          for name, model in self.local_models.items()}
            self.public_dataset, self.public_labels = create_synthetic_dataset(models_data, sample_size)
        
        if self.public_dataset is None:
            logger.error("Failed to prepare public dataset")
            return False
        
        # Preprocess public dataset
        self.public_dataset, self.public_labels = preprocess_public_dataset(
            self.public_dataset, self.public_labels)
        
        # Prepare feedback dataset (smaller subset for local model updates)
        feedback_size = min(self.config.get('feedback_sample_size', 10000), len(self.public_dataset))
        feedback_indices = np.random.choice(len(self.public_dataset), feedback_size, replace=False)
        self.feedback_dataset = self.public_dataset.iloc[feedback_indices].copy()
        self.feedback_labels = self.public_labels.iloc[feedback_indices].copy()
        
        logger.info(f"Prepared datasets - Public: {len(self.public_dataset)}, Feedback: {len(self.feedback_dataset)}")
        return True
    
    def get_local_predictions(self, round_num: int) -> Dict[str, Dict]:
        """Get predictions from all local models and record metrics."""
        logger.info(f"Getting predictions from local models for round {round_num}")
        
        models_data = {}
        for model_name, model in self.local_models.items():
            artifacts = self.local_artifacts[model_name]
            models_data[model_name] = {
                'model': model,
                'encoder': artifacts['encoder'],
                'scaler': artifacts['scaler']
            }
        
        # Get predictions using the existing function
        predictions = get_model_predictions(models_data, self.public_dataset)
        
        # Record metrics for each local model
        for model_name, pred_data in predictions.items():
            if pred_data is not None:
                # Calculate basic metrics
                probabilities = pred_data['probabilities']
                hard_preds = np.argmax(probabilities, axis=1)
                
                # Calculate confidence and entropy
                max_probs = np.max(probabilities, axis=1)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
                
                metrics = {
                    'mean_confidence': float(np.mean(max_probs)),
                    'mean_entropy': float(np.mean(entropy)),
                    'prediction_diversity': float(np.std(hard_preds))
                }
                
                predictions_stats = {
                    'class_distribution': {int(k): int(v) for k, v in 
                                         zip(*np.unique(hard_preds, return_counts=True))},
                    'confidence_stats': {
                        'min': float(np.min(max_probs)),
                        'max': float(np.max(max_probs)),
                        'std': float(np.std(max_probs))
                    }
                }
                
                self.metrics_tracker.record_local_model_metrics(
                    round_num, model_name, metrics, predictions_stats)
        
        return predictions
    
    def aggregate_and_update_global(self, predictions: Dict[str, Dict], round_num: int) -> bool:
        """Aggregate predictions and update global model."""
        logger.info(f"Aggregating predictions and updating global model for round {round_num}")
        
        # Aggregate predictions
        aggregation_result = aggregate_predictions(predictions)
        if aggregation_result[0] is None:
            logger.error("Failed to aggregate predictions")
            return False
        
        soft_labels, hard_labels, classes = aggregation_result
        
        # Record aggregation metrics
        unique_classes, class_counts = np.unique(hard_labels, return_counts=True)
        aggregation_stats = {
            'num_models_aggregated': len(predictions),
            'total_samples': len(hard_labels),
            'num_classes': len(unique_classes),
            'class_balance_entropy': float(-np.sum((class_counts/len(hard_labels)) * 
                                                  np.log(class_counts/len(hard_labels) + 1e-8))),
            'aggregation_confidence': float(np.mean(np.max(soft_labels, axis=1)))
        }
        
        self.metrics_tracker.record_aggregation_metrics(round_num, aggregation_stats)
        
        # Train or update global model
        if self.global_model is None:
            # First round: train initial global model
            logger.info("Training initial global model")
            self.global_model = train_global_model(
                self.public_dataset,
                soft_labels,
                self.global_model_config['type'],
                self.global_model_config['params']
            )
            
            if self.global_model is None:
                logger.error("Failed to train initial global model")
                return False
            
            # Set global encoder (use first local model's encoder)
            first_model_name = list(self.local_artifacts.keys())[0]
            self.global_encoder = self.local_artifacts[first_model_name]['encoder']
            
        else:
            # Subsequent rounds: update existing global model
            logger.info(f"Updating global model for round {round_num}")
            try:
                # Use balanced soft labels for training
                balanced_soft_labels = balance_predictions(soft_labels)
                hard_labels_balanced = np.argmax(balanced_soft_labels, axis=1)
                
                # Compute sample weights
                sample_weights = compute_sample_weight(class_weight='balanced', y=hard_labels_balanced)
                
                # Update global model
                self.global_model.fit(self.public_dataset, hard_labels_balanced, sample_weight=sample_weights)
                logger.info("Global model updated successfully")
                
            except Exception as e:
                logger.error(f"Error updating global model: {e}")
                return False
        
        # Evaluate global model and record metrics
        try:
            y_pred = self.global_model.predict(self.public_dataset)
            y_prob = self.global_model.predict_proba(self.public_dataset)
            
            global_metrics = {
                'accuracy': float(accuracy_score(hard_labels, y_pred)),
                'precision': float(precision_score(hard_labels, y_pred, average='weighted')),
                'recall': float(recall_score(hard_labels, y_pred, average='weighted')),
                'f1': float(f1_score(hard_labels, y_pred, average='weighted'))
            }
            
            # Add AUC if binary classification
            if len(classes) == 2:
                global_metrics['auc'] = float(roc_auc_score(hard_labels, y_prob[:, 1]))
            
            self.metrics_tracker.record_global_model_metrics(round_num, global_metrics)
            logger.info(f"Global model metrics: {global_metrics}")
            
        except Exception as e:
            logger.warning(f"Error evaluating global model: {e}")
        
        return True
    
    def generate_pseudo_labels(self, round_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pseudo-labels from global model for local model updates."""
        logger.info(f"Generating pseudo-labels for round {round_num}")
        
        if self.global_model is None:
            logger.error("Global model not available for pseudo-label generation")
            return None, None
        
        try:
            # Generate predictions on feedback dataset
            y_pseudo_prob = self.global_model.predict_proba(self.feedback_dataset)
            y_pseudo_hard = self.global_model.predict(self.feedback_dataset)
            
            # Filter by confidence threshold
            confidence_threshold = self.config.get('pseudo_label_confidence_threshold', 0.7)
            max_probs = np.max(y_pseudo_prob, axis=1)
            confident_indices = max_probs >= confidence_threshold
            
            if np.sum(confident_indices) == 0:
                logger.warning("No confident pseudo-labels generated, using all predictions")
                confident_indices = np.ones(len(y_pseudo_hard), dtype=bool)
            
            pseudo_soft = y_pseudo_prob[confident_indices]
            pseudo_hard = y_pseudo_hard[confident_indices]
            
            logger.info(f"Generated {len(pseudo_hard)} confident pseudo-labels "
                       f"(threshold: {confidence_threshold}, confidence rate: {np.mean(confident_indices):.3f})")
            
            return pseudo_soft, pseudo_hard
            
        except Exception as e:
            logger.error(f"Error generating pseudo-labels: {e}")
            return None, None
    
    def update_local_models(self, pseudo_soft: np.ndarray, pseudo_hard: np.ndarray, round_num: int) -> bool:
        """Update local models using pseudo-labels from global model."""
        logger.info(f"Updating local models with global knowledge for round {round_num}")
        
        if pseudo_soft is None or pseudo_hard is None:
            logger.warning("No pseudo-labels available for local model updates")
            return False
        
        # Prepare feedback data for local updates
        feedback_indices = np.random.choice(len(self.feedback_dataset), 
                                          min(len(pseudo_hard), len(self.feedback_dataset)), 
                                          replace=False)
        X_feedback = self.feedback_dataset.iloc[feedback_indices]
        
        knowledge_transfer_weight = self.config.get('knowledge_transfer_weight', 0.3)
        local_update_epochs = self.config.get('local_update_epochs', 3)
        
        updated_models = {}
        
        for model_name, model in self.local_models.items():
            logger.info(f"Updating {model_name} with global knowledge")
            
            try:
                # Align features for this model
                scaler = self.local_artifacts[model_name]['scaler']
                X_scaled = pd.DataFrame(scaler.transform(X_feedback), columns=X_feedback.columns)
                X_aligned = align_features(model, X_scaled, model_name)
                
                if X_aligned is None:
                    logger.warning(f"Failed to align features for {model_name}, skipping update")
                    continue
                
                # Create mixed labels (combine original confidence with pseudo-labels)
                # Use knowledge transfer weight to balance original vs global knowledge
                y_mixed = pseudo_hard[:len(X_aligned)]
                
                # Update model using incremental learning
                model_type = self._get_model_type(model_name)
                updated_model = self.incremental_manager.update_model_by_type(
                    model_type, model, X_aligned.values, y_mixed,
                    learning_rate=self.config.get('local_update_learning_rate', 0.1),
                    n_rounds=local_update_epochs
                )
                
                updated_models[model_name] = updated_model
                
                # Save model state
                if self.config.get('enable_model_checkpoints', True):
                    metrics = {'knowledge_transfer_samples': len(y_mixed)}
                    self.state_manager.save_model_state(
                        model_name, updated_model, round_num, metrics)
                
                logger.info(f"Successfully updated {model_name}")
                
            except Exception as e:
                logger.error(f"Error updating {model_name}: {e}")
                continue
        
        # Update local models with updated versions
        self.local_models.update(updated_models)
        
        logger.info(f"Updated {len(updated_models)} local models with global knowledge")
        return len(updated_models) > 0
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name."""
        if 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
            return 'xgboost'
        elif 'catboost' in model_name.lower() or 'cat' in model_name.lower():
            return 'catboost'
        elif 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
            return 'random_forest'
        else:
            return 'unknown'
    
    def run_federated_round(self, round_num: int) -> bool:
        """Execute a complete bidirectional federated learning round."""
        logger.info(f"Starting bidirectional FL round {round_num}")
        round_start_time = time.time()
        
        # Record round start
        self.metrics_tracker.record_round_start(round_num)
        
        try:
            # Step 1: Get local model predictions
            predictions = self.get_local_predictions(round_num)
            if not predictions:
                logger.error(f"No predictions obtained in round {round_num}")
                return False
            
            # Step 2: Aggregate predictions and update global model
            if not self.aggregate_and_update_global(predictions, round_num):
                logger.error(f"Failed to aggregate and update global model in round {round_num}")
                return False
            
            # Step 3: Generate pseudo-labels from global model
            pseudo_soft, pseudo_hard = self.generate_pseudo_labels(round_num)
            
            # Step 4: Update local models with global knowledge
            if pseudo_soft is not None and pseudo_hard is not None:
                self.update_local_models(pseudo_soft, pseudo_hard, round_num)
            else:
                logger.warning(f"Skipping local model updates in round {round_num} due to pseudo-label generation failure")
            
            # Step 5: Create checkpoint if needed
            if (self.config.get('enable_model_checkpoints', True) and 
                round_num % self.config.get('checkpoint_frequency', 2) == 0):
                checkpoint_id = self.state_manager.create_checkpoint(
                    round_num, self.global_model, self.local_models, 
                    self.metrics_tracker.get_latest_metrics())
                logger.info(f"Created checkpoint: {checkpoint_id}")
            
            # Record round completion
            round_time = time.time() - round_start_time
            self.metrics_tracker.record_round_end(round_num, round_time)
            
            logger.info(f"Completed bidirectional FL round {round_num} in {round_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error in bidirectional FL round {round_num}: {e}")
            return False
    
    def run_federated_learning(self, num_rounds: int = None) -> bool:
        """Run the complete bidirectional federated learning process."""
        if num_rounds is None:
            num_rounds = self.config.get('num_rounds', 5)
        
        logger.info(f"Starting bidirectional federated learning for {num_rounds} rounds")
        overall_start_time = time.time()
        
        # Emit FL start event
        self.event_emitter.on_fl_start({'num_rounds': num_rounds})
        
        try:
            # Initialize
            if not self.load_local_models():
                logger.error("Failed to load local models")
                return False
            
            # Prepare datasets (using config from original system)
            from heterofl_aggregation import DATASET_PATH, SAMPLE_SIZE
            if not self.prepare_datasets(DATASET_PATH, SAMPLE_SIZE):
                logger.error("Failed to prepare datasets")
                return False
            
            # Run federated rounds
            successful_rounds = 0
            for round_num in range(1, num_rounds + 1):
                # Emit round start event
                self.event_emitter.on_round_start(round_num, num_rounds)
                
                if self.run_federated_round(round_num):
                    successful_rounds += 1
                    self.event_emitter.on_round_complete(round_num, True)
                else:
                    logger.warning(f"Round {round_num} failed, continuing with next round")
                    self.event_emitter.on_round_complete(round_num, False)
            
            # Final evaluation and reporting
            total_time = time.time() - overall_start_time
            logger.info(f"Completed bidirectional FL: {successful_rounds}/{num_rounds} successful rounds")
            logger.info(f"Total time: {total_time:.2f} seconds")
            
            # Generate final plots and exports
            self._generate_final_reports()
            
            # Save final models
            self._save_final_models()
            
            # Emit FL completion event
            final_metrics = self.metrics_tracker.get_latest_metrics()
            self.event_emitter.on_fl_complete(num_rounds, final_metrics)
            
            return successful_rounds > 0
            
        except Exception as e:
            logger.error(f"Error in bidirectional federated learning: {e}")
            return False
    
    def _generate_final_reports(self):
        """Generate final plots and export metrics."""
        logger.info("Generating final reports and plots")
        
        try:
            # Plot model evolution
            self.metrics_tracker.plot_model_evolution('accuracy')
            self.metrics_tracker.plot_model_evolution('f1')
            
            # Plot global model evolution
            self.metrics_tracker.plot_global_model_evolution()
            
            # Export metrics to CSV
            self.metrics_tracker.export_metrics_csv()
            
            # Save metrics state
            self.metrics_tracker.save_state()
            
            logger.info("Final reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating final reports: {e}")
    
    def _save_final_models(self):
        """Save final global and local models."""
        logger.info("Saving final models")
        
        try:
            # Save global model
            if self.global_model is not None:
                save_global_model(self.global_model, self.global_encoder)
            
            # Save final local model states
            final_round = self.current_round
            for model_name, model in self.local_models.items():
                metrics = {'final_model': True}
                self.state_manager.save_model_state(
                    model_name, model, final_round, metrics)
            
            logger.info("Final models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving final models: {e}")


def main():
    """Main execution function for bidirectional federated learning."""
    logger.info("=" * 80)
    logger.info("BIDIRECTIONAL HETROFL - Enhanced Federated Learning with Feedback Loops")
    logger.info("=" * 80)
    
    # Import configuration from original system
    from heterofl_aggregation import (
        MODEL_DIRS, GLOBAL_MODEL_TYPE, GLOBAL_MODEL_PARAMS, NUM_ROUNDS
    )
    
    # Create enhanced configuration
    global_model_config = {
        'type': GLOBAL_MODEL_TYPE,
        'params': GLOBAL_MODEL_PARAMS[GLOBAL_MODEL_TYPE]
    }
    
    # Update bidirectional config with original settings
    BIDIRECTIONAL_CONFIG['num_rounds'] = NUM_ROUNDS
    
    # Initialize and run bidirectional FL system
    bidirectional_fl = BidirectionalHeteroFL(
        config=BIDIRECTIONAL_CONFIG,
        model_dirs=MODEL_DIRS,
        global_model_config=global_model_config
    )
    
    # Run the federated learning process
    success = bidirectional_fl.run_federated_learning()
    
    if success:
        logger.info("Bidirectional federated learning completed successfully!")
    else:
        logger.error("Bidirectional federated learning failed!")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
