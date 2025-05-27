"""
Enhanced Model State Management for Bidirectional HeteroFL
Handles model persistence, state tracking, and incremental learning capabilities.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class ModelStateManager:
    """Manages model states, persistence, and incremental learning for bidirectional FL."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.states_dir = os.path.join(base_dir, 'model_states')
        self.checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories for state management."""
        for directory in [self.states_dir, self.checkpoints_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def save_model_state(self, model_name: str, model: Any, round_num: int, 
                        metrics: Dict[str, float], additional_data: Dict = None) -> str:
        """Save complete model state including metrics and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_id = f"{model_name}_round_{round_num}_{timestamp}"
        
        state_data = {
            'model_name': model_name,
            'round_num': round_num,
            'timestamp': timestamp,
            'metrics': metrics,
            'additional_data': additional_data or {}
        }
        
        # Save model object
        model_path = os.path.join(self.states_dir, f"{state_id}_model.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            state_data['model_path'] = model_path
            logger.info(f"Saved model state for {model_name} round {round_num}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return None
        
        # Save metadata
        metadata_path = os.path.join(self.states_dir, f"{state_id}_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"Saved metadata for {model_name} round {round_num}")
        except Exception as e:
            logger.error(f"Error saving metadata for {model_name}: {e}")
        
        return state_id
    
    def load_model_state(self, state_id: str) -> Tuple[Any, Dict]:
        """Load model state and metadata by state ID."""
        model_path = os.path.join(self.states_dir, f"{state_id}_model.pkl")
        metadata_path = os.path.join(self.states_dir, f"{state_id}_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            logger.error(f"State files not found for {state_id}")
            return None, None
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded model state {state_id}")
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model state {state_id}: {e}")
            return None, None
    
    def get_latest_state(self, model_name: str) -> Tuple[Any, Dict]:
        """Get the most recent state for a specific model."""
        state_files = [f for f in os.listdir(self.states_dir) 
                      if f.startswith(f"{model_name}_round_") and f.endswith("_metadata.json")]
        
        if not state_files:
            logger.warning(f"No saved states found for {model_name}")
            return None, None
        
        # Sort by timestamp to get latest
        state_files.sort(reverse=True)
        latest_metadata_file = state_files[0]
        state_id = latest_metadata_file.replace("_metadata.json", "")
        
        return self.load_model_state(state_id)
    
    def create_checkpoint(self, round_num: int, global_model: Any, 
                         local_models: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Create a complete system checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_round_{round_num}_{timestamp}"
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_data = {
            'round_num': round_num,
            'timestamp': timestamp,
            'metrics': metrics,
            'local_models': list(local_models.keys())
        }
        
        try:
            # Save global model
            if global_model is not None:
                global_path = os.path.join(checkpoint_dir, "global_model.pkl")
                with open(global_path, 'wb') as f:
                    pickle.dump(global_model, f)
                checkpoint_data['global_model_path'] = global_path
            
            # Save local models
            for model_name, model in local_models.items():
                local_path = os.path.join(checkpoint_dir, f"{model_name}_model.pkl")
                with open(local_path, 'wb') as f:
                    pickle.dump(model, f)
                checkpoint_data[f'{model_name}_path'] = local_path
            
            # Save checkpoint metadata
            metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Created checkpoint {checkpoint_id} for round {round_num}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Any, Dict[str, Any], Dict]:
        """Load a complete system checkpoint."""
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return None, None, None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load global model
            global_model = None
            if 'global_model_path' in metadata:
                with open(metadata['global_model_path'], 'rb') as f:
                    global_model = pickle.load(f)
            
            # Load local models
            local_models = {}
            for model_name in metadata['local_models']:
                model_path = metadata[f'{model_name}_path']
                with open(model_path, 'rb') as f:
                    local_models[model_name] = pickle.load(f)
            
            logger.info(f"Loaded checkpoint {checkpoint_id}")
            return global_model, local_models, metadata
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None, None, None
    
    def get_model_evolution(self, model_name: str) -> pd.DataFrame:
        """Get the evolution of a model's performance across rounds."""
        state_files = [f for f in os.listdir(self.states_dir) 
                      if f.startswith(f"{model_name}_round_") and f.endswith("_metadata.json")]
        
        evolution_data = []
        for metadata_file in state_files:
            try:
                with open(os.path.join(self.states_dir, metadata_file), 'r') as f:
                    metadata = json.load(f)
                
                row = {
                    'round_num': metadata['round_num'],
                    'timestamp': metadata['timestamp'],
                    **metadata['metrics']
                }
                evolution_data.append(row)
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
        
        if evolution_data:
            df = pd.DataFrame(evolution_data)
            df = df.sort_values('round_num')
            return df
        else:
            return pd.DataFrame()
    
    def cleanup_old_states(self, keep_last_n: int = 10):
        """Clean up old model states, keeping only the most recent N states per model."""
        model_names = set()
        state_files = [f for f in os.listdir(self.states_dir) if f.endswith("_metadata.json")]
        
        # Extract model names
        for file in state_files:
            parts = file.split('_')
            if len(parts) >= 3:
                model_name = parts[0]
                model_names.add(model_name)
        
        # Clean up each model's states
        for model_name in model_names:
            model_files = [f for f in state_files if f.startswith(f"{model_name}_round_")]
            model_files.sort(reverse=True)  # Most recent first
            
            # Remove old states
            for file_to_remove in model_files[keep_last_n:]:
                state_id = file_to_remove.replace("_metadata.json", "")
                try:
                    # Remove model file
                    model_path = os.path.join(self.states_dir, f"{state_id}_model.pkl")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    
                    # Remove metadata file
                    metadata_path = os.path.join(self.states_dir, file_to_remove)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    logger.info(f"Cleaned up old state: {state_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up state {state_id}: {e}")


class IncrementalLearningManager:
    """Manages incremental learning capabilities for different model types."""
    
    @staticmethod
    def update_xgboost_model(model, X_new: np.ndarray, y_new: np.ndarray, 
                           learning_rate: float = 0.1, n_rounds: int = 10) -> Any:
        """Update XGBoost model with new data using incremental learning."""
        try:
            import xgboost as xgb
            
            # Create DMatrix for new data
            dtrain_new = xgb.DMatrix(X_new, label=y_new)
            
            # Get current booster
            booster = model.get_booster()
            
            # Update with new data
            for i in range(n_rounds):
                booster.update(dtrain_new, i)
            
            logger.info(f"Updated XGBoost model with {len(X_new)} new samples")
            return model
            
        except Exception as e:
            logger.error(f"Error updating XGBoost model: {e}")
            # Fallback: retrain on new data
            model.fit(X_new, y_new)
            return model
    
    @staticmethod
    def update_catboost_model(model, X_new: np.ndarray, y_new: np.ndarray, 
                            n_iterations: int = 50) -> Any:
        """Update CatBoost model with new data."""
        try:
            # CatBoost supports incremental learning through fit with init_model
            model.fit(X_new, y_new, verbose=False, init_model=model)
            logger.info(f"Updated CatBoost model with {len(X_new)} new samples")
            return model
            
        except Exception as e:
            logger.error(f"Error updating CatBoost model: {e}")
            # Fallback: retrain on new data
            model.fit(X_new, y_new, verbose=False)
            return model
    
    @staticmethod
    def update_random_forest_model(model, X_new: np.ndarray, y_new: np.ndarray, 
                                 n_estimators_add: int = 10) -> Any:
        """Update Random Forest model by adding new trees."""
        try:
            if hasattr(model, 'warm_start'):
                original_n_estimators = model.n_estimators
                model.warm_start = True
                model.n_estimators += n_estimators_add
                model.fit(X_new, y_new)
                
                logger.info(f"Updated Random Forest from {original_n_estimators} to {model.n_estimators} trees")
                return model
            else:
                # Fallback: retrain on new data
                model.fit(X_new, y_new)
                return model
                
        except Exception as e:
            logger.error(f"Error updating Random Forest model: {e}")
            # Fallback: retrain on new data
            model.fit(X_new, y_new)
            return model
    
    @staticmethod
    def update_model_by_type(model_type: str, model: Any, X_new: np.ndarray, 
                           y_new: np.ndarray, **kwargs) -> Any:
        """Update model based on its type."""
        if model_type.lower() == 'xgboost':
            return IncrementalLearningManager.update_xgboost_model(model, X_new, y_new, **kwargs)
        elif model_type.lower() == 'catboost':
            return IncrementalLearningManager.update_catboost_model(model, X_new, y_new, **kwargs)
        elif model_type.lower() == 'random_forest':
            return IncrementalLearningManager.update_random_forest_model(model, X_new, y_new, **kwargs)
        else:
            logger.warning(f"Unknown model type {model_type}, using default fit method")
            model.fit(X_new, y_new)
            return model