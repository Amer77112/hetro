"""
Comprehensive Performance Tracking for Bidirectional HeteroFL
Tracks metrics, model evolution, and provides export capabilities.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and manages performance metrics across federated learning rounds."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.metrics_dir = os.path.join(base_dir, 'metrics')
        self.plots_dir = os.path.join(base_dir, 'heterofl_plots')
        self.exports_dir = os.path.join(base_dir, 'exports')
        self._ensure_directories()
        
        # Initialize tracking data
        self.round_metrics = []
        self.model_evolution = {}
        self.aggregation_history = []
        self.global_model_history = []
        
    def _ensure_directories(self):
        """Create necessary directories for metrics tracking."""
        for directory in [self.metrics_dir, self.plots_dir, self.exports_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def record_round_start(self, round_num: int, timestamp: str = None):
        """Record the start of a federated learning round."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        round_data = {
            'round_num': round_num,
            'start_time': timestamp,
            'local_models': {},
            'aggregation_metrics': {},
            'global_model_metrics': {}
        }
        
        self.round_metrics.append(round_data)
        logger.info(f"Started tracking round {round_num}")
    
    def record_local_model_metrics(self, round_num: int, model_name: str, 
                                 metrics: Dict[str, float], predictions_stats: Dict = None):
        """Record metrics for a local model in a specific round."""
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not initialized, creating new round entry")
            self.record_round_start(round_num)
        
        current_round = self.round_metrics[-1]
        current_round['local_models'][model_name] = {
            'metrics': metrics,
            'predictions_stats': predictions_stats or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Update model evolution tracking
        if model_name not in self.model_evolution:
            self.model_evolution[model_name] = []
        
        evolution_entry = {
            'round_num': round_num,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.model_evolution[model_name].append(evolution_entry)
        
        logger.info(f"Recorded metrics for {model_name} in round {round_num}")
    
    def record_aggregation_metrics(self, round_num: int, aggregation_stats: Dict[str, Any]):
        """Record aggregation process metrics."""
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not initialized")
            return
        
        current_round = self.round_metrics[-1]
        current_round['aggregation_metrics'] = {
            **aggregation_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to aggregation history
        aggregation_entry = {
            'round_num': round_num,
            'timestamp': datetime.now().isoformat(),
            **aggregation_stats
        }
        self.aggregation_history.append(aggregation_entry)
        
        logger.info(f"Recorded aggregation metrics for round {round_num}")
    
    def record_global_model_metrics(self, round_num: int, metrics: Dict[str, float], 
                                  additional_info: Dict = None):
        """Record global model performance metrics."""
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not initialized")
            return
        
        current_round = self.round_metrics[-1]
        current_round['global_model_metrics'] = {
            'metrics': metrics,
            'additional_info': additional_info or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to global model history
        global_entry = {
            'round_num': round_num,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.global_model_history.append(global_entry)
        
        logger.info(f"Recorded global model metrics for round {round_num}")
    
    def record_round_end(self, round_num: int, total_time: float):
        """Record the end of a federated learning round."""
        if not self.round_metrics or self.round_metrics[-1]['round_num'] != round_num:
            logger.warning(f"Round {round_num} not found")
            return
        
        current_round = self.round_metrics[-1]
        current_round['end_time'] = datetime.now().isoformat()
        current_round['total_time'] = total_time
        
        logger.info(f"Completed tracking round {round_num} (duration: {total_time:.2f}s)")
    
    def get_model_evolution_df(self, model_name: str = None) -> pd.DataFrame:
        """Get model evolution data as DataFrame."""
        if model_name:
            if model_name in self.model_evolution:
                return pd.DataFrame(self.model_evolution[model_name])
            else:
                return pd.DataFrame()
        else:
            # Return all models' evolution
            all_data = []
            for name, evolution in self.model_evolution.items():
                for entry in evolution:
                    entry_copy = entry.copy()
                    entry_copy['model_name'] = name
                    all_data.append(entry_copy)
            return pd.DataFrame(all_data)
    
    def get_global_model_evolution_df(self) -> pd.DataFrame:
        """Get global model evolution data as DataFrame."""
        return pd.DataFrame(self.global_model_history)
    
    def get_aggregation_history_df(self) -> pd.DataFrame:
        """Get aggregation history as DataFrame."""
        return pd.DataFrame(self.aggregation_history)
    
    def get_round_summary(self, round_num: int = None) -> Dict:
        """Get summary of a specific round or all rounds."""
        if round_num is not None:
            for round_data in self.round_metrics:
                if round_data['round_num'] == round_num:
                    return round_data
            return {}
        else:
            return {
                'total_rounds': len(self.round_metrics),
                'rounds': self.round_metrics
            }
    
    def plot_model_evolution(self, metric: str = 'accuracy', save_path: str = None):
        """Plot model evolution across rounds."""
        df = self.get_model_evolution_df()
        if df.empty:
            logger.warning("No model evolution data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            if metric in model_data.columns:
                plt.plot(model_data['round_num'], model_data[metric], 
                        marker='o', label=f'{model_name}', linewidth=2)
        
        plt.xlabel('Round Number')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Evolution: {metric.capitalize()} Across Rounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f'model_evolution_{metric}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model evolution plot to: {save_path}")
    
    def plot_global_model_evolution(self, save_path: str = None):
        """Plot global model performance evolution."""
        df = self.get_global_model_evolution_df()
        if df.empty:
            logger.warning("No global model evolution data to plot")
            return
        
        metrics = [col for col in df.columns if col not in ['round_num', 'timestamp']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
            if i < len(axes):
                axes[i].plot(df['round_num'], df[metric], marker='o', linewidth=2, color='blue')
                axes[i].set_xlabel('Round Number')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].set_title(f'Global Model {metric.capitalize()}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'global_model_evolution.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved global model evolution plot to: {save_path}")
    
    def plot_class_distribution_heatmap(self, round_data: Dict, save_path: str = None):
        """Plot class distribution heatmap for a specific round."""
        if 'local_models' not in round_data:
            logger.warning("No local model data for heatmap")
            return
        
        # Extract class distributions
        models = []
        class_data = []
        
        for model_name, model_info in round_data['local_models'].items():
            if 'predictions_stats' in model_info and 'class_distribution' in model_info['predictions_stats']:
                models.append(model_name)
                class_dist = model_info['predictions_stats']['class_distribution']
                class_data.append(list(class_dist.values()))
        
        if not class_data:
            logger.warning("No class distribution data available")
            return
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        class_matrix = np.array(class_data)
        
        sns.heatmap(class_matrix, 
                   xticklabels=[f'Class {i}' for i in range(class_matrix.shape[1])],
                   yticklabels=models,
                   annot=True, fmt='d', cmap='Blues')
        
        plt.title(f"Class Distribution Heatmap - Round {round_data['round_num']}")
        plt.xlabel('Classes')
        plt.ylabel('Models')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, f"class_distribution_round_{round_data['round_num']}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution heatmap to: {save_path}")
    
    def export_metrics_csv(self, export_path: str = None):
        """Export all metrics to CSV files."""
        if export_path is None:
            export_path = self.exports_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export model evolution
        model_evolution_df = self.get_model_evolution_df()
        if not model_evolution_df.empty:
            model_path = os.path.join(export_path, f'model_evolution_{timestamp}.csv')
            model_evolution_df.to_csv(model_path, index=False)
            logger.info(f"Exported model evolution to: {model_path}")
        
        # Export global model evolution
        global_evolution_df = self.get_global_model_evolution_df()
        if not global_evolution_df.empty:
            global_path = os.path.join(export_path, f'global_model_evolution_{timestamp}.csv')
            global_evolution_df.to_csv(global_path, index=False)
            logger.info(f"Exported global model evolution to: {global_path}")
        
        # Export aggregation history
        aggregation_df = self.get_aggregation_history_df()
        if not aggregation_df.empty:
            agg_path = os.path.join(export_path, f'aggregation_history_{timestamp}.csv')
            aggregation_df.to_csv(agg_path, index=False)
            logger.info(f"Exported aggregation history to: {agg_path}")
        
        # Export round summary
        round_summary = self.get_round_summary()
        summary_path = os.path.join(export_path, f'round_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(round_summary, f, indent=2)
        logger.info(f"Exported round summary to: {summary_path}")
    
    def save_state(self, file_path: str = None):
        """Save the current state of metrics tracker."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.metrics_dir, f'metrics_state_{timestamp}.json')
        
        state_data = {
            'round_metrics': self.round_metrics,
            'model_evolution': self.model_evolution,
            'aggregation_history': self.aggregation_history,
            'global_model_history': self.global_model_history,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"Saved metrics state to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving metrics state: {e}")
    
    def load_state(self, file_path: str):
        """Load a previously saved state."""
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            self.round_metrics = state_data.get('round_metrics', [])
            self.model_evolution = state_data.get('model_evolution', {})
            self.aggregation_history = state_data.get('aggregation_history', [])
            self.global_model_history = state_data.get('global_model_history', [])
            
            logger.info(f"Loaded metrics state from: {file_path}")
        except Exception as e:
            logger.error(f"Error loading metrics state: {e}")
    
    def get_latest_metrics(self) -> Dict:
        """Get the latest metrics from the most recent round."""
        if not self.round_metrics:
            return {}
        
        latest_round = self.round_metrics[-1]
        
        # Get latest metrics for each model
        latest_local_metrics = {}
        for model_name, model_data in latest_round.get('local_models', {}).items():
            latest_local_metrics[model_name] = model_data.get('metrics', {})
        
        # Get latest global metrics
        latest_global_metrics = latest_round.get('global_model_metrics', {}).get('metrics', {})
        
        # Get latest aggregation metrics
        latest_aggregation_metrics = latest_round.get('aggregation_metrics', {})
        
        return {
            'round_num': latest_round['round_num'],
            'local_models': latest_local_metrics,
            'global_model': latest_global_metrics,
            'aggregation': latest_aggregation_metrics,
            'timestamp': latest_round.get('end_time', latest_round.get('start_time', ''))
        }