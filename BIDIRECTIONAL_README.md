# Bidirectional HeteroFL System

## Overview

This enhanced Heterogeneous Federated Learning (HeteroFL) system implements bidirectional communication where the global model sends fine-tuning signals back to local models after each aggregation round. The system includes a modern, responsive GUI dashboard for real-time monitoring and control.

## Architecture

```
┌────────────────────────────────────────────┐
│            Shared Public Dataset           │
└────────────────────┬───────────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │         Local Models           │
     │ (RF / XGB / CatBoost...)       │
     └────────────┬┬┬─────────────────┘
                  │││
┌─────────────────▼▼▼──────────────────────────┐
│     Each Local Model Predicts Soft Labels    │
└─────────────────┬▲▲▲──────────────────────────┘
                  │││
     ┌────────────▼▼▼─────────────┐
     │     Aggregation Engine     │
     │  (Weighted Averaging etc.) │
     └────────────┬┬┬─────────────┘
                  │││
     ┌────────────▼▼▼─────────────┐
     │      Global Model          │
     │  Trained from Soft Labels  │
     └────────────┬┬┬─────────────┘
                  │││
     ┌────────────▼▼▼─────────────┐
     │ Global Model Finetunes and │
     │   Generates New Knowledge  │
     └────────────┬┬┬─────────────┘
                  │││
┌─────────────────▼▼▼────────────────────┐
│ Sends Fine-Tuning Updates or Pseudo-Labels │
│     back to Local Models in Each Round     │
└─────────────────┬┬┬────────────────────┘
                  │││
     ┌────────────▼▼▼─────────────┐
     │    Locals Update Models    │
     │  using Fine-tuned Labels   │
     └────────────┬┬┬─────────────┘
                  │││
      [ Next Round Begins 🔁 ]
```

## Key Features

### 1. Bidirectional Federated Learning
- **Pseudo-label Generation**: Global model creates soft targets for local model retraining
- **Local Model Updates**: Each local model retrains using global knowledge + original data
- **Knowledge Transfer Metrics**: Track how much each model learns from global feedback
- **Iterative Feedback Loop**: Global → Local → Global knowledge transfer

### 2. Modern GUI Dashboard
- **Real-time Monitoring**: Live performance graphs, metrics, and system status
- **Interactive Controls**: Manual round triggering, model loading, configuration
- **Visual Flow Animations**: Animated data flow between models
- **Export Capabilities**: Save models, metrics, and reports in multiple formats

### 3. Enhanced Model Management
- **State Persistence**: Save/load complete model states between rounds
- **Incremental Learning**: Warm-start capabilities for all model types
- **Performance Tracking**: Comprehensive metrics collection and evolution tracking
- **Memory Management**: Efficient handling of large model states

## Installation

### Prerequisites
- Python 3.7+
- Required packages (see requirements.txt)

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have trained local models in the respective directories:
   - `xgboost/` - XGBoost model artifacts
   - `catboost/` - CatBoost ensemble model artifacts  
   - `Random_forest/` - Random Forest model artifacts

## Usage

### 1. Command Line Interface

Run the bidirectional federated learning system:
```bash
python heterofl_bidirectional.py
```

### 2. GUI Dashboard

Start the Flask dashboard:
```bash
cd flask_dashboard
python app.py
```

Then open your browser to `http://localhost:5000`

#### Dashboard Features:
- **Status Panel**: Real-time system status and progress tracking
- **Control Panel**: Start/stop FL process, configure rounds, export data
- **Model Evolution**: Live charts showing local model performance over rounds
- **Global Model**: Global model performance metrics and evolution
- **Class Distribution**: Heatmap visualization of prediction distributions
- **System Logs**: Real-time logging and alerts

### 3. Configuration

Key configuration options in `heterofl_bidirectional.py`:

```python
BIDIRECTIONAL_CONFIG = {
    'num_rounds': 5,                              # Number of FL rounds
    'pseudo_label_confidence_threshold': 0.7,     # Confidence threshold for pseudo-labels
    'local_update_learning_rate': 0.1,            # Learning rate for local updates
    'local_update_epochs': 3,                     # Epochs for local model updates
    'feedback_sample_size': 10000,                # Size of feedback dataset
    'knowledge_transfer_weight': 0.3,             # Weight for knowledge transfer
    'enable_model_checkpoints': True,             # Enable model state checkpoints
    'checkpoint_frequency': 2,                    # Checkpoint every N rounds
    'enable_metrics_tracking': True               # Enable comprehensive metrics
}
```

## File Structure

```
├── heterofl_bidirectional.py          # Main bidirectional FL engine
├── utils/
│   ├── model_manager.py               # Model state management
│   ├── metrics_tracker.py             # Performance tracking
│   └── websocket_handler.py           # Real-time communication
├── flask_dashboard/
│   ├── app.py                         # Flask application
│   ├── templates/
│   │   └── dashboard.html             # Dashboard HTML template
│   └── static/
│       └── js/
│           └── dashboard.js           # Dashboard JavaScript
├── heterofl_aggregation.py            # Original aggregation system
├── requirements.txt                   # Updated dependencies
└── BIDIRECTIONAL_README.md           # This file
```

## API Endpoints

The Flask dashboard provides the following API endpoints:

### Status and Metrics
- `GET /api/status` - Get current system status
- `GET /api/metrics/latest` - Get latest metrics
- `GET /api/metrics/evolution/<model_name>` - Get model evolution data
- `GET /api/metrics/global_evolution` - Get global model evolution

### Control
- `POST /api/control/start` - Start FL process
- `POST /api/control/stop` - Stop FL process

### Visualization
- `GET /api/plots/model_evolution` - Model evolution plot data
- `GET /api/plots/global_evolution` - Global model evolution plot data
- `GET /api/plots/class_distribution` - Class distribution heatmap data

### Export
- `GET /api/export/metrics` - Export metrics to CSV
- `GET /api/export/models` - Export trained models

## WebSocket Events

Real-time communication via WebSocket:

### Client → Server
- `connect` - Client connection
- `request_metrics` - Request latest metrics

### Server → Client
- `status_update` - System status updates
- `round_start` - Round start notification
- `round_complete` - Round completion notification
- `metrics_update` - Metrics updates
- `fl_complete` - FL process completion
- `fl_error` - Error notifications

## Advanced Features

### 1. Model State Management
- Automatic model state saving after each round
- Checkpoint creation for system recovery
- Model evolution tracking and analysis
- Cleanup of old states to manage storage

### 2. Incremental Learning
- XGBoost: Boosting with new data using `update()` method
- CatBoost: Incremental training with `init_model` parameter
- Random Forest: Warm-start with additional trees

### 3. Knowledge Transfer
- Confidence-based pseudo-label filtering
- Weighted combination of original and global knowledge
- Transfer learning metrics and effectiveness tracking

### 4. Performance Monitoring
- Round-by-round performance tracking
- Model-specific metrics collection
- Aggregation statistics and analysis
- Export capabilities for detailed analysis

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure model files exist in correct directories
2. **Dashboard not connecting**: Check Flask server is running on correct port
3. **Memory issues**: Reduce `feedback_sample_size` or enable cleanup
4. **Feature alignment errors**: Check model feature compatibility

### Logging

The system provides comprehensive logging:
- `heterofl_bidirectional.log` - Main FL system logs
- Dashboard logs in browser console
- Real-time logs in dashboard interface

### Performance Optimization

- Adjust `feedback_sample_size` based on available memory
- Use `checkpoint_frequency` to balance storage vs recovery capability
- Enable/disable metrics tracking based on requirements
- Configure batch update intervals for dashboard responsiveness

## Contributing

When contributing to the bidirectional FL system:

1. Follow existing code structure and naming conventions
2. Add comprehensive logging for new features
3. Update metrics tracking for new components
4. Test with existing model artifacts
5. Update documentation for new features

## License

This project extends the original HeteroFL system with bidirectional capabilities and modern GUI interface while maintaining compatibility with existing components.