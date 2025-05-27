"""
Flask Dashboard for Bidirectional HeteroFL System
Real-time monitoring and control interface for federated learning.
"""

import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import logging
from typing import Dict, Any, List

# Import our FL system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heterofl_bidirectional import BidirectionalHeteroFL, BIDIRECTIONAL_CONFIG
from utils.metrics_tracker import MetricsTracker
from utils.model_manager import ModelStateManager

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'heterofl_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for FL system state
fl_system = None
fl_thread = None
fl_running = False
current_round = 0
total_rounds = 0

# Dashboard state
dashboard_state = {
    'fl_status': 'idle',  # idle, running, paused, completed, error
    'current_round': 0,
    'total_rounds': 0,
    'round_progress': 0,
    'models_loaded': [],
    'latest_metrics': {},
    'system_logs': [],
    'round_history': []
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics tracker and model manager
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metrics_tracker = MetricsTracker(script_dir)
model_manager = ModelStateManager(script_dir)


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify(dashboard_state)


@app.route('/api/metrics/latest')
def get_latest_metrics():
    """Get latest metrics from the FL system."""
    try:
        latest_metrics = metrics_tracker.get_latest_metrics()
        return jsonify(latest_metrics)
    except Exception as e:
        logger.error(f"Error getting latest metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/evolution/<model_name>')
def get_model_evolution(model_name):
    """Get evolution data for a specific model."""
    try:
        evolution_df = metrics_tracker.get_model_evolution_df(model_name)
        if not evolution_df.empty:
            return jsonify(evolution_df.to_dict('records'))
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting model evolution for {model_name}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/global_evolution')
def get_global_evolution():
    """Get global model evolution data."""
    try:
        evolution_df = metrics_tracker.get_global_model_evolution_df()
        if not evolution_df.empty:
            return jsonify(evolution_df.to_dict('records'))
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting global model evolution: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/model_evolution')
def get_model_evolution_plot():
    """Generate model evolution plot data for Plotly."""
    try:
        evolution_df = metrics_tracker.get_model_evolution_df()
        if evolution_df.empty:
            return jsonify({'data': [], 'layout': {}})
        
        # Create traces for each model
        traces = []
        for model_name in evolution_df['model_name'].unique():
            model_data = evolution_df[evolution_df['model_name'] == model_name]
            
            trace = go.Scatter(
                x=model_data['round_num'],
                y=model_data.get('accuracy', []),
                mode='lines+markers',
                name=model_name,
                line=dict(width=3),
                marker=dict(size=8)
            )
            traces.append(trace)
        
        layout = go.Layout(
            title='Model Evolution: Accuracy Across Rounds',
            xaxis=dict(title='Round Number'),
            yaxis=dict(title='Accuracy'),
            hovermode='closest',
            showlegend=True
        )
        
        fig = go.Figure(data=traces, layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error generating model evolution plot: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/global_evolution')
def get_global_evolution_plot():
    """Generate global model evolution plot data."""
    try:
        evolution_df = metrics_tracker.get_global_model_evolution_df()
        if evolution_df.empty:
            return jsonify({'data': [], 'layout': {}})
        
        # Create traces for different metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        traces = []
        
        for metric in metrics:
            if metric in evolution_df.columns:
                trace = go.Scatter(
                    x=evolution_df['round_num'],
                    y=evolution_df[metric],
                    mode='lines+markers',
                    name=metric.capitalize(),
                    line=dict(width=3),
                    marker=dict(size=8)
                )
                traces.append(trace)
        
        layout = go.Layout(
            title='Global Model Performance Evolution',
            xaxis=dict(title='Round Number'),
            yaxis=dict(title='Score'),
            hovermode='closest',
            showlegend=True
        )
        
        fig = go.Figure(data=traces, layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error generating global evolution plot: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/class_distribution')
def get_class_distribution_plot():
    """Generate class distribution heatmap."""
    try:
        latest_metrics = metrics_tracker.get_latest_metrics()
        if not latest_metrics or 'local_models' not in latest_metrics:
            return jsonify({'data': [], 'layout': {}})
        
        # Extract class distribution data
        models = []
        class_data = []
        
        for model_name, model_metrics in latest_metrics['local_models'].items():
            # This would need to be implemented based on actual metrics structure
            models.append(model_name)
            # Placeholder data - would be replaced with actual class distribution
            class_data.append([10, 20, 30, 40])  # Example class counts
        
        if not class_data:
            return jsonify({'data': [], 'layout': {}})
        
        trace = go.Heatmap(
            z=class_data,
            x=[f'Class {i}' for i in range(len(class_data[0]))],
            y=models,
            colorscale='Blues'
        )
        
        layout = go.Layout(
            title='Class Distribution Heatmap',
            xaxis=dict(title='Classes'),
            yaxis=dict(title='Models')
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error generating class distribution plot: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/control/start', methods=['POST'])
def start_fl():
    """Start federated learning process."""
    global fl_system, fl_thread, fl_running
    
    try:
        if fl_running:
            return jsonify({'error': 'FL process already running'}), 400
        
        # Get configuration from request
        config = request.json or {}
        num_rounds = config.get('num_rounds', BIDIRECTIONAL_CONFIG['num_rounds'])
        
        # Initialize FL system
        from heterofl_aggregation import MODEL_DIRS, GLOBAL_MODEL_TYPE, GLOBAL_MODEL_PARAMS
        
        global_model_config = {
            'type': GLOBAL_MODEL_TYPE,
            'params': GLOBAL_MODEL_PARAMS[GLOBAL_MODEL_TYPE]
        }
        
        fl_config = BIDIRECTIONAL_CONFIG.copy()
        fl_config.update(config)
        
        fl_system = BidirectionalHeteroFL(
            config=fl_config,
            model_dirs=MODEL_DIRS,
            global_model_config=global_model_config
        )
        
        # Start FL in separate thread
        fl_thread = threading.Thread(target=run_fl_process, args=(num_rounds,))
        fl_thread.daemon = True
        fl_thread.start()
        
        # Update dashboard state
        dashboard_state.update({
            'fl_status': 'running',
            'total_rounds': num_rounds,
            'current_round': 0
        })
        
        # Emit status update
        socketio.emit('status_update', dashboard_state)
        
        return jsonify({'message': 'FL process started', 'rounds': num_rounds})
        
    except Exception as e:
        logger.error(f"Error starting FL process: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/control/stop', methods=['POST'])
def stop_fl():
    """Stop federated learning process."""
    global fl_running
    
    try:
        fl_running = False
        dashboard_state['fl_status'] = 'idle'
        
        socketio.emit('status_update', dashboard_state)
        
        return jsonify({'message': 'FL process stopped'})
        
    except Exception as e:
        logger.error(f"Error stopping FL process: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/metrics')
def export_metrics():
    """Export metrics to CSV."""
    try:
        # Export metrics
        export_path = metrics_tracker.exports_dir
        metrics_tracker.export_metrics_csv(export_path)
        
        # Return the most recent export file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_evolution_{timestamp}.csv'
        filepath = os.path.join(export_path, filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Export file not found'}), 404
            
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/models')
def export_models():
    """Export trained models."""
    try:
        # Create a zip file with all models
        import zipfile
        from io import BytesIO
        
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add global model
            global_model_path = os.path.join(script_dir, 'heterofl_models', 'heterofl_global_model.pkl')
            if os.path.exists(global_model_path):
                zf.write(global_model_path, 'global_model.pkl')
            
            # Add local models
            for model_dir in ['xgboost', 'catboost', 'Random_forest']:
                model_path = os.path.join(script_dir, model_dir)
                if os.path.exists(model_path):
                    for file in os.listdir(model_path):
                        if file.endswith('.pkl'):
                            file_path = os.path.join(model_path, file)
                            zf.write(file_path, f'{model_dir}/{file}')
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'heterofl_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
        
    except Exception as e:
        logger.error(f"Error exporting models: {e}")
        return jsonify({'error': str(e)}), 500


def run_fl_process(num_rounds):
    """Run the federated learning process in a separate thread."""
    global fl_running, current_round, dashboard_state
    
    try:
        fl_running = True
        logger.info(f"Starting FL process for {num_rounds} rounds")
        
        # Load models
        dashboard_state['fl_status'] = 'loading'
        socketio.emit('status_update', dashboard_state)
        
        if not fl_system.load_local_models():
            raise Exception("Failed to load local models")
        
        dashboard_state['models_loaded'] = list(fl_system.local_models.keys())
        
        # Prepare datasets
        from heterofl_aggregation import DATASET_PATH, SAMPLE_SIZE
        if not fl_system.prepare_datasets(DATASET_PATH, SAMPLE_SIZE):
            raise Exception("Failed to prepare datasets")
        
        dashboard_state['fl_status'] = 'running'
        socketio.emit('status_update', dashboard_state)
        
        # Run rounds
        for round_num in range(1, num_rounds + 1):
            if not fl_running:
                break
            
            current_round = round_num
            dashboard_state.update({
                'current_round': round_num,
                'round_progress': (round_num / num_rounds) * 100
            })
            
            # Emit round start
            socketio.emit('round_start', {'round': round_num})
            socketio.emit('status_update', dashboard_state)
            
            # Run FL round
            success = fl_system.run_federated_round(round_num)
            
            if success:
                # Get latest metrics and emit updates
                latest_metrics = metrics_tracker.get_latest_metrics()
                dashboard_state['latest_metrics'] = latest_metrics
                
                socketio.emit('metrics_update', latest_metrics)
                socketio.emit('round_complete', {'round': round_num, 'success': True})
            else:
                socketio.emit('round_complete', {'round': round_num, 'success': False})
                logger.warning(f"Round {round_num} failed")
            
            # Small delay to allow UI updates
            time.sleep(1)
        
        # Complete
        dashboard_state['fl_status'] = 'completed'
        socketio.emit('fl_complete', {'total_rounds': num_rounds})
        socketio.emit('status_update', dashboard_state)
        
        logger.info("FL process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in FL process: {e}")
        dashboard_state['fl_status'] = 'error'
        socketio.emit('fl_error', {'error': str(e)})
        socketio.emit('status_update', dashboard_state)
    
    finally:
        fl_running = False


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status_update', dashboard_state)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')


@socketio.on('request_metrics')
def handle_metrics_request():
    """Handle request for latest metrics."""
    try:
        latest_metrics = metrics_tracker.get_latest_metrics()
        emit('metrics_update', latest_metrics)
    except Exception as e:
        logger.error(f"Error handling metrics request: {e}")
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('flask_dashboard/templates', exist_ok=True)
    os.makedirs('flask_dashboard/static/css', exist_ok=True)
    os.makedirs('flask_dashboard/static/js', exist_ok=True)
    
    # Run the Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)