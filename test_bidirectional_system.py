#!/usr/bin/env python3
"""
Test script for Bidirectional HeteroFL System
Verifies that all components are working correctly.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from utils.model_manager import ModelStateManager, IncrementalLearningManager
from utils.metrics_tracker import MetricsTracker
from utils.websocket_handler import WebSocketHandler, FLEventEmitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestModelStateManager(unittest.TestCase):
    """Test cases for ModelStateManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelStateManager(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """Test that necessary directories are created."""
        self.assertTrue(os.path.exists(self.manager.states_dir))
        self.assertTrue(os.path.exists(self.manager.checkpoints_dir))
    
    def test_save_and_load_model_state(self):
        """Test saving and loading model states."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1, 0, 1])
        
        metrics = {'accuracy': 0.85, 'f1': 0.82}
        
        # Save model state
        state_id = self.manager.save_model_state('test_model', mock_model, 1, metrics)
        self.assertIsNotNone(state_id)
        
        # Load model state
        loaded_model, metadata = self.manager.load_model_state(state_id)
        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['model_name'], 'test_model')
        self.assertEqual(metadata['round_num'], 1)
        self.assertEqual(metadata['metrics'], metrics)
    
    def test_get_latest_state(self):
        """Test getting the latest state for a model."""
        mock_model = Mock()
        
        # Save multiple states
        self.manager.save_model_state('test_model', mock_model, 1, {'accuracy': 0.8})
        self.manager.save_model_state('test_model', mock_model, 2, {'accuracy': 0.85})
        
        # Get latest state
        latest_model, metadata = self.manager.get_latest_state('test_model')
        self.assertIsNotNone(latest_model)
        self.assertEqual(metadata['round_num'], 2)
        self.assertEqual(metadata['metrics']['accuracy'], 0.85)


class TestMetricsTracker(unittest.TestCase):
    """Test cases for MetricsTracker."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tracker = MetricsTracker(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """Test that necessary directories are created."""
        self.assertTrue(os.path.exists(self.tracker.metrics_dir))
        self.assertTrue(os.path.exists(self.tracker.plots_dir))
        self.assertTrue(os.path.exists(self.tracker.exports_dir))
    
    def test_record_round_metrics(self):
        """Test recording round metrics."""
        # Record round start
        self.tracker.record_round_start(1)
        
        # Record local model metrics
        metrics = {'accuracy': 0.85, 'f1': 0.82}
        self.tracker.record_local_model_metrics(1, 'test_model', metrics)
        
        # Record global model metrics
        global_metrics = {'accuracy': 0.87, 'f1': 0.84}
        self.tracker.record_global_model_metrics(1, global_metrics)
        
        # Record round end
        self.tracker.record_round_end(1, 120.5)
        
        # Check data
        latest_metrics = self.tracker.get_latest_metrics()
        self.assertEqual(latest_metrics['round_num'], 1)
        self.assertIn('test_model', latest_metrics['local_models'])
        self.assertEqual(latest_metrics['local_models']['test_model']['accuracy'], 0.85)
    
    def test_model_evolution_tracking(self):
        """Test model evolution tracking."""
        # Record multiple rounds
        for round_num in range(1, 4):
            self.tracker.record_round_start(round_num)
            metrics = {'accuracy': 0.8 + round_num * 0.02, 'f1': 0.75 + round_num * 0.03}
            self.tracker.record_local_model_metrics(round_num, 'test_model', metrics)
        
        # Get evolution data
        evolution_df = self.tracker.get_model_evolution_df('test_model')
        self.assertEqual(len(evolution_df), 3)
        self.assertTrue(evolution_df['accuracy'].is_monotonic_increasing)
    
    def test_save_and_load_state(self):
        """Test saving and loading tracker state."""
        # Record some data
        self.tracker.record_round_start(1)
        self.tracker.record_local_model_metrics(1, 'test_model', {'accuracy': 0.85})
        
        # Save state
        state_file = os.path.join(self.test_dir, 'test_state.json')
        self.tracker.save_state(state_file)
        self.assertTrue(os.path.exists(state_file))
        
        # Create new tracker and load state
        new_tracker = MetricsTracker(self.test_dir)
        new_tracker.load_state(state_file)
        
        # Verify data
        self.assertEqual(len(new_tracker.round_metrics), 1)
        self.assertIn('test_model', new_tracker.model_evolution)


class TestWebSocketHandler(unittest.TestCase):
    """Test cases for WebSocketHandler."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_socketio = Mock()
        self.handler = WebSocketHandler(self.mock_socketio)
        
    def test_emit_status_update(self):
        """Test emitting status updates."""
        status_data = {'fl_status': 'running', 'current_round': 2}
        self.handler.emit_status_update(status_data)
        
        self.mock_socketio.emit.assert_called_with('status_update', status_data)
    
    def test_emit_round_events(self):
        """Test emitting round-related events."""
        # Test round start
        self.handler.emit_round_start(1)
        self.mock_socketio.emit.assert_called()
        
        # Test round complete
        self.handler.emit_round_complete(1, True, {'accuracy': 0.85})
        self.mock_socketio.emit.assert_called()
    
    def test_fl_event_emitter(self):
        """Test FLEventEmitter functionality."""
        emitter = FLEventEmitter(self.handler)
        
        # Test various events
        emitter.on_fl_start({'num_rounds': 5})
        emitter.on_models_loaded(['model1', 'model2'])
        emitter.on_round_start(1, 5)
        emitter.on_round_complete(1, True)
        emitter.on_fl_complete(5)
        
        # Verify emissions
        self.assertTrue(self.mock_socketio.emit.called)
        self.assertGreater(self.mock_socketio.emit.call_count, 5)


class TestIncrementalLearning(unittest.TestCase):
    """Test cases for IncrementalLearningManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = IncrementalLearningManager()
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_new = np.random.randn(50, 10)
        self.y_new = np.random.randint(0, 2, 50)
    
    def test_update_model_by_type(self):
        """Test updating models by type."""
        # Create mock models
        mock_model = Mock()
        mock_model.fit = Mock()
        
        # Test different model types
        model_types = ['xgboost', 'catboost', 'random_forest', 'unknown']
        
        for model_type in model_types:
            updated_model = self.manager.update_model_by_type(
                model_type, mock_model, self.X_new, self.y_new
            )
            self.assertIsNotNone(updated_model)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_component_integration(self):
        """Test that all components work together."""
        # Initialize components
        model_manager = ModelStateManager(self.test_dir)
        metrics_tracker = MetricsTracker(self.test_dir)
        websocket_handler = WebSocketHandler()
        
        # Simulate a FL round
        mock_model = Mock()
        metrics = {'accuracy': 0.85, 'f1': 0.82}
        
        # Save model state
        state_id = model_manager.save_model_state('test_model', mock_model, 1, metrics)
        self.assertIsNotNone(state_id)
        
        # Record metrics
        metrics_tracker.record_round_start(1)
        metrics_tracker.record_local_model_metrics(1, 'test_model', metrics)
        metrics_tracker.record_round_end(1, 120.0)
        
        # Verify integration
        latest_metrics = metrics_tracker.get_latest_metrics()
        self.assertEqual(latest_metrics['round_num'], 1)
        
        loaded_model, metadata = model_manager.load_model_state(state_id)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(metadata['metrics'], metrics)


def run_system_checks():
    """Run basic system checks."""
    logger.info("Running system checks...")
    
    # Check imports
    try:
        import flask
        import flask_socketio
        import plotly
        logger.info("✓ Flask dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing Flask dependencies: {e}")
        return False
    
    try:
        import sklearn
        import pandas
        import numpy
        logger.info("✓ ML dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing ML dependencies: {e}")
        return False
    
    # Check file structure
    required_files = [
        'heterofl_bidirectional.py',
        'utils/model_manager.py',
        'utils/metrics_tracker.py',
        'utils/websocket_handler.py',
        'flask_dashboard/app.py',
        'flask_dashboard/templates/dashboard.html',
        'flask_dashboard/static/js/dashboard.js'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    else:
        logger.info("✓ All required files present")
    
    # Check model directories
    model_dirs = ['xgboost', 'catboost', 'Random_forest']
    available_models = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            available_models.append(model_dir)
    
    if available_models:
        logger.info(f"✓ Available model directories: {available_models}")
    else:
        logger.warning("⚠ No model directories found - system will use synthetic data")
    
    logger.info("System checks completed successfully!")
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("HeteroFL Bidirectional System Test Suite")
    print("=" * 60)
    
    # Run system checks first
    if not run_system_checks():
        print("System checks failed!")
        sys.exit(1)
    
    # Run unit tests
    print("\nRunning unit tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestModelStateManager,
        TestMetricsTracker,
        TestWebSocketHandler,
        TestIncrementalLearning,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All tests passed! System is ready to use.")
        print("\nTo start the system:")
        print("1. Dashboard mode: python launch_dashboard.py")
        print("2. FL-only mode: python launch_dashboard.py --mode fl-only")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()