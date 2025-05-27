#!/usr/bin/env python3
"""
Main execution script for Bidirectional HeteroFL System
Provides unified entry point for both CLI and GUI modes.
"""

import os
import sys
import argparse
import logging
import threading
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('heterofl_bidirectional.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_cli_mode(rounds=5, config_overrides=None):
    """Run the bidirectional FL system in CLI mode."""
    logger.info("Starting Bidirectional HeteroFL in CLI mode")
    
    try:
        # Import and configure the FL system
        from heterofl_bidirectional import BidirectionalHeteroFL, BIDIRECTIONAL_CONFIG
        from heterofl_aggregation import MODEL_DIRS, GLOBAL_MODEL_TYPE, GLOBAL_MODEL_PARAMS
        
        # Apply configuration overrides
        fl_config = BIDIRECTIONAL_CONFIG.copy()
        fl_config['num_rounds'] = rounds
        if config_overrides:
            fl_config.update(config_overrides)
        
        # Setup global model configuration
        global_model_config = {
            'type': GLOBAL_MODEL_TYPE,
            'params': GLOBAL_MODEL_PARAMS[GLOBAL_MODEL_TYPE]
        }
        
        # Initialize and run FL system
        fl_system = BidirectionalHeteroFL(
            config=fl_config,
            model_dirs=MODEL_DIRS,
            global_model_config=global_model_config
        )
        
        success = fl_system.run_federated_learning(rounds)
        
        if success:
            logger.info("✓ Bidirectional FL completed successfully!")
            return True
        else:
            logger.error("✗ Bidirectional FL failed!")
            return False
            
    except Exception as e:
        logger.error(f"Error in CLI mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dashboard_mode(host='0.0.0.0', port=5000, debug=False):
    """Run the bidirectional FL system with GUI dashboard."""
    logger.info("Starting Bidirectional HeteroFL with GUI Dashboard")
    
    try:
        # Import Flask app
        from flask_dashboard.app import app, socketio
        from utils.websocket_handler import websocket_handler
        
        # Set the websocket handler in the Flask app
        websocket_handler.set_socketio(socketio)
        
        logger.info(f"Dashboard starting on http://{host}:{port}")
        logger.info("Open your browser and navigate to the dashboard to control the FL process")
        
        # Run the Flask app with SocketIO
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in dashboard mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_system_requirements():
    """Check if system requirements are met."""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("Python 3.7+ required")
        return False
    
    # Check required packages
    required_packages = [
        'flask', 'flask_socketio', 'plotly', 'pandas', 'numpy', 
        'sklearn', 'matplotlib', 'seaborn', 'xgboost', 'catboost'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    # Check file structure
    required_files = [
        'heterofl_bidirectional.py',
        'utils/model_manager.py',
        'utils/metrics_tracker.py',
        'utils/websocket_handler.py',
        'flask_dashboard/app.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    logger.info("✓ System requirements met")
    return True

def print_system_info():
    """Print system information and capabilities."""
    print("\n" + "="*80)
    print("BIDIRECTIONAL HETEROFL SYSTEM")
    print("="*80)
    print("Enhanced Heterogeneous Federated Learning with Bidirectional Communication")
    print()
    print("Key Features:")
    print("• Bidirectional FL: Global model sends feedback to local models")
    print("• Real-time GUI Dashboard: Monitor and control FL process")
    print("• Model State Management: Save/load model states between rounds")
    print("• Performance Tracking: Comprehensive metrics and evolution tracking")
    print("• Export Capabilities: Save models, metrics, and reports")
    print()
    print("Architecture:")
    print("Local Models → Predictions → Global Aggregation → Global Model")
    print("     ↑                                               ↓")
    print("Knowledge Transfer ← Pseudo-labels ← Global Feedback")
    print()
    print("="*80)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Bidirectional HeteroFL System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GUI dashboard (recommended)
  python run_bidirectional_heterofl.py --mode dashboard
  
  # Run CLI mode with 10 rounds
  python run_bidirectional_heterofl.py --mode cli --rounds 10
  
  # Run dashboard on custom host/port
  python run_bidirectional_heterofl.py --mode dashboard --host 127.0.0.1 --port 8080
  
  # Run with debug mode
  python run_bidirectional_heterofl.py --mode dashboard --debug
        """
    )
    
    parser.add_argument('--mode', choices=['dashboard', 'cli'], default='dashboard',
                       help='Execution mode (default: dashboard)')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of FL rounds for CLI mode (default: 5)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Dashboard host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Dashboard port (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip system requirement checks')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    
    args = parser.parse_args()
    
    # Show system info if requested
    if args.info:
        print_system_info()
        return
    
    print_system_info()
    
    # Check system requirements
    if not args.skip_checks:
        if not check_system_requirements():
            logger.error("System requirements not met. Use --skip-checks to bypass.")
            sys.exit(1)
    
    # Run based on mode
    success = False
    
    if args.mode == 'dashboard':
        logger.info("Starting in Dashboard mode...")
        success = run_dashboard_mode(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    elif args.mode == 'cli':
        logger.info("Starting in CLI mode...")
        success = run_cli_mode(rounds=args.rounds)
    
    # Exit with appropriate code
    if success:
        logger.info("System completed successfully!")
        sys.exit(0)
    else:
        logger.error("System encountered errors!")
        sys.exit(1)

if __name__ == '__main__':
    main()