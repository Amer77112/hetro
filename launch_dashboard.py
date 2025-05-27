#!/usr/bin/env python3
"""
Launcher script for HeteroFL Bidirectional Dashboard
Provides easy startup and configuration options.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask', 'flask_socketio', 'plotly', 'pandas', 'numpy', 
        'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist."""
    script_dir = Path(__file__).parent
    model_dirs = ['xgboost', 'catboost', 'Random_forest']
    
    missing_models = []
    for model_dir in model_dirs:
        model_path = script_dir / model_dir
        if not model_path.exists():
            missing_models.append(model_dir)
            continue
        
        # Check for essential files
        essential_files = ['label_encoder.pkl', 'scaler.pkl']
        for file in essential_files:
            if not (model_path / file).exists():
                missing_models.append(f"{model_dir}/{file}")
    
    if missing_models:
        logger.warning(f"Missing model files/directories: {', '.join(missing_models)}")
        logger.info("Some models may not be available. Train models first or use synthetic data.")
    
    return len(missing_models) == 0

def start_dashboard(host='0.0.0.0', port=5000, debug=False, open_browser=True):
    """Start the Flask dashboard."""
    logger.info("Starting HeteroFL Bidirectional Dashboard...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'flask_dashboard.app'
    if debug:
        os.environ['FLASK_ENV'] = 'development'
    
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent / 'flask_dashboard'
    os.chdir(dashboard_dir)
    
    # Start Flask app
    try:
        from flask_dashboard.app import app, socketio
        
        logger.info(f"Dashboard starting on http://{host}:{port}")
        
        if open_browser and not debug:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{port}')
            
            import threading
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()
        
        # Run the app
        socketio.run(app, host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False
    
    return True

def run_fl_only(rounds=5):
    """Run only the FL system without dashboard."""
    logger.info(f"Running bidirectional FL for {rounds} rounds...")
    
    try:
        # Import and run FL system
        from heterofl_bidirectional import main
        
        # Update configuration
        from heterofl_bidirectional import BIDIRECTIONAL_CONFIG
        BIDIRECTIONAL_CONFIG['num_rounds'] = rounds
        
        # Run FL
        main()
        
    except Exception as e:
        logger.error(f"Error running FL system: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='HeteroFL Bidirectional System Launcher')
    
    parser.add_argument('--mode', choices=['dashboard', 'fl-only'], default='dashboard',
                       help='Run mode: dashboard (GUI) or fl-only (command line)')
    parser.add_argument('--host', default='0.0.0.0', help='Dashboard host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds (for fl-only mode)')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and model checks')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HeteroFL Bidirectional System Launcher")
    logger.info("=" * 60)
    
    # Perform checks unless skipped
    if not args.skip_checks:
        logger.info("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        logger.info("Checking model files...")
        check_model_files()  # Warning only, don't exit
    
    # Run based on mode
    if args.mode == 'dashboard':
        logger.info("Starting dashboard mode...")
        success = start_dashboard(
            host=args.host,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
    else:
        logger.info("Starting FL-only mode...")
        success = run_fl_only(rounds=args.rounds)
    
    if success:
        logger.info("System completed successfully!")
    else:
        logger.error("System encountered errors!")
        sys.exit(1)

if __name__ == '__main__':
    main()