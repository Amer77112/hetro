#!/usr/bin/env python3
"""
Installation script for HeteroFL Bidirectional System
Helps users set up the environment and dependencies.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        logger.error(f"Python 3.7+ required, but found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    logger.info("Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        logger.info("✓ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error installing packages: {e}")
        return False
    except FileNotFoundError:
        logger.error("✗ requirements.txt not found")
        return False

def create_directories():
    """Create necessary directories for the system."""
    logger.info("Creating necessary directories...")
    
    directories = [
        'heterofl_models',
        'heterofl_plots', 
        'heterofl_debug',
        'model_states',
        'checkpoints',
        'metrics',
        'exports',
        'flask_dashboard/static/css',
        'flask_dashboard/static/js',
        'flask_dashboard/templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True

def check_optional_dependencies():
    """Check for optional dependencies and provide recommendations."""
    logger.info("Checking optional dependencies...")
    
    optional_packages = {
        'torch': 'PyTorch for advanced neural network models',
        'tensorflow': 'TensorFlow for deep learning models',
        'jupyter': 'Jupyter notebooks for interactive analysis',
        'ipywidgets': 'Interactive widgets for Jupyter'
    }
    
    missing_optional = []
    for package, description in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_optional.append((package, description))
    
    if missing_optional:
        logger.info("Optional packages not found (can be installed later):")
        for package, description in missing_optional:
            logger.info(f"  - {package}: {description}")
    
    return True

def verify_installation():
    """Verify that the installation was successful."""
    logger.info("Verifying installation...")
    
    # Test imports
    test_imports = [
        'flask',
        'flask_socketio', 
        'plotly',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'catboost',
        'lightgbm'
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            logger.info(f"✓ {module} imported successfully")
        except ImportError as e:
            failed_imports.append((module, str(e)))
            logger.error(f"✗ Failed to import {module}: {e}")
    
    if failed_imports:
        logger.error("Some imports failed. Please check the errors above.")
        return False
    
    # Test file structure
    required_files = [
        'heterofl_bidirectional.py',
        'utils/model_manager.py',
        'utils/metrics_tracker.py',
        'utils/websocket_handler.py',
        'flask_dashboard/app.py',
        'flask_dashboard/templates/dashboard.html',
        'flask_dashboard/static/js/dashboard.js',
        'launch_dashboard.py',
        'test_bidirectional_system.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✓ All required files are present")
    return True

def create_sample_config():
    """Create a sample configuration file."""
    logger.info("Creating sample configuration...")
    
    config_content = """# HeteroFL Bidirectional Configuration
# Copy this file to config.py and modify as needed

# Federated Learning Configuration
FL_CONFIG = {
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

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'auto_refresh_interval': 30  # seconds
}

# Model Configuration
MODEL_CONFIG = {
    'use_synthetic_data': True,
    'synthetic_data_size': 50000,
    'sample_size': 1000000,
    'dataset_preference': 'auto'  # 'auto', 'real', 'synthetic'
}
"""
    
    config_file = Path('config_sample.py')
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✓ Created sample configuration: {config_file}")
    return True

def print_next_steps():
    """Print next steps for the user."""
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    print("\nNext steps:")
    print("1. Train your local models (XGBoost, CatBoost, Random Forest)")
    print("   - Place model files in respective directories:")
    print("     * xgboost/xgboost_model.pkl, label_encoder.pkl, scaler.pkl")
    print("     * catboost/ensemble_model.pkl, label_encoder.pkl, scaler.pkl") 
    print("     * Random_forest/random_forest_model.pkl, label_encoder.pkl")
    print()
    print("2. Test the installation:")
    print("   python test_bidirectional_system.py")
    print()
    print("3. Start the system:")
    print("   a) Dashboard mode (recommended):")
    print("      python launch_dashboard.py")
    print("   b) Command-line mode:")
    print("      python launch_dashboard.py --mode fl-only --rounds 5")
    print()
    print("4. Access the dashboard:")
    print("   Open http://localhost:5000 in your browser")
    print()
    print("5. For help:")
    print("   python launch_dashboard.py --help")
    print("   Read BIDIRECTIONAL_README.md for detailed documentation")
    print()
    print("="*60)

def main():
    """Main installation function."""
    logger.info("="*60)
    logger.info("HeteroFL Bidirectional System Installation")
    logger.info("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        sys.exit(1)
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Verify installation
    if not verify_installation():
        logger.error("Installation verification failed")
        sys.exit(1)
    
    # Create sample config
    create_sample_config()
    
    # Print next steps
    print_next_steps()

if __name__ == '__main__':
    main()