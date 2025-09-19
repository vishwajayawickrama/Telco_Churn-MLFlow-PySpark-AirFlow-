import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class BaseModelBuilder(ABC):
    """
    Abstract base class for model builders.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize base model builder.
        
        Args:
            model_name (str): Name of the model
            **kwargs: Model parameters
        """
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        self.build_timestamp = None
        
        ProjectLogger.log_section_header(logger, f"INITIALIZING {model_name.upper()} MODEL BUILDER")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Model parameters ({len(self.model_params)}):")
        
        for param, value in self.model_params.items():
            logger.info(f"  - {param}: {value}")

    @abstractmethod
    def build_model(self):
        """
        Abstract method to build the model.
        
        Returns:
            Model object
        """
        pass

    @log_exceptions(logger)
    def save_model(self, filepath: str, create_dirs: bool = True) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
            create_dirs (bool): Whether to create directories if they don't exist
            
        Raises:
            ValueError: If no model is built or filepath is invalid
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "SAVING MODEL")
        
        try:
            # Validate model exists
            if self.model is None:
                raise ValueError("No model to save. Build the model first using build_model()")
            
            # Validate filepath
            if not filepath:
                raise ValueError("Filepath cannot be empty")
            
            # Create directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                logger.info(f"Directory created/verified: {os.path.dirname(filepath)}")
            
            # Get file size before saving (if exists)
            file_existed = os.path.exists(filepath)
            old_size = os.path.getsize(filepath) if file_existed else 0
            
            logger.info(f"Saving {self.model_name} model to: {filepath}")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Verify save and get file info
            if os.path.exists(filepath):
                new_size = os.path.getsize(filepath)
                logger.info(f"Model saved successfully")
                logger.info(f"File size: {new_size:,} bytes ({new_size / 1024 / 1024:.2f} MB)")
                
                if file_existed:
                    size_diff = new_size - old_size
                    logger.info(f"Size change: {size_diff:+,} bytes")
                
                # Save metadata
                self._save_model_metadata(filepath)
                
            else:
                raise Exception("Model file was not created successfully")
            
            ProjectLogger.log_success_header(logger, "MODEL SAVED SUCCESSFULLY")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL SAVE VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL SAVING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_model(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to load the model from
            
        Raises:
            ValueError: If file doesn't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING MODEL")
        
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                raise ValueError(f"Model file not found: {filepath}")
            
            # Get file info
            file_size = os.path.getsize(filepath)
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            logger.info(f"Loading model from: {filepath}")
            logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"Last modified: {file_modified}")
            
            # Load model
            self.model = joblib.load(filepath)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Load metadata if available
            self._load_model_metadata(filepath)
            
            ProjectLogger.log_success_header(logger, "MODEL LOADED SUCCESSFULLY")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOAD VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def _save_model_metadata(self, filepath: str) -> None:
        """Save model metadata to a companion file."""
        try:
            metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            metadata = {
                'model_name': self.model_name,
                'model_type': type(self.model).__name__,
                'model_params': self.model_params,
                'build_timestamp': self.build_timestamp,
                'save_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Could not save model metadata: {str(e)}")
    
    def _load_model_metadata(self, filepath: str) -> None:
        """Load model metadata from companion file."""
        try:
            metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                logger.info("Model metadata loaded:")
                logger.info(f"  - Build timestamp: {metadata.get('build_timestamp', 'Unknown')}")
                logger.info(f"  - Save timestamp: {metadata.get('save_timestamp', 'Unknown')}")
                
        except Exception as e:
            logger.warning(f"Could not load model metadata: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {'model_name': self.model_name, 'model': None, 'status': 'Not built'}
        
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'model_params': self.model_params,
            'build_timestamp': self.build_timestamp,
            'status': 'Built'
        }


class RandomForestModelBuilder(BaseModelBuilder):
    """
    Random Forest model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model builder.
        
        Args:
            **kwargs: Random Forest parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build Random Forest classifier.
        
        Returns:
            RandomForestClassifier: Built model
            
        Raises:
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING RANDOM FOREST MODEL")
        
        try:
            logger.info("Creating Random Forest classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = RandomForestClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("Random Forest model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Min samples split: {self.model.min_samples_split}")
            logger.info(f"  - Min samples leaf: {self.model.min_samples_leaf}")
            logger.info(f"  - Random state: {self.model.random_state}")
            
            ProjectLogger.log_success_header(logger, "RANDOM FOREST MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN RANDOM FOREST BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class XGBoostModelBuilder(BaseModelBuilder):
    """
    XGBoost model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model builder.
        
        Args:
            **kwargs: XGBoost parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build XGBoost classifier.
        
        Returns:
            XGBClassifier: Built model
            
        Raises:
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING XGBOOST MODEL")
        
        try:
            logger.info("Creating XGBoost classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = XGBClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("XGBoost model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Learning rate: {self.model.learning_rate}")
            logger.info(f"  - Random state: {self.model.random_state}")
            
            ProjectLogger.log_success_header(logger, "XGBOOST MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN XGBOOST BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class LightGBMModelBuilder(BaseModelBuilder):
    """
    LightGBM model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM model builder.
        
        Args:
            **kwargs: LightGBM parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'verbose': -1  # Suppress LightGBM warnings
        }
        default_params.update(kwargs)
        super().__init__('LightGBM', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build LightGBM classifier.
        
        Returns:
            LGBMClassifier: Built model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING LIGHTGBM MODEL")
        
        try:
            logger.info("Creating LightGBM classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = LGBMClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("LightGBM model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Learning rate: {self.model.learning_rate}")
            
            ProjectLogger.log_success_header(logger, "LIGHTGBM MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN LIGHTGBM BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
