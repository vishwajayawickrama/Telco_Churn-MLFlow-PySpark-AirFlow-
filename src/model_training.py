import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Any, Optional, Dict
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, validation_curve

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class ModelTrainer:
    """
    Comprehensive model training class with logging and validation.
    """
    
    def __init__(self):
        """Initialize model trainer."""
        ProjectLogger.log_section_header(logger, "INITIALIZING MODEL TRAINER")
        logger.info("Model trainer ready for training operations")

    @log_exceptions(logger)
    def train(
        self, 
        model,
        X_train,
        Y_train,
    ):
        """
        Train a model with simple training approach.
        
        Args:
            model: Scikit-learn compatible model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            Tuple: (trained_model, train_score)
        """
        ProjectLogger.log_step_header(logger, "STEP", "TRAINING MODEL WITH SIMPLE APPROACH")
        
        try:
            # Validate inputs
            if X_train.empty:
                raise ValueError("Training features (X_train) is empty")
            
            if Y_train.empty:
                raise ValueError("Training targets (Y_train) is empty")
            
            if len(X_train) != len(Y_train):
                raise ValueError(f"Feature and target lengths don't match: {len(X_train)} vs {len(Y_train)}")
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Target data shape: {Y_train.shape}")
            logger.info(f"Model type: {type(model).__name__}")
            
            
            # Start training
            training_start = datetime.now()
            logger.info(f"Starting model training at: {training_start}")
            
            # Fit the model
            logger.info("Fitting model to training data...")
            model.fit(X_train, Y_train)
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Calculate training score
            logger.info("Calculating training score...")
            train_score = model.score(X_train, Y_train)
            
            ProjectLogger.log_success_header(logger, "MODEL TRAINING COMPLETED")
            
            return model, train_score
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL TRAINING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    

    @log_exceptions(logger)
    def save_model(self, model: BaseEstimator, filepath: str, method: str = 'joblib') -> None:
        """
        Save a trained model to file.
        
        Args:
            model: Trained model to save
            filepath (str): Path to save the model
            method (str): Serialization method ('joblib' or 'pickle')
        """
        ProjectLogger.log_step_header(logger, "STEP", "SAVING TRAINED MODEL")
        
        try:
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            if not filepath:
                raise ValueError("Filepath cannot be empty")
            
            if method not in ['joblib', 'pickle']:
                raise ValueError("Method must be 'joblib' or 'pickle'")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            logger.info(f"Directory created/verified: {os.path.dirname(filepath)}")
            
            logger.info(f"Saving model using {method} method")
            logger.info(f"Model type: {type(model).__name__}")
            logger.info(f"Save path: {filepath}")
            
            # Save based on method
            if method == 'joblib':
                joblib.dump(model, filepath)
            else:  # pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            # Verify save
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Model saved successfully")
                logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            else:
                raise Exception("Model file was not created")
            
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
    def load_model(self, filepath: str, method: str = 'auto') -> BaseEstimator:
        """
        Load a trained model from file.
        
        Args:
            filepath (str): Path to load the model from
            method (str): Loading method ('joblib', 'pickle', or 'auto')
            
        Returns:
            BaseEstimator: Loaded model
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING TRAINED MODEL")
        
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
            
            # Determine method
            if method == 'auto':
                if filepath.endswith('.pkl'):
                    method = 'pickle'
                else:
                    method = 'joblib'
            
            logger.info(f"Loading method: {method}")
            
            # Load model
            if method == 'joblib':
                model = joblib.load(filepath)
            else:  # pickle
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(model).__name__}")
            
            ProjectLogger.log_success_header(logger, "MODEL LOADED SUCCESSFULLY")
            
            return model
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOAD VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise




# @log_exceptions(logger)
#     def train_with_cross_validation(
#         self,
#         model: BaseEstimator,
#         X_train: pd.DataFrame,
#         y_train: pd.Series,
#         cv_folds: int = 5,
#         scoring: str = 'accuracy'
#     ) -> Tuple[BaseEstimator, Dict[str, float]]:
#         """
#         Train a model with cross-validation.
        
#         Args:
#             model: Scikit-learn compatible model
#             X_train (pd.DataFrame): Training features
#             y_train (pd.Series): Training targets
#             cv_folds (int): Number of cross-validation folds
#             scoring (str): Scoring metric for cross-validation
            
#         Returns:
#             Tuple: (trained_model, cv_results)
#         """
#         ProjectLogger.log_step_header(logger, "STEP", "TRAINING MODEL WITH CROSS-VALIDATION")
        
#         try:
#             logger.info(f"Cross-validation folds: {cv_folds}")
#             logger.info(f"Scoring metric: {scoring}")
            
#             # Perform cross-validation before training
#             logger.info("Performing cross-validation...")
#             cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
            
#             cv_results = {
#                 'mean_score': cv_scores.mean(),
#                 'std_score': cv_scores.std(),
#                 'min_score': cv_scores.min(),
#                 'max_score': cv_scores.max(),
#                 'individual_scores': cv_scores.tolist()
#             }
            
#             logger.info("Cross-validation results:")
#             logger.info(f"  - Mean {scoring}: {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f})")
#             logger.info(f"  - Min {scoring}: {cv_results['min_score']:.4f}")
#             logger.info(f"  - Max {scoring}: {cv_results['max_score']:.4f}")
            
#             # Train the final model on all data
#             logger.info("Training final model on complete dataset...")
#             final_model, train_score = self.train_simple(model, X_train, y_train)
            
#             cv_results['final_train_score'] = train_score
            
#             ProjectLogger.log_success_header(logger, "CROSS-VALIDATION TRAINING COMPLETED")
            
#             return final_model, cv_results
            
#         except Exception as e:
#             ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN CV TRAINING")
#             logger.error(f"Unexpected error: {str(e)}")
#             raise