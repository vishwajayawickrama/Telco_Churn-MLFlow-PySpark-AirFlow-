import os
import sys
import json
import joblib
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from sklearn.base import BaseEstimator

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class ModelInference:
    """
    Comprehensive model inference class for making predictions with trained models.
    """
    
    def __init__(self, model_path: Optional[str] = None, model: Optional[BaseEstimator] = None):
        """
        Initialize model inference.
        
        Args:
            model_path (Optional[str]): Path to saved model file
            model (Optional[BaseEstimator]): Pre-loaded model object
        """
        self.model = None
        self.model_path = model_path
        self.model_metadata = {}
        self.feature_columns = None
        self.encoders = {}
        self.scaler = None
        self.inference_history = []
        
        ProjectLogger.log_section_header(logger, "INITIALIZING MODEL INFERENCE")
        
        if model is not None:
            self.model = model
            logger.info(f"Model provided directly: {type(model).__name__}")
        elif model_path is not None:
            self.load_model(model_path)
        else:
            logger.info("No model provided. Use load_model() to load a model.")

    @log_exceptions(logger)
    def load_model(self, model_path: str, method: str = 'auto') -> None:
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the model file
            method (str): Loading method ('auto', 'joblib', 'pickle')
            
        Raises:
            ValueError: If file doesn't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING MODEL FOR INFERENCE")
        
        try:
            # Validate file exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            self.model_path = model_path
            
            # Get file info
            file_size = os.path.getsize(model_path)
            file_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"Last modified: {file_modified}")
            
            # Determine loading method
            if method == 'auto':
                if model_path.endswith('.pkl'):
                    method = 'pickle'
                else:
                    method = 'joblib'
            
            logger.info(f"Loading method: {method}")
            
            # Load model
            if method == 'joblib':
                self.model = joblib.load(model_path)
            else:  # pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Load metadata if available
            self._load_model_metadata()
            
            # Load encoders if available
            self._load_encoders()
            
            ProjectLogger.log_success_header(logger, "MODEL LOADED FOR INFERENCE")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOADING VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def predict(
        self, 
        data: Union[pd.DataFrame, np.ndarray, dict], 
        return_probabilities: bool = False,
        preprocess: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            return_probabilities (bool): Whether to return prediction probabilities
            preprocess (bool): Whether to apply preprocessing
            
        Returns:
            Union[np.ndarray, Tuple]: Predictions and optionally probabilities
            
        Raises:
            ValueError: If model is not loaded or data is invalid
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "MAKING MODEL PREDICTIONS")
        
        try:
            # Validate model is loaded
            if self.model is None:
                raise ValueError("No model loaded. Use load_model() or provide model in constructor.")
            
            # Convert data to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
                logger.info("Converted dictionary input to DataFrame")
            elif isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
                logger.info("Converted numpy array to DataFrame")
            elif not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be DataFrame, numpy array, or dictionary")
            
            logger.info(f"Input data shape: {data.shape}")
            logger.info(f"Input data type: {type(data).__name__}")
            
            # Store original data info
            inference_start = datetime.now()
            original_shape = data.shape
            
            # Preprocess data if requested
            if preprocess:
                logger.info("Applying preprocessing...")
                data = self._preprocess_data(data)
                logger.info(f"Preprocessed data shape: {data.shape}")
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Found {missing_values} missing values in input data")
                logger.warning("Missing values may affect prediction quality")
            
            # Make predictions
            logger.info("Generating predictions...")
            predictions = self.model.predict(data)
            
            logger.info(f"Predictions generated: {len(predictions)} samples")
            logger.info(f"Prediction shape: {predictions.shape}")
            
            # Log prediction distribution
            unique_predictions, counts = np.unique(predictions, return_counts=True)
            logger.info("Prediction distribution:")
            for pred, count in zip(unique_predictions, counts):
                percentage = (count / len(predictions)) * 100
                logger.info(f"  - {pred}: {count} samples ({percentage:.2f}%)")
            
            result = predictions
            
            # Get probabilities if requested and available
            probabilities = None
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                logger.info("Generating prediction probabilities...")
                probabilities = self.model.predict_proba(data)
                logger.info(f"Probabilities shape: {probabilities.shape}")
                
                # Log probability statistics
                if probabilities.shape[1] == 2:  # Binary classification
                    avg_confidence = np.mean(np.max(probabilities, axis=1))
                    logger.info(f"Average prediction confidence: {avg_confidence:.4f}")
                
                result = (predictions, probabilities)
            
            # Store inference history
            inference_record = {
                'timestamp': inference_start.isoformat(),
                'input_shape': original_shape,
                'output_shape': predictions.shape,
                'model_type': type(self.model).__name__,
                'preprocessing_applied': preprocess,
                'probabilities_returned': return_probabilities,
                'missing_values': missing_values
            }
            self.inference_history.append(inference_record)
            
            inference_end = datetime.now()
            inference_duration = (inference_end - inference_start).total_seconds()
            logger.info(f"Inference completed in {inference_duration:.3f} seconds")
            
            ProjectLogger.log_success_header(logger, "PREDICTIONS GENERATED SUCCESSFULLY")
            
            return result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "PREDICTION VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN PREDICTION")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def predict_single(
        self, 
        sample: Union[dict, pd.Series, np.ndarray],
        return_probabilities: bool = False,
        preprocess: bool = True
    ) -> Union[Any, Tuple[Any, np.ndarray]]:
        """
        Make prediction on a single sample.
        
        Args:
            sample: Single sample for prediction
            return_probabilities (bool): Whether to return prediction probabilities
            preprocess (bool): Whether to apply preprocessing
            
        Returns:
            Union[Any, Tuple]: Single prediction and optionally probabilities
        """
        ProjectLogger.log_step_header(logger, "STEP", "MAKING SINGLE SAMPLE PREDICTION")
        
        try:
            # Convert to DataFrame
            if isinstance(sample, dict):
                data = pd.DataFrame([sample])
            elif isinstance(sample, pd.Series):
                data = pd.DataFrame([sample])
            elif isinstance(sample, np.ndarray):
                data = pd.DataFrame([sample])
            else:
                raise ValueError("Sample must be dict, Series, or numpy array")
            
            logger.info("Processing single sample prediction")
            
            # Make prediction
            result = self.predict(data, return_probabilities=return_probabilities, preprocess=preprocess)
            
            # Extract single result
            if return_probabilities:
                predictions, probabilities = result
                single_prediction = predictions[0]
                single_probabilities = probabilities[0]
                
                logger.info(f"Single prediction: {single_prediction}")
                if probabilities.shape[1] == 2:  # Binary classification
                    confidence = np.max(single_probabilities)
                    logger.info(f"Prediction confidence: {confidence:.4f}")
                
                return single_prediction, single_probabilities
            else:
                single_prediction = result[0]
                logger.info(f"Single prediction: {single_prediction}")
                return single_prediction
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "ERROR IN SINGLE SAMPLE PREDICTION")
            logger.error(f"Error: {str(e)}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Applying data preprocessing...")
        
        # This is a placeholder for preprocessing logic
        # In a real implementation, you would:
        # 1. Apply the same encoders used during training
        # 2. Apply the same scaler used during training
        # 3. Handle missing values consistently
        # 4. Ensure feature order matches training data
        
        processed_data = data.copy()
        
        # Apply encoders if available
        if self.encoders:
            logger.info(f"Applying {len(self.encoders)} encoders...")
            for column, encoder_path in self.encoders.items():
                if column in processed_data.columns:
                    try:
                        with open(encoder_path, 'r') as f:
                            encoder_mapping = json.load(f)
                        processed_data[column] = processed_data[column].map(encoder_mapping)
                        logger.debug(f"Applied encoder to column: {column}")
                    except Exception as e:
                        logger.warning(f"Could not apply encoder to {column}: {str(e)}")
        
        # Apply scaler if available
        if self.scaler is not None:
            logger.info("Applying scaling transformation...")
            # This would apply the same scaler used during training
            # Implementation depends on how the scaler was saved
        
        logger.info("Preprocessing completed")
        return processed_data

    def _load_model_metadata(self) -> None:
        """Load model metadata if available."""
        try:
            metadata_path = self.model_path.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                logger.info("Model metadata loaded:")
                logger.info(f"  - Build timestamp: {self.model_metadata.get('build_timestamp', 'Unknown')}")
                logger.info(f"  - Model parameters: {len(self.model_metadata.get('model_params', {}))}")
                
        except Exception as e:
            logger.warning(f"Could not load model metadata: {str(e)}")

    def _load_encoders(self) -> None:
        """Load feature encoders if available."""
        try:
            artifacts_dir = os.path.join(os.path.dirname(self.model_path), '..', 'artifacts', 'encode')
            
            if os.path.exists(artifacts_dir):
                encoder_files = [f for f in os.listdir(artifacts_dir) if f.endswith('_encoder.json')]
                
                for encoder_file in encoder_files:
                    column_name = encoder_file.replace('_encoder.json', '')
                    encoder_path = os.path.join(artifacts_dir, encoder_file)
                    self.encoders[column_name] = encoder_path
                
                if self.encoders:
                    logger.info(f"Found {len(self.encoders)} encoders: {list(self.encoders.keys())}")
                
        except Exception as e:
            logger.warning(f"Could not load encoders: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = {
            'model_type': type(self.model).__name__,
            'model_path': self.model_path,
            'metadata': self.model_metadata,
            'available_encoders': list(self.encoders.keys()),
            'inference_count': len(self.inference_history),
            'status': 'Model loaded and ready'
        }
        
        return info

    def get_inference_history(self) -> List[Dict[str, Any]]:
        """
        Get the inference history.
        
        Returns:
            List: List of inference records
        """
        return self.inference_history

    def clear_inference_history(self) -> None:
        """Clear the inference history."""
        self.inference_history = []
        logger.info("Inference history cleared")