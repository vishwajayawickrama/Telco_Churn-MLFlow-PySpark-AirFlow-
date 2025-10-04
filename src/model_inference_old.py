import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config
from logger import get_logger, ProjectLogger, log_exceptions

logger = get_logger(__name__)


class ModelInference:
    """
    Comprehensive model inference class for Telco customer churn prediction.
    
    This class handles the complete inference pipeline for predicting customer churn
    in telecommunications services, including data preprocessing, feature encoding,
    tenure binning, and model prediction.
    """

    def __init__(self, model_path: str):
        """
        Initialize model inference.
        
        Args:
            model_path (str): Path to saved model file
        """
        try:
            ProjectLogger.log_section_header(logger, "INITIALIZING MODEL INFERENCE")
            logger.info(f"Starting model inference initialization with model path: {model_path}")
            
            self.model_path = model_path
            self.encoders = {} 
            self.load_model()
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            
            logger.info("Model inference initialization completed successfully")
            ProjectLogger.log_success_header(logger, "MODEL INFERENCE INITIALIZED")
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "MODEL INFERENCE INITIALIZATION FAILED")
            logger.error(f"Failed to initialize model inference: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_model(self) -> None:
        """
        Load a trained model from file.
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING MODEL FOR INFERENCE")
        
        try:
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)

            if self.model is None or not isinstance(self.model, BaseEstimator):
                raise ValueError("Loaded object is not a valid sklearn model")

            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
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
    def load_encoders(self, encoder_dir) -> None:
        """
        Load all encoder files from the artifacts/encode directory.
        
        Args:
            encoder_dir (str): Path to the directory containing encoder files
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING ENCODERS FOR INFERENCE")
        
        try:
            logger.info(f"Starting to load encoders from directory: {encoder_dir}")
            
            # Validate encoder directory exists
            if not os.path.exists(encoder_dir):
                raise FileNotFoundError(f"Encoder directory not found: {encoder_dir}")
            
            if not os.path.isdir(encoder_dir):
                raise ValueError(f"Path is not a directory: {encoder_dir}")
            
            # Get list of encoder files
            try:
                all_files = os.listdir(encoder_dir)
                encoder_files = [f for f in all_files if f.endswith('_encoder.json')]
                logger.info(f"Found {len(encoder_files)} encoder files in directory")
                
                if not encoder_files:
                    logger.warning("No encoder files found in directory")
                    return
                    
            except OSError as e:
                logger.error(f"Error accessing directory {encoder_dir}: {str(e)}")
                raise
            
            # Load each encoder file
            loaded_count = 0
            failed_count = 0
            
            for file in encoder_files:
                try:
                    logger.debug(f"Processing encoder file: {file}")
                    
                    # Extract feature name from filename
                    if not file.endswith('_encoder.json'):
                        logger.warning(f"Skipping non-encoder file: {file}")
                        continue
                    
                    feature_name = file.split('_encoder.json')[0]
                    file_path = os.path.join(encoder_dir, file)
                    
                    # Load encoder data
                    with open(file_path, 'r') as f:
                        encoder_data = json.load(f)
                    
                    # Validate encoder data
                    if not isinstance(encoder_data, dict):
                        logger.error(f"Invalid encoder format in {file}: expected dict, got {type(encoder_data)}")
                        failed_count += 1
                        continue
                    
                    # Store encoder
                    self.encoders[feature_name] = encoder_data
                    loaded_count += 1
                    logger.debug(f"Successfully loaded encoder for feature: {feature_name}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON format in {file}: {str(e)}")
                    failed_count += 1
                    continue
                    
                except IOError as e:
                    logger.error(f"Error reading file {file}: {str(e)}")
                    failed_count += 1
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error loading {file}: {str(e)}")
                    failed_count += 1
                    continue
            
            # Log summary
            total_files = len(encoder_files)
            logger.info(f"Encoder loading summary: {loaded_count} successful, {failed_count} failed out of {total_files} files")
            
            if loaded_count == 0:
                logger.warning("No encoders were successfully loaded")
            else:
                loaded_features = list(self.encoders.keys())
                logger.info(f"Successfully loaded encoders for features: {loaded_features}")
            
            ProjectLogger.log_success_header(logger, "ENCODERS LOADING COMPLETED")
            
        except FileNotFoundError as e:
            ProjectLogger.log_error_header(logger, "ENCODER DIRECTORY NOT FOUND")
            logger.error(f"Directory error: {str(e)}")
            raise
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "ENCODER DIRECTORY VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "ENCODER LOADING FAILED")
            logger.error(f"Unexpected error loading encoders: {str(e)}")
            raise

    
    @log_exceptions(logger)
    def preprocess_input(self, data):
        """
        Preprocess input data for Telco customer churn prediction.
        
        This method handles the complete preprocessing pipeline including:
        - Label encoding for categorical features (gender, InternetService, etc.)
        - Tenure binning (New, Established, Loyal customers)
        - Ordinal encoding for Contract types
        - Removal of identifier columns
        
        Args:
            data: Input customer data to preprocess (dict or DataFrame)
            
        Returns:
            pd.DataFrame: Preprocessed data ready for churn prediction
        """
        ProjectLogger.log_step_header(logger, "STEP", "PREPROCESSING INPUT DATA")
        
        try:
            logger.info("Starting input data preprocessing")
            
            # Convert input to DataFrame
            if not isinstance(data, pd.DataFrame):
                logger.info("Converting input data to DataFrame")
                data = pd.DataFrame([data])
            else:
                logger.info(f"Input data shape: {data.shape}")
            
            original_columns = data.columns.tolist()
            logger.debug(f"Original columns: {original_columns}")
            
            # Apply encoders
            if self.encoders:
                logger.info(f"Applying {len(self.encoders)} encoders to data")
                encoded_columns = []
                
                for col, encoder in self.encoders.items():
                    if col in data.columns:
                        try:
                            logger.debug(f"Encoding column: {col}")
                            data[col] = data[col].map(encoder)
                            encoded_columns.append(col)
                        except Exception as e:
                            logger.error(f"Failed to encode column {col}: {str(e)}")
                            continue
                
                logger.info(f"Successfully encoded {len(encoded_columns)} columns: {encoded_columns}")
            else:
                logger.info("No encoders available - proceeding with original data")

            # Apply binning
            try:
                logger.info("Applying tenure binning")
                binning = CustomBinningStrategy(self.binning_config['tenure_bins'])
                data = binning.bin_feature(data, 'tenure')
                logger.debug("Tenure binning completed successfully")
            except KeyError as e:
                logger.error(f"tenure column not found for binning: {str(e)}")
            except Exception as e:
                logger.error(f"Error during binning: {str(e)}")
                raise

            # Apply ordinal encoding
            try:
                logger.info("Applying ordinal encoding")
                encoder = OrdinalEncodingStrategy(self.encoding_config['ordinal_mappings'])
                data = encoder.encode(data)
                logger.debug("Ordinal encoding completed successfully")
            except Exception as e:
                logger.error(f"Error during ordinal encoding: {str(e)}")
                raise

            # Remove unnecessary columns
            columns_to_drop = ['customerID']  # Remove customer identifier column
            existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
            
            if existing_columns_to_drop:
                logger.info(f"Dropping columns: {existing_columns_to_drop}")
                data = data.drop(columns=existing_columns_to_drop, errors='ignore')
            else:
                logger.debug("No unnecessary columns found to drop")
            
            final_columns = data.columns.tolist()
            logger.info(f"Preprocessing completed. Final shape: {data.shape}")
            logger.debug(f"Final columns: {final_columns}")
            
            ProjectLogger.log_success_header(logger, "INPUT DATA PREPROCESSING COMPLETED")
            return data
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "INPUT DATA PREPROCESSING FAILED")
            logger.error(f"Error in preprocessing input data: {str(e)}")
            raise
    
    @log_exceptions(logger)
    def predict(self, data):
        """
        Make Telco customer churn predictions on input customer data.
        
        Args:
            data: Input customer data (dict or DataFrame) containing Telco customer features
                 like gender, tenure, InternetService, Contract, etc.
            
        Returns:
            dict: Churn prediction results with:
                - Status: 'Churn' or 'Retained' 
                - Confidence: Prediction confidence percentage
        """
        ProjectLogger.log_step_header(logger, "STEP", "MAKING PREDICTION")
        
        try:
            logger.info("Starting prediction process")
            
            # Preprocess the input data
            logger.info("Preprocessing input data for prediction")
            pp_data = self.preprocess_input(data)
            logger.debug(f"Preprocessed data shape: {pp_data.shape}")
            
            # Validate preprocessed data
            if pp_data.empty:
                raise ValueError("Preprocessed data is empty")
            
            if pp_data.isnull().any().any():
                null_columns = pp_data.columns[pp_data.isnull().any()].tolist()
                logger.warning(f"Null values found in columns: {null_columns}")
            
            # Make prediction
            logger.info("Generating prediction")
            y_pred = self.model.predict(pp_data)
            logger.debug(f"Raw prediction: {y_pred}")
            
            # Get prediction probabilities
            logger.info("Generating prediction probabilities")
            y_prob_array = self.model.predict_proba(pp_data)
            logger.debug(f"Prediction probabilities shape: {y_prob_array.shape}")
            
            if y_prob_array.shape[1] < 2:
                raise ValueError("Model does not provide probabilities for both classes")
            
            y_prob = float(y_prob_array[0][1])
            logger.debug(f"Raw probability for positive class: {y_prob}")
            
            # Convert prediction to readable format
            y_pred_label = 'Churn' if y_pred[0] == 1 else 'Retained'
            y_prob_percentage = round(y_prob * 100)
            
            logger.info(f"Prediction: {y_pred_label} with {y_prob_percentage}% confidence")
            
            result = {
                "Status": y_pred_label,
                "Confidence": f"{y_prob_percentage} %"
            }
            
            logger.info("Prediction completed successfully")
            ProjectLogger.log_success_header(logger, "PREDICTION COMPLETED")
            
            return result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "PREDICTION VALIDATION ERROR")
            logger.error(f"Validation error during prediction: {str(e)}")
            raise
            
        except AttributeError as e:
            ProjectLogger.log_error_header(logger, "MODEL ATTRIBUTE ERROR")
            logger.error(f"Model method not available: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "PREDICTION FAILED")
            logger.error(f"Unexpected error during prediction: {str(e)}")
            raise
