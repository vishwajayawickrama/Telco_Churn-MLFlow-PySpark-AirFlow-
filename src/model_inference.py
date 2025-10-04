"""
Model inference module for PySpark ML operations.
Provides comprehensive inference functionality for customer churn prediction.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

from utils.logger import ProjectLogger, log_exceptions
from utils.spark_utils import get_spark_session
from utils.config import get_binning_config, get_encoding_config

# Configure logger
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Comprehensive model inference class for PySpark ML Telco customer churn prediction.
    
    This class handles the complete inference pipeline for predicting customer churn
    using PySpark ML models, including data preprocessing, feature encoding,
    tenure binning, and model prediction.
    """

    def __init__(self, model_path: str, spark: Optional[SparkSession] = None):
        """
        Initialize model inference.
        
        Args:
            model_path (str): Path to saved PySpark ML Pipeline model
            spark: Optional SparkSession
        """
        try:
            ProjectLogger.log_section_header(logger, "INITIALIZING PYSPARK MODEL INFERENCE")
            logger.info(f"Starting model inference initialization with model path: {model_path}")
            
            self.spark = spark or get_spark_session()
            self.model_path = model_path
            self.model = None
            self.encoders = {}
            
            self.load_model()
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            
            logger.info("PySpark model inference initialization completed successfully")
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL INFERENCE INITIALIZED")
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "MODEL INFERENCE INITIALIZATION FAILED")
            logger.error(f"Failed to initialize model inference: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_model(self) -> None:
        """
        Load a trained PySpark ML Pipeline model from file.
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING PYSPARK MODEL FOR INFERENCE")
        
        try:
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model directory not found: {self.model_path}")

            logger.info(f"Loading PySpark ML Pipeline model from: {self.model_path}")
            self.model = PipelineModel.load(self.model_path)

            if self.model is None or not isinstance(self.model, PipelineModel):
                raise ValueError("Loaded object is not a valid PySpark ML PipelineModel")

            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            logger.info(f"Pipeline stages: {len(self.model.stages)}")
            
            # Log stage information
            for i, stage in enumerate(self.model.stages):
                logger.debug(f"  Stage {i+1}: {type(stage).__name__}")
            
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL LOADED FOR INFERENCE")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOADING VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_encoders(self, encoder_dir: str) -> None:
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
            ProjectLogger.log_error_header(logger, "ENCODER VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN ENCODER LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def preprocess_input(self, data: Union[Dict, List[Dict], DataFrame]) -> DataFrame:
        """
        Preprocess input data for inference using PySpark operations.
        
        Args:
            data: Input data (dict, list of dicts, or DataFrame)
            
        Returns:
            DataFrame: Preprocessed PySpark DataFrame ready for inference
        """
        ProjectLogger.log_step_header(logger, "STEP", "PREPROCESSING INPUT DATA FOR INFERENCE")
        
        try:
            logger.info("Starting input data preprocessing")
            
            # Convert input to DataFrame
            if isinstance(data, dict):
                logger.info("Converting dict input to DataFrame")
                data_df = self.spark.createDataFrame([data])
            elif isinstance(data, list):
                logger.info(f"Converting list of {len(data)} records to DataFrame")
                data_df = self.spark.createDataFrame(data)
            elif isinstance(data, DataFrame):
                logger.info("Input is already a DataFrame")
                data_df = data
            else:
                raise ValueError(f"Unsupported input type: {type(data)}")
            
            row_count = data_df.count()
            logger.info(f"Input data shape: {row_count} rows, {len(data_df.columns)} columns")
            logger.debug(f"Input columns: {data_df.columns}")
            
            # Handle missing values in TotalCharges (convert empty strings to 0)
            if "TotalCharges" in data_df.columns:
                logger.info("Handling TotalCharges column")
                data_df = data_df.withColumn(
                    "TotalCharges",
                    F.when(F.col("TotalCharges") == "", "0.0")
                    .otherwise(F.col("TotalCharges"))
                    .cast(DoubleType())
                )
                logger.debug("TotalCharges column processed")
            
            # Apply binning (specifically for tenure)
            if "tenure" in data_df.columns and self.binning_config:
                logger.info("Applying tenure binning")
                try:
                    # Apply custom binning for tenure using PySpark
                    tenure_bins = self.binning_config.get('tenure_bins', [])
                    if tenure_bins:
                        # Create binning conditions
                        conditions = []
                        labels = []
                        
                        # Assuming tenure_bins is a list of (min, max, label) tuples
                        for i, (min_val, max_val, label) in enumerate(tenure_bins):
                            if i == 0:
                                condition = F.when(
                                    (F.col("tenure") >= min_val) & (F.col("tenure") <= max_val), 
                                    label
                                )
                            else:
                                condition = condition.when(
                                    (F.col("tenure") >= min_val) & (F.col("tenure") <= max_val), 
                                    label
                                )
                        
                        # Apply binning
                        data_df = data_df.withColumn("tenure_bin", condition.otherwise("Unknown"))
                        logger.debug("Tenure binning applied successfully")
                    
                except Exception as e:
                    logger.error(f"Error during binning: {str(e)}")
                    raise
            
            # Apply ordinal encoding using the loaded encoders
            if self.encoders:
                logger.info("Applying ordinal encoding using loaded encoders")
                try:
                    for feature_name, encoder_mapping in self.encoders.items():
                        if feature_name in data_df.columns:
                            logger.debug(f"Encoding feature: {feature_name}")
                            
                            # Create mapping expression
                            mapping_expr = None
                            for original_value, encoded_value in encoder_mapping.items():
                                if mapping_expr is None:
                                    mapping_expr = F.when(F.col(feature_name) == original_value, encoded_value)
                                else:
                                    mapping_expr = mapping_expr.when(F.col(feature_name) == original_value, encoded_value)
                            
                            # Apply encoding with default value for unmapped entries
                            if mapping_expr is not None:
                                data_df = data_df.withColumn(
                                    feature_name,
                                    mapping_expr.otherwise(0)  # Default value for unmapped categories
                                )
                                
                except Exception as e:
                    logger.error(f"Error during ordinal encoding: {str(e)}")
                    raise
            
            # Remove unnecessary columns
            columns_to_drop = ['customerID']  # Remove customer identifier column
            existing_columns_to_drop = [col for col in columns_to_drop if col in data_df.columns]
            
            if existing_columns_to_drop:
                logger.info(f"Dropping columns: {existing_columns_to_drop}")
                data_df = data_df.drop(*existing_columns_to_drop)
            else:
                logger.debug("No unnecessary columns found to drop")
            
            final_columns = data_df.columns
            final_count = data_df.count()
            logger.info(f"Preprocessing completed. Final shape: {final_count} rows, {len(final_columns)} columns")
            logger.debug(f"Final columns: {final_columns}")
            
            ProjectLogger.log_success_header(logger, "INPUT DATA PREPROCESSING COMPLETED")
            return data_df
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "INPUT DATA PREPROCESSING FAILED")
            logger.error(f"Error in preprocessing input data: {str(e)}")
            raise

    @log_exceptions(logger)
    def predict(self, data: Union[Dict, List[Dict], DataFrame]) -> Dict[str, Any]:
        """
        Make Telco customer churn predictions on input customer data using PySpark ML.
        
        Args:
            data: Input customer data (dict, list of dicts, or DataFrame) containing Telco customer features
                 like gender, tenure, InternetService, Contract, etc.
            
        Returns:
            dict: Churn prediction results with:
                - Status: 'Churn' or 'Retained' 
                - Confidence: Prediction confidence percentage
                - Probabilities: Array of class probabilities
        """
        ProjectLogger.log_step_header(logger, "STEP", "MAKING PYSPARK PREDICTION")
        
        try:
            logger.info("Starting PySpark prediction process")
            
            # Preprocess the input data
            logger.info("Preprocessing input data for prediction")
            preprocessed_df = self.preprocess_input(data)
            
            input_count = preprocessed_df.count()
            logger.debug(f"Preprocessed data shape: {input_count} rows")
            
            # Validate preprocessed data
            if input_count == 0:
                raise ValueError("Preprocessed data is empty")
            
            # Make prediction using the PySpark ML Pipeline
            logger.info("Generating prediction using PySpark ML Pipeline")
            predictions_df = self.model.transform(preprocessed_df)
            
            # Cache predictions for performance
            predictions_df.cache()
            
            # Collect predictions
            logger.info("Collecting prediction results")
            predictions = predictions_df.collect()
            
            if not predictions:
                raise ValueError("No predictions generated")
            
            # Extract results from first row (assuming single prediction)
            first_prediction = predictions[0]
            
            # Get prediction and probability
            y_pred = first_prediction['prediction']
            
            # Extract probability from probability vector
            if 'probability' in predictions_df.columns:
                # Probability is a DenseVector, extract values
                prob_vector = first_prediction['probability']
                y_prob_array = prob_vector.toArray()
                y_prob = float(y_prob_array[1])  # Probability of positive class (churn)
            else:
                logger.warning("Probability column not found, using raw prediction confidence")
                y_prob = 0.5  # Default probability
            
            logger.debug(f"Raw prediction: {y_pred}")
            logger.debug(f"Raw probability for positive class: {y_prob}")
            
            # Convert prediction to readable format
            y_pred_label = 'Churn' if y_pred == 1.0 else 'Retained'
            y_prob_percentage = round(y_prob * 100)
            
            logger.info(f"Prediction: {y_pred_label} with {y_prob_percentage}% confidence")
            
            # Prepare result
            result = {
                "Status": y_pred_label,
                "Confidence": f"{y_prob_percentage}%",
                "Probabilities": {
                    "Retained": round((1 - y_prob) * 100, 2),
                    "Churn": round(y_prob * 100, 2)
                },
                "Raw_Prediction": float(y_pred),
                "Prediction_Timestamp": datetime.now().isoformat()
            }
            
            # Clean up cache
            predictions_df.unpersist()
            
            logger.info("PySpark prediction completed successfully")
            ProjectLogger.log_success_header(logger, "PYSPARK PREDICTION COMPLETED")
            
            return result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "PREDICTION VALIDATION ERROR")
            logger.error(f"Validation error during prediction: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "PREDICTION FAILED")
            logger.error(f"Unexpected error during prediction: {str(e)}")
            raise

    @log_exceptions(logger)
    def predict_batch(self, data_df: DataFrame) -> DataFrame:
        """
        Make batch predictions on a large DataFrame.
        
        Args:
            data_df (DataFrame): Input DataFrame with multiple customer records
            
        Returns:
            DataFrame: DataFrame with predictions and probabilities
        """
        ProjectLogger.log_step_header(logger, "STEP", "MAKING BATCH PREDICTIONS")
        
        try:
            logger.info("Starting batch prediction process")
            
            input_count = data_df.count()
            logger.info(f"Input batch size: {input_count} records")
            
            if input_count == 0:
                raise ValueError("Input DataFrame is empty")
            
            # Preprocess the input data
            logger.info("Preprocessing batch data for prediction")
            preprocessed_df = self.preprocess_input(data_df)
            
            # Make predictions
            logger.info("Generating batch predictions using PySpark ML Pipeline")
            predictions_df = self.model.transform(preprocessed_df)
            
            # Add human-readable prediction labels
            predictions_df = predictions_df.withColumn(
                "prediction_label",
                F.when(F.col("prediction") == 1.0, "Churn").otherwise("Retained")
            )
            
            # Extract probability for positive class (churn)
            if 'probability' in predictions_df.columns:
                # Extract churn probability from probability vector
                from pyspark.ml.linalg import VectorUDT
                from pyspark.sql.functions import udf
                
                def extract_prob(prob_vector):
                    return float(prob_vector[1])
                
                extract_prob_udf = udf(extract_prob, DoubleType())
                
                predictions_df = predictions_df.withColumn(
                    "churn_probability",
                    extract_prob_udf(F.col("probability"))
                )
                
                predictions_df = predictions_df.withColumn(
                    "confidence_percentage",
                    F.round(F.col("churn_probability") * 100, 2)
                )
            
            # Add timestamp
            predictions_df = predictions_df.withColumn(
                "prediction_timestamp",
                F.current_timestamp()
            )
            
            final_count = predictions_df.count()
            logger.info(f"Batch prediction completed for {final_count} records")
            
            ProjectLogger.log_success_header(logger, "BATCH PREDICTION COMPLETED")
            
            return predictions_df
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "BATCH PREDICTION FAILED")
            logger.error(f"Error during batch prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information including stages and metadata
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_type": type(self.model).__name__,
            "pipeline_stages": len(self.model.stages),
            "stage_types": [type(stage).__name__ for stage in self.model.stages],
            "model_path": self.model_path,
            "encoders_loaded": len(self.encoders),
            "encoder_features": list(self.encoders.keys())
        }


def create_model_inference(model_path: str, spark: Optional[SparkSession] = None) -> ModelInference:
    """
    Factory function to create a ModelInference instance.
    
    Args:
        model_path (str): Path to saved PySpark ML Pipeline model
        spark: Optional SparkSession
        
    Returns:
        ModelInference: Configured model inference instance
    """
    return ModelInference(model_path, spark)