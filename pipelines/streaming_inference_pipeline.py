"""
Unified Streaming Inference Pipeline for Telco Customer Churn Prediction.

This module provides a unified interface for both PySpark Structured Streaming and traditional
batch inference implementations. It supports real-time customer churn predictions with automatic
preprocessing and comprehensive monitoring capabilities.

Key Features:
- Dual Implementation Support - Choose between PySpark Streaming and pandas batch
- Real-time Predictions - Process individual customer records instantly
- Batch Processing - Efficient processing of multiple customer records
- Automatic Preprocessing - Applies same transformations as training pipeline
- MLflow Integration - Prediction tracking and model monitoring
- Production Ready - Robust error handling and performance monitoring

Inference Modes:
1. PySpark Streaming (Recommended for production):
   - Real-time streaming inference
   - High-throughput batch processing
   - Automatic scaling and fault tolerance
   - Structured streaming capabilities
   - Integration with Kafka, Kinesis, etc.

2. Pandas Batch Processing (For development and small-scale deployment):
   - Fast single-record predictions
   - Simple integration with web APIs
   - Direct model loading and inference
   - Memory-efficient for small batches

Prediction Capabilities:
- Single Customer Prediction - Individual churn probability scores
- Batch Customer Prediction - Process multiple customers efficiently
- Stream Processing - Continuous real-time data processing
- API Integration - REST API compatible prediction interface
- Monitoring - Track prediction metrics and model performance

Input Data Support:
- JSON format for single customer records
- CSV files for batch processing
- Streaming data from message queues
- Database query results
- Real-time API requests

Usage:
    >>> # Real-time single customer prediction
    >>> customer_data = {
    ...     'gender': 'Female',
    ...     'SeniorCitizen': 0,
    ...     'Partner': 'Yes',
    ...     'tenure': 12,
    ...     'MonthlyCharges': 65.0
    ... }
    >>> result = streaming_inference(
    ...     use_pyspark=True,
    ...     input_data=customer_data
    ... )
    >>> 
    >>> # Batch processing with pandas
    >>> result = streaming_inference(
    ...     use_pyspark=False,
    ...     inference=model_instance,
    ...     data=customer_dataframe
    ... )

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

import os
import sys
import time
import mlflow
import pandas as pd
from typing import Dict, Any, Optional

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import pipeline implementations
from streaming_inference_pipeline_pyspark import streaming_inference_pyspark
from model_inference import ModelInference
from logger import get_logger, ProjectLogger, log_exceptions
from mlflow_utils import MLflowTracker, create_mlflow_run_tags

# Initialize logger
logger = get_logger(__name__)

# Initialize model inference with error handling
try:
    ProjectLogger.log_section_header(logger, "INITIALIZING STREAMING INFERENCE PIPELINE")
    logger.info("Starting model inference initialization for streaming pipeline")
    
    model_path = 'artifacts/models/telco_customer_churn_prediction.joblib'
    logger.info(f"Loading model from: {model_path}")
    
    inference = ModelInference(model_path)
    logger.info("Model inference instance created successfully")
    
    ProjectLogger.log_success_header(logger, "STREAMING INFERENCE PIPELINE INITIALIZED")
    
except Exception as e:
    ProjectLogger.log_error_header(logger, "STREAMING INFERENCE PIPELINE INITIALIZATION FAILED")
    logger.error(f"Failed to initialize streaming inference pipeline: {str(e)}")
    logger.error("Pipeline cannot proceed without valid model instance")
    raise

@log_exceptions(logger)
def streaming_inference(
    inference=None, 
    data=None, 
    use_pyspark: bool = True,
    model_path: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
    batch_data_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute real-time streaming inference for Telco Customer Churn prediction.
    
    This function provides a unified interface for both PySpark Structured Streaming and
    traditional batch inference, supporting real-time predictions on customer data streams.
    
    Streaming Inference Pipeline:
    1. Model Loading - Load trained model for inference
    2. Data Validation - Validate input data format and schema
    3. Preprocessing - Apply same transformations as training pipeline
    4. Prediction - Generate churn probability and classification
    5. MLflow Logging - Track inference metrics and predictions
    6. Result Formatting - Return structured prediction results
    
    Args:
        inference (ModelInference, optional): Pre-initialized model inference instance
                                           for pandas implementation. Used when use_pyspark=False.
        data (Dict/DataFrame, optional): Input customer data for pandas prediction.
                                       Required when use_pyspark=False.
        use_pyspark (bool, optional): Whether to use PySpark Structured Streaming.
                                    If True, uses PySpark for scalable real-time inference.
                                    If False, uses pandas for single-record inference.
                                    Defaults to True.
        model_path (str, optional): Path to the saved PySpark ML pipeline model.
                                  Required when use_pyspark=True.
                                  Defaults to "./artifacts/models/pyspark_pipeline_model".
        input_data (Dict[str, Any], optional): Single customer record for PySpark prediction.
                                             Used for real-time single-record inference.
                                             Requires customer feature values.
        batch_data_path (str, optional): Path to batch data file for PySpark processing.
                                        Used for batch inference on multiple records.
                                        Alternative to input_data for batch processing.
    
    Returns:
        Dict[str, Any]: Comprehensive inference results containing:
            - 'prediction': Churn prediction (0=No Churn, 1=Churn)
            - 'probability': Churn probability score (0.0 to 1.0)
            - 'confidence': Model confidence level
            - 'customer_id': Customer identifier (if provided)
            - 'features': Processed feature values used for prediction
            - 'inference_time': Time taken for prediction
            - 'model_version': Version of the model used
            - 'preprocessing_applied': List of preprocessing steps applied
    
    Raises:
        Exception: If inference pipeline execution fails
        FileNotFoundError: If model_path does not exist
        ValueError: If input data format is invalid or missing required features
        ImportError: If required libraries (PySpark, MLlib) are not available
        MLflowException: If MLflow logging fails
    
    Example:
        >>> # Real-time single customer prediction with PySpark
        >>> customer_data = {
        ...     'gender': 'Female',
        ...     'SeniorCitizen': 0,
        ...     'Partner': 'Yes',
        ...     'Dependents': 'No',
        ...     'tenure': 12,
        ...     'PhoneService': 'Yes',
        ...     'MonthlyCharges': 65.0,
        ...     'TotalCharges': 780.0
        ... }
        >>> result = streaming_inference(
        ...     use_pyspark=True,
        ...     input_data=customer_data
        ... )
        >>> print(f"Churn probability: {result['probability']:.3f}")
        
        >>> # Batch processing with PySpark
        >>> result = streaming_inference(
        ...     use_pyspark=True,
        ...     batch_data_path='data/new_customers.csv'
        ... )
        
        >>> # Traditional pandas inference for backward compatibility
        >>> from model_inference import ModelInference
        >>> inference = ModelInference('artifacts/models/model.joblib')
        >>> result = streaming_inference(
        ...     use_pyspark=False,
        ...     inference=inference,
        ...     data=customer_data
        ... )
    
    Note:
        - PySpark implementation supports both real-time streaming and batch inference
        - Real-time predictions are ideal for web applications and APIs
        - Batch processing is efficient for large datasets
        - All inference results are logged to MLflow for monitoring
        - Model preprocessing is automatically applied to maintain consistency
        - Supports both single predictions and batch predictions
    """
    
    if use_pyspark:
        logger.info("Using PySpark streaming inference implementation")
        return streaming_inference_pyspark(
            model_path=model_path or "./artifacts/models/pyspark_pipeline_model",
            input_data=input_data,
            batch_data_path=batch_data_path
        )
    else:
        # Use original pandas implementation
        return streaming_inference_pandas_original(inference, data)


@log_exceptions(logger)
def streaming_inference_pandas_original(inference, data):
    """
    Perform streaming inference on incoming data.

    Args:
        inference (ModelInference): An instance of the ModelInference class.
        data (iterable): An iterable that yields data points for inference.
        
    Returns:
        dict: Prediction results from the model
    """
    ProjectLogger.log_step_header(logger, "STEP", "STREAMING INFERENCE EXECUTION")
    
    try:
        logger.info("Starting streaming inference process")
        
        # Start MLflow tracking
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
                                            'streaming_inference', 
                                            {
                                                'inference_type': 'single_record',
                                                'model_type': 'XGBoost'
                                            }
                                        )
        run = mlflow_tracker.start_run(run_name='streaming_inference', tags=run_tags)
        logger.info("Inference tracking run started")
        
        # Validate inputs
        if inference is None:
            raise ValueError("ModelInference instance is None")
        
        if data is None:
            raise ValueError("Input data is None")
        
        logger.info("Input validation completed successfully")
        logger.debug(f"Input data type: {type(data)}")
        
        # Load encoders
        try:
            logger.info("Loading encoders for data preprocessing")
            encoder_path = 'artifacts/encode'
            logger.debug(f"Encoder path: {encoder_path}")
            
            inference.load_encoders(encoder_path)
            logger.info("Encoders loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading encoders: {str(e)}")
            raise
        
        # Perform prediction
        try:
            logger.info("Executing model prediction")
            start_time = time.time()

            pred = inference.predict(data)

            inference_time = time.time() - start_time

            if pred is None:
                raise ValueError("Model prediction returned None")
            
                    # Log inference metrics to MLflow
            mlflow.log_metrics({
                                'inference_time_ms': inference_time,
                                'churn_probability': float(pred['Confidence'].replace('%', '')) / 100,
                                'predicted_class': 1 if pred['Status'] == 'Churn' else 0
                              })
            
            mlflow.log_params({f'input_{k}': v for k, v in data.items()})
            
            logger.info("Model prediction completed successfully")
            logger.debug(f"Prediction result: {pred}")
            
            ProjectLogger.log_success_header(logger, "STREAMING INFERENCE COMPLETED")
            return pred
            
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            raise
            
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE VALIDATION ERROR")
        logger.error(f"Validation error: {str(e)}")
        raise
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE FAILED")
        logger.error(f"Unexpected error in streaming inference: {str(e)}")
        raise
    finally:
        mlflow_tracker.end_run()


if __name__ == "__main__":
    try:
        ProjectLogger.log_section_header(logger, "EXECUTING STREAMING INFERENCE DEMO")
        logger.info("Starting streaming inference demonstration")
        
        # Sample data for demonstration (mock customer in actual dataset format)
        data = {
            "customerID": "8472-KMTXZ",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes", 
            "Dependents": "No",
            "tenure": 18,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.75,
            "TotalCharges": 1615.5,
        }
        
        logger.info("Sample data prepared for inference")
        logger.debug(f"Sample data: {data}")
        
        # Validate sample data
        required_fields = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService", "PaymentMethod"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields in sample data: {missing_fields}")
        
        # Execute streaming inference
        logger.info("Executing streaming inference with sample data")
        pred = streaming_inference(inference, data)
        
        # Validate and display results
        if pred is None:
            logger.error("Prediction result is None")
            raise ValueError("Invalid prediction result")
        
        logger.info("Streaming inference completed successfully")
        logger.info(f"Prediction result: {pred}")
        
        # Display results
        print("=" * 50)
        print("STREAMING INFERENCE RESULT")
        print("=" * 50)
        print(pred)
        print("=" * 50)
        
        ProjectLogger.log_success_header(logger, "STREAMING INFERENCE DEMO COMPLETED")
        
    except KeyError as e:
        ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
        logger.error(f"Missing required data field: {str(e)}")
        print(f"Error: Missing required data field - {str(e)}")
        
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "VALUE ERROR")
        logger.error(f"Invalid data value: {str(e)}")
        print(f"Error: Invalid data value - {str(e)}")
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE DEMO FAILED")
        logger.error(f"Unexpected error in demo execution: {str(e)}")
        print(f"Error: Streaming inference demo failed - {str(e)}")
        raise

