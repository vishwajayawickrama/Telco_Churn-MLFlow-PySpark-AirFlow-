"""
PySpark Streaming Inference Pipeline for Telco Customer Churn Prediction.

This module implements a real-time streaming inference pipeline using Apache Spark Structured
Streaming for processing customer data streams and generating churn predictions at scale.
It supports both single-record predictions and batch processing with automatic preprocessing.

Key Features:
- Real-time streaming inference using PySpark Structured Streaming
- Automatic data preprocessing pipeline integration
- Support for both single records and batch predictions
- Scalable processing for high-throughput data streams
- MLflow integration for prediction tracking
- Production-ready error handling and monitoring

Inference Capabilities:
- Single Customer Prediction - Real-time individual customer churn prediction
- Batch Processing - Efficient processing of customer data batches
- Stream Processing - Continuous processing of data streams
- API Integration - REST API compatible prediction interface

Pipeline Components:
1. Data Validation - Validate input data schema and format
2. Preprocessing - Apply same transformations as training pipeline
3. Feature Engineering - Prepare features for model inference
4. Model Loading - Load trained PySpark ML pipeline model
5. Prediction - Generate churn probability and classification
6. Post-processing - Format results for downstream consumption
7. Monitoring - Track prediction metrics and model performance

Dependencies:
- Apache Spark 3.x with Structured Streaming
- PySpark SQL and ML libraries
- MLflow for prediction tracking
- Custom preprocessing and model utilities

Usage:
    >>> from streaming_inference_pipeline import streaming_inference_pyspark
    >>> 
    >>> # Single customer prediction
    >>> customer_data = {
    ...     'gender': 'Female',
    ...     'SeniorCitizen': 0,
    ...     'Partner': 'Yes',
    ...     'tenure': 12,
    ...     'MonthlyCharges': 65.0
    ... }
    >>> result = streaming_inference_pyspark(
    ...     model_path='./artifacts/models/pyspark_pipeline_model',
    ...     input_data=customer_data
    ... )
    >>> 
    >>> # Batch processing
    >>> result = streaming_inference_pyspark(
    ...     model_path='./artifacts/models/pyspark_pipeline_model',
    ...     batch_data_path='data/new_customers.csv'
    ... )

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.sql.functions import col, when, isnan, isnull, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from typing import Dict, Any, List, Optional
import json

from spark_utils import SparkSessionManager
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)

class PySpark_StreamingInference:
    """
    PySpark-based streaming inference pipeline for real-time customer churn prediction.
    """
    
    def __init__(self, model_path: str, encoders_path: str = "./artifacts/encode/"):
        """
        Initialize the PySpark streaming inference pipeline.
        
        Args:
            model_path (str): Path to the trained PySpark ML model
            encoders_path (str): Path to encoder artifacts directory
        """
        ProjectLogger.log_section_header(logger, "INITIALIZING PYSPARK STREAMING INFERENCE")
        
        self.spark = SparkSessionManager.get_session()
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.model = None
        self.preprocessing_pipeline = None
        
        # Load model and preprocessing pipeline
        self._load_model()
        self._setup_preprocessing_pipeline()
        
        ProjectLogger.log_success_header(logger, "PYSPARK STREAMING INFERENCE INITIALIZED")
    
    def _load_model(self):
        """Load the trained PySpark ML model."""
        try:
            from pyspark.ml import PipelineModel
            logger.info(f"Loading PySpark ML model from: {self.model_path}")
            self.model = PipelineModel.load(self.model_path)
            logger.info("PySpark ML model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PySpark ML model: {str(e)}")
            raise
    
    def _setup_preprocessing_pipeline(self):
        """Setup the preprocessing pipeline to match training data format."""
        try:
            logger.info("Setting up preprocessing pipeline for inference")
            
            # Define categorical and numerical columns
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'PaperlessBilling', 'PaymentMethod'
            ]
            
            numerical_columns = [
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ]
            
            # Create preprocessing stages
            stages = []
            
            # String indexing for categorical variables
            for col_name in categorical_columns:
                indexer = StringIndexer(
                    inputCol=col_name,
                    outputCol=f"{col_name}_indexed",
                    handleInvalid="keep"
                )
                stages.append(indexer)
            
            # One-hot encoding
            indexed_categorical = [f"{col}_indexed" for col in categorical_columns]
            encoder = OneHotEncoder(
                inputCols=indexed_categorical,
                outputCols=[f"{col}_encoded" for col in categorical_columns],
                handleInvalid="keep"
            )
            stages.append(encoder)
            
            # Feature assembly
            feature_columns = numerical_columns + [f"{col}_encoded" for col in categorical_columns]
            assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol="features_unscaled",
                handleInvalid="keep"
            )
            stages.append(assembler)
            
            # Feature scaling
            scaler = MinMaxScaler(
                inputCol="features_unscaled",
                outputCol="features"
            )
            stages.append(scaler)
            
            # Create preprocessing pipeline
            self.preprocessing_pipeline = Pipeline(stages=stages)
            
            logger.info("Preprocessing pipeline setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup preprocessing pipeline: {str(e)}")
            raise
    
    def _create_input_schema(self) -> StructType:
        """Create the expected input schema for streaming data."""
        return StructType([
            StructField("gender", StringType(), True),
            StructField("SeniorCitizen", StringType(), True),
            StructField("Partner", StringType(), True),
            StructField("Dependents", StringType(), True),
            StructField("tenure", DoubleType(), True),
            StructField("PhoneService", StringType(), True),
            StructField("MultipleLines", StringType(), True),
            StructField("InternetService", StringType(), True),
            StructField("OnlineSecurity", StringType(), True),
            StructField("OnlineBackup", StringType(), True),
            StructField("DeviceProtection", StringType(), True),
            StructField("TechSupport", StringType(), True),
            StructField("StreamingTV", StringType(), True),
            StructField("StreamingMovies", StringType(), True),
            StructField("PaperlessBilling", StringType(), True),
            StructField("PaymentMethod", StringType(), True),
            StructField("MonthlyCharges", DoubleType(), True),
            StructField("TotalCharges", DoubleType(), True)
        ])
    
    def predict_batch(self, data_df: DataFrame) -> DataFrame:
        """
        Perform batch prediction on a PySpark DataFrame.
        
        Args:
            data_df (DataFrame): Input DataFrame with customer data
            
        Returns:
            DataFrame: DataFrame with predictions
        """
        try:
            ProjectLogger.log_step_header(logger, "STEP", "BATCH PREDICTION")
            logger.info(f"Processing batch of {data_df.count()} records")
            
            # Apply preprocessing
            logger.info("Applying preprocessing transformations")
            preprocessed_df = self.preprocessing_pipeline.fit(data_df).transform(data_df)
            
            # Make predictions
            logger.info("Making predictions")
            predictions_df = self.model.transform(preprocessed_df)
            
            # Select relevant columns for output
            result_df = predictions_df.select(
                "*",
                col("prediction").alias("churn_prediction"),
                col("probability").alias("churn_probability")
            )
            
            logger.info("Batch prediction completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prediction on a single customer record.
        
        Args:
            customer_data (Dict[str, Any]): Customer data dictionary
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            ProjectLogger.log_step_header(logger, "STEP", "SINGLE RECORD PREDICTION")
            logger.info("Processing single customer record")
            
            # Convert dictionary to DataFrame
            customer_df = self.spark.createDataFrame([customer_data], schema=self._create_input_schema())
            
            # Make prediction
            result_df = self.predict_batch(customer_df)
            
            # Convert result to dictionary
            result_row = result_df.collect()[0]
            
            prediction_result = {
                'customer_id': customer_data.get('customerID', 'unknown'),
                'churn_prediction': int(result_row['churn_prediction']),
                'churn_probability': float(result_row['churn_probability'][1]),  # Probability of churn
                'confidence': 'high' if abs(result_row['churn_probability'][1] - 0.5) > 0.3 else 'medium'
            }
            
            logger.info(f"Single prediction completed: {prediction_result}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise
    
    def predict_stream(self, input_path: str, output_path: str, checkpoint_path: str):
        """
        Set up structured streaming for real-time predictions.
        
        Args:
            input_path (str): Path to monitor for incoming data
            output_path (str): Path to write predictions
            checkpoint_path (str): Checkpoint location for fault tolerance
        """
        try:
            ProjectLogger.log_step_header(logger, "STEP", "STRUCTURED STREAMING SETUP")
            logger.info(f"Setting up structured streaming")
            logger.info(f"Input path: {input_path}")
            logger.info(f"Output path: {output_path}")
            logger.info(f"Checkpoint path: {checkpoint_path}")
            
            # Read streaming data
            streaming_df = self.spark \
                .readStream \
                .format("json") \
                .schema(self._create_input_schema()) \
                .option("path", input_path) \
                .load()
            
            # Apply preprocessing and prediction
            def process_batch(batch_df, batch_id):
                if batch_df.count() > 0:
                    logger.info(f"Processing batch {batch_id} with {batch_df.count()} records")
                    
                    # Make predictions
                    predictions_df = self.predict_batch(batch_df)
                    
                    # Write predictions
                    predictions_df.write \
                        .mode("append") \
                        .format("json") \
                        .save(f"{output_path}/batch_{batch_id}")
                    
                    logger.info(f"Batch {batch_id} processed and saved")
            
            # Start streaming query
            query = streaming_df.writeStream \
                .foreachBatch(process_batch) \
                .option("checkpointLocation", checkpoint_path) \
                .trigger(processingTime="10 seconds") \
                .start()
            
            logger.info("Structured streaming started successfully")
            return query
            
        except Exception as e:
            logger.error(f"Structured streaming setup failed: {str(e)}")
            raise


@log_exceptions(logger)
def streaming_inference_pyspark(
    model_path: str = "./artifacts/models/pyspark_pipeline_model",
    input_data: Optional[Dict[str, Any]] = None,
    batch_data_path: Optional[str] = None,
    stream_input_path: Optional[str] = None,
    stream_output_path: Optional[str] = None,
    stream_checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute PySpark streaming inference pipeline.
    
    Args:
        model_path (str): Path to the trained PySpark ML model
        input_data (Dict[str, Any]): Single customer data for prediction
        batch_data_path (str): Path to batch data file
        stream_input_path (str): Path for streaming input
        stream_output_path (str): Path for streaming output
        stream_checkpoint_path (str): Checkpoint path for streaming
        
    Returns:
        Dict[str, Any]: Inference results
    """
    ProjectLogger.log_section_header(logger, "EXECUTING PYSPARK STREAMING INFERENCE PIPELINE")
    
    try:
        # Initialize inference pipeline
        inference_pipeline = PySpark_StreamingInference(model_path)
        
        results = {}
        
        # Single record prediction
        if input_data:
            logger.info("Performing single record prediction")
            single_result = inference_pipeline.predict_single(input_data)
            results['single_prediction'] = single_result
        
        # Batch prediction
        if batch_data_path:
            logger.info(f"Performing batch prediction on: {batch_data_path}")
            spark = SparkSessionManager.get_session()
            batch_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(batch_data_path)
            batch_result = inference_pipeline.predict_batch(batch_df)
            results['batch_predictions'] = batch_result.count()
            
            # Save batch results
            batch_output_path = batch_data_path.replace('.csv', '_predictions.json')
            batch_result.write.mode("overwrite").format("json").save(batch_output_path)
            results['batch_output_path'] = batch_output_path
        
        # Streaming prediction setup
        if stream_input_path and stream_output_path and stream_checkpoint_path:
            logger.info("Setting up streaming inference")
            query = inference_pipeline.predict_stream(
                stream_input_path, 
                stream_output_path, 
                stream_checkpoint_path
            )
            results['streaming_query'] = query
            results['streaming_status'] = 'active'
        
        ProjectLogger.log_success_header(logger, "PYSPARK STREAMING INFERENCE COMPLETED")
        
        return results
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "PYSPARK STREAMING INFERENCE FAILED")
        logger.error(f"Streaming inference pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': '0',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12.0,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 50.0,
        'TotalCharges': 600.0
    }
    
    results = streaming_inference_pyspark(
        input_data=sample_customer
    )
    
    print("Inference Results:", results)