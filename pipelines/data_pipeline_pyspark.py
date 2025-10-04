"""
PySpark Data Pipeline for Telco Customer Churn Prediction.
Handles the complete data preprocessing pipeline using PySpark operations.
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy
from feature_scaling import FeatureScaler, ScalingType
from data_spiltter import create_data_splitter, SplitType
from config import (
    get_data_paths, get_columns, get_outlier_config, get_binning_config, 
    get_encoding_config, get_scaling_config, get_splitting_config
)
from logger import get_logger, ProjectLogger, log_exceptions
from spark_utils import get_spark_session

logger = get_logger(__name__)


@log_exceptions(logger)
def data_pipeline_pyspark(
    data_path: str = "./data/raw/TelcoCustomerChurnPrediction.csv",
    target_column: str = 'Churn',
    test_size: float = 0.2,
    force_rebuild: bool = False,
    spark: Optional[SparkSession] = None
) -> Dict[str, DataFrame]:
    """
    Execute the complete data preprocessing pipeline using PySpark.
    
    Args:
        data_path (str): Path to the raw data file
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        force_rebuild (bool): Whether to force rebuild processed data
        spark: Optional SparkSession
        
    Returns:
        Dict[str, DataFrame]: Dictionary containing train/test splits
    """
    
    ProjectLogger.log_section_header(logger, "STARTING PYSPARK DATA PIPELINE EXECUTION")
    logger.info(f"Input parameters:")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Target column: {target_column}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Force rebuild: {force_rebuild}")
    
    try:
        # Initialize Spark session
        spark = spark or get_spark_session()
        logger.info(f"Using Spark session: {spark.sparkContext.appName}")
        
        # Load configuration
        data_paths = get_data_paths()
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        
        logger.info("Configuration loaded successfully")

        # Define output paths for Parquet format (better for Spark)
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir']))
        
        # Check if processed data already exists
        parquet_train_path = os.path.join(artifacts_dir, 'train_data.parquet')
        parquet_test_path = os.path.join(artifacts_dir, 'test_data.parquet')
        
        if (not force_rebuild and 
            os.path.exists(parquet_train_path) and 
            os.path.exists(parquet_test_path)):
            
            logger.info("Found existing processed Parquet data files, loading them...")
            train_df = spark.read.parquet(parquet_train_path)
            test_df = spark.read.parquet(parquet_test_path)
            
            logger.info(f"Loaded existing data:")
            logger.info(f"  - Train data: {train_df.count()} rows, {len(train_df.columns)} columns")
            logger.info(f"  - Test data: {test_df.count()} rows, {len(test_df.columns)} columns")
            
            return {
                'train_df': train_df,
                'test_df': test_df
            }

        """
        01. Data Ingestion
        """
        ProjectLogger.log_step_header(logger, "STEP", "1: DATA INGESTION")
        
        logger.info(f"Loading raw data from: {data_path}")
        ingestor = DataIngestorCSV(spark=spark)
        df = ingestor.ingest(data_path)
        
        initial_count = df.count()
        logger.info(f"Successfully loaded data with {initial_count} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {df.columns}")
        
        # Cache the initial DataFrame for performance
        df.cache()
        
        """
        02. Handling Missing Values
        """
        ProjectLogger.log_step_header(logger, "STEP", "2: HANDLING MISSING VALUES")
        
        logger.info("Checking for missing values...")
        
        # Check for missing values in each column
        missing_counts = {}
        for col in df.columns:
            missing_count = df.filter(F.col(col).isNull() | (F.col(col) == "")).count()
            if missing_count > 0:
                missing_counts[col] = missing_count
        
        if missing_counts:
            logger.warning(f"Found missing values: {missing_counts}")
        else:
            logger.info("No missing values found in the dataset")

        # Handle missing values
        drop_handler = DropMissingValuesStrategy(
            critical_columns=columns['critical_columns'],
            spark=spark
        )
        df = drop_handler.handle(df)
        
        rows_after_missing_handling = df.count()
        logger.info(f"Data shape after handling missing values: {rows_after_missing_handling} rows")
        
        """
        03. Handle Outliers
        """
        ProjectLogger.log_step_header(logger, "STEP", "3: OUTLIER DETECTION AND HANDLING")
        
        logger.info(f"Checking for outliers in columns: {columns['outlier_columns']}")
        outlier_detector = OutlierDetector(
            strategy=IQROutlierDetection(spark=spark)
        )
        
        initial_outlier_count = df.count()
        df = outlier_detector.detect_and_handle(df, columns['outlier_columns'])
        final_outlier_count = df.count()
        
        outliers_removed = initial_outlier_count - final_outlier_count
        logger.info(f"Outlier detection completed. Removed {outliers_removed} outlier rows")
        logger.info(f"Data shape after outlier handling: {final_outlier_count} rows")
        
        """
        04. Feature Engineering - Binning
        """
        ProjectLogger.log_step_header(logger, "STEP", "4: FEATURE BINNING")
        
        logger.info("Applying feature binning...")
        binning_strategy = CustomBinningStrategy(
            binning_config=binning_config,
            spark=spark
        )
        df = binning_strategy.bin_features(df)
        logger.info("Feature binning completed successfully")
        
        """
        05. Feature Encoding
        """
        ProjectLogger.log_step_header(logger, "STEP", "5: FEATURE ENCODING")
        
        logger.info("Applying ordinal encoding...")
        encoding_strategy = OrdinalEncodingStrategy(
            ordinal_mappings=encoding_config['ordinal_mappings'],
            spark=spark
        )
        df = encoding_strategy.encode(df)
        logger.info("Feature encoding completed successfully")
        
        # Save encoders for later use in inference
        encoding_artifacts_dir = os.path.join(artifacts_dir, '..', 'encode')
        os.makedirs(encoding_artifacts_dir, exist_ok=True)
        encoding_strategy.save_encoders(encoding_artifacts_dir)
        
        """
        06. Data Splitting
        """
        ProjectLogger.log_step_header(logger, "STEP", "6: DATA SPLITTING")
        
        logger.info(f"Splitting data into train/test sets (test_size: {test_size})")
        
        # Use stratified splitting to maintain class distribution
        splitter = create_data_splitter(
            split_type=SplitType.STRATIFIED,
            test_size=test_size,
            random_state=42,
            spark=spark
        )
        
        train_df, test_df = splitter.split_data(df, target_column)
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        logger.info(f"Data splitting completed:")
        logger.info(f"  - Training set: {train_count} rows")
        logger.info(f"  - Test set: {test_count} rows")
        logger.info(f"  - Actual split ratio: {test_count / (train_count + test_count):.3f}")
        
        """
        07. Feature Scaling
        """
        ProjectLogger.log_step_header(logger, "STEP", "7: FEATURE SCALING")
        
        # Get numerical columns for scaling (exclude target column)
        numerical_columns = [col for col in columns['outlier_columns'] if col in train_df.columns and col != target_column]
        
        if numerical_columns:
            logger.info(f"Scaling numerical features: {numerical_columns}")
            
            # Initialize scaler
            scaler = FeatureScaler(
                scaling_type=ScalingType.STANDARD,  # Use standard scaling
                spark=spark
            )
            
            # Fit scaler on training data and transform both train and test
            train_df = scaler.scale_features(train_df, numerical_columns)
            test_df = scaler.scale_features(test_df, numerical_columns)
            
            logger.info("Feature scaling completed successfully")
        else:
            logger.info("No numerical columns found for scaling")
        
        """
        08. Save Processed Data
        """
        ProjectLogger.log_step_header(logger, "STEP", "8: SAVING PROCESSED DATA")
        
        # Create artifacts directory if it doesn't exist
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save as Parquet for better performance with Spark
        logger.info(f"Saving processed data to: {artifacts_dir}")
        
        train_df.write.mode('overwrite').parquet(parquet_train_path)
        test_df.write.mode('overwrite').parquet(parquet_test_path)
        
        # Also save as CSV for backward compatibility
        csv_dir = os.path.join(artifacts_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        
        # Separate features and target for CSV export
        feature_columns = [col for col in train_df.columns if col != target_column]
        
        # Save training data
        X_train = train_df.select(*feature_columns)
        Y_train = train_df.select(target_column)
        
        X_train.coalesce(1).write.mode('overwrite').option('header', True).csv(os.path.join(csv_dir, 'X_train'))
        Y_train.coalesce(1).write.mode('overwrite').option('header', True).csv(os.path.join(csv_dir, 'Y_train'))
        
        # Save test data
        X_test = test_df.select(*feature_columns)
        Y_test = test_df.select(target_column)
        
        X_test.coalesce(1).write.mode('overwrite').option('header', True).csv(os.path.join(csv_dir, 'X_test'))
        Y_test.coalesce(1).write.mode('overwrite').option('header', True).csv(os.path.join(csv_dir, 'Y_test'))
        
        logger.info("Data saved successfully in both Parquet and CSV formats")
        
        """
        09. Data Quality Summary
        """
        ProjectLogger.log_step_header(logger, "STEP", "9: DATA QUALITY SUMMARY")
        
        # Generate data quality summary
        final_train_count = train_df.count()
        final_test_count = test_df.count()
        final_feature_count = len(feature_columns)
        
        # Check target distribution
        train_target_dist = train_df.groupBy(target_column).count().collect()
        test_target_dist = test_df.groupBy(target_column).count().collect()
        
        summary = {
            'initial_rows': initial_count,
            'final_train_rows': final_train_count,
            'final_test_rows': final_test_count,
            'final_features': final_feature_count,
            'rows_removed_missing': initial_count - rows_after_missing_handling,
            'rows_removed_outliers': outliers_removed,
            'test_split_ratio': test_count / (train_count + test_count),
            'train_target_distribution': {str(row[target_column]): row['count'] for row in train_target_dist},
            'test_target_distribution': {str(row[target_column]): row['count'] for row in test_target_dist},
            'processing_timestamp': datetime.now().isoformat(),
            'feature_columns': feature_columns
        }
        
        # Save summary
        summary_path = os.path.join(artifacts_dir, 'data_pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data pipeline summary:")
        logger.info(f"  - Initial rows: {initial_count}")
        logger.info(f"  - Final train rows: {final_train_count}")
        logger.info(f"  - Final test rows: {final_test_count}")
        logger.info(f"  - Final features: {final_feature_count}")
        logger.info(f"  - Rows removed (missing): {summary['rows_removed_missing']}")
        logger.info(f"  - Rows removed (outliers): {summary['rows_removed_outliers']}")
        
        # Clean up cache
        df.unpersist()
        
        ProjectLogger.log_success_header(logger, "PYSPARK DATA PIPELINE COMPLETED SUCCESSFULLY")
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'summary': summary
        }
        
    except FileNotFoundError as e:
        ProjectLogger.log_error_header(logger, "DATA PIPELINE FAILED - FILE NOT FOUND")
        logger.error(f"File not found error: {str(e)}")
        raise
        
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "DATA PIPELINE FAILED - DATA VALIDATION ERROR")
        logger.error(f"Data validation error: {str(e)}")
        raise
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "DATA PIPELINE FAILED - UNEXPECTED ERROR")
        logger.error(f"Unexpected error during data pipeline: {str(e)}")
        logger.error("Data pipeline failed", exc_info=True)
        raise


def data_pipeline(
    data_path: str = "./data/raw/TelcoCustomerChurnPrediction.csv",
    target_column: str = 'Churn',
    test_size: float = 0.2,
    force_rebuild: bool = False
) -> Dict[str, DataFrame]:
    """
    Legacy function wrapper for backward compatibility.
    """
    return data_pipeline_pyspark(
        data_path=data_path,
        target_column=target_column,
        test_size=test_size,
        force_rebuild=force_rebuild
    )


if __name__ == "__main__":
    logger.info("Starting PySpark data pipeline execution")
    
    try:
        result = data_pipeline_pyspark()
        logger.info("Data pipeline execution completed successfully")
        logger.info(f"Results: {list(result.keys())}")
        
        ProjectLogger.log_success_header(logger, "DATA PIPELINE MAIN EXECUTION COMPLETED")
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "DATA PIPELINE MAIN EXECUTION FAILED")
        logger.error(f"Error: {str(e)}")
        raise