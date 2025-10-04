import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
import mlflow
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_pipeline_pyspark import data_pipeline_pyspark
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_spiltter import SimpleTrainTestSplitStrategy
from config import get_data_paths, get_columns, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from logger import get_logger, ProjectLogger, log_exceptions
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags

logger = get_logger(__name__)

@log_exceptions(logger)
def data_pipeline(
    data_path: str = "./data/raw/TelcoCustomerChurnPrediction.csv",
    target_column: str = 'Churn',
    test_size: float = 0.2,
    force_rebuild: bool = False,
    use_pyspark: bool = True
) -> Dict[str, np.ndarray]:
    """
    Execute data pipeline with option to use PySpark or pandas implementation.
    
    Args:
        data_path (str): Path to the raw data file
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        force_rebuild (bool): Whether to force rebuild processed data
        use_pyspark (bool): Whether to use PySpark implementation (default: True)
        
    Returns:
        Dict[str, np.ndarray]: Data arrays for backward compatibility
    """
    
    if use_pyspark:
        logger.info("Using PySpark data pipeline implementation")
        pyspark_result = data_pipeline_pyspark(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            force_rebuild=force_rebuild
        )
        
        # Convert PySpark DataFrames to pandas for backward compatibility
        train_df = pyspark_result['train_df'].toPandas()
        test_df = pyspark_result['test_df'].toPandas()
        
        # Separate features and target
        feature_columns = [col for col in train_df.columns if col != target_column]
        
        X_train = train_df[feature_columns]
        Y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        Y_test = test_df[target_column]
        
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'Y_train': Y_train.values,
            'Y_test': Y_test.values
        }
    else:
        # Use original pandas implementation
        return data_pipeline_pandas_original(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            force_rebuild=force_rebuild
        )


@log_exceptions(logger)
def data_pipeline_pandas_original(
    data_path: str = "./data/raw/TelcoCustomerChurnPrediction.csv",
                    target_column: str = 'Churn',
                    test_size: float = 0.2,
                    force_rebuild: bool = False
                    ) -> Dict[str, np.ndarray]:
    
    ProjectLogger.log_section_header(logger, "STARTING DATA PIPELINE EXECUTION")
    logger.info(f"Input parameters:")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Target column: {target_column}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Force rebuild: {force_rebuild}")
    
    try:
        # Load configuration
        data_paths = get_data_paths()
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        
        logger.info("Configuration loaded successfully")

        mlflow_tracker = MLflowTracker()
        setup_mlflow_autolog()
        run_tags = create_mlflow_run_tags(
                                        'data_pipeline', 
                                        {
                                            'data_source': data_path,
                                        }
                                    )
        run = mlflow_tracker.start_run(run_name='01_data_pipeline_initial', tags=run_tags)

        """
            01. Data Ingestion
        """
        ProjectLogger.log_step_header(logger, "STEP", "1: DATA INGESTION")
        
        relative_path= os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
        artifacts_dir = os.path.abspath(relative_path)
        x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
        x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
        y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
        y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

        # Check if processed data already exists
        if (os.path.exists(x_train_path) and 
            os.path.exists(x_test_path) and 
            os.path.exists(y_train_path) and 
            os.path.exists(y_test_path)):
            
            logger.info("Found existing processed data files, loading them...")
            X_train = pd.read_csv(x_train_path)
            X_test = pd.read_csv(x_test_path)
            Y_train = pd.read_csv(y_train_path)
            Y_test = pd.read_csv(y_test_path)
            
            logger.info(f"Loaded existing data:")
            logger.info(f"  - X_train shape: {X_train.shape}")
            logger.info(f"  - X_test shape: {X_test.shape}")
            logger.info(f"  - Y_train shape: {Y_train.shape}")
            logger.info(f"  - Y_test shape: {Y_test.shape}")

            mlflow_tracker.log_data_pipeline_metrics({
                'total_rows': len(X_train) + len(X_test),
                'train_rows': len(X_train),
                'test_rows': len(X_test),
                'num_features': X_train.shape[1],
                'missing_values': 0,  # Assuming processed data has no missing values
                'outliers_removed': 0,  # Data already processed
                'test_size': test_size,
                'random_state': 42,
                'missing_strategy': 'drop',
                'outlier_method': 'iqr',
                'encoding_applied': True,
                'scaling_applied': True,
                'feature_names': list(X_train.columns)
            })
            mlflow_tracker.end_run()
            return {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'Y_train': Y_train.values.ravel(),
                'Y_test': Y_test.values.ravel()
            }

        logger.info(f"Loading raw data from: {data_path}")
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Store initial metrics for MLflow
        initial_row_count = len(df)

        """
            02. Handling Missing Values
        """
        ProjectLogger.log_step_header(logger, "STEP", "2: HANDLING MISSING VALUES")
        
        logger.info("Checking for missing values...")
        missing_counts = df.isnull().sum()
        missing_summary = missing_counts[missing_counts > 0]
        total_missing_values = missing_counts.sum()
        
        if len(missing_summary) > 0:
            logger.warning(f"Found missing values:\n{missing_summary}")
        else:
            logger.info("No missing values found in the dataset")

        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])
        df = drop_handler.handle(df)
        logger.info(f"Data shape after handling missing values: {df.shape}")
        
        # Track rows removed due to missing values
        rows_after_missing_handling = len(df)
        
        """
            03. Handle Outliers
        """
        ProjectLogger.log_step_header(logger, "STEP", "3: OUTLIER DETECTION AND HANDLING")
        
        logger.info(f"Checking for outliers in columns: {columns['outlier_columns']}")
        outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
        df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
        logger.info(f"Data shape after outlier handling: {df.shape}")
        
        # Track rows removed due to outliers
        rows_after_outlier_handling = len(df)
        outliers_removed = rows_after_missing_handling - rows_after_outlier_handling

        """
            04. Feature Binning
        """
        ProjectLogger.log_step_header(logger, "STEP", "4: FEATURE BINNING")
        
        logger.info(f"Applying binning to 'tenure' feature with config: {binning_config['tenure_bins']}")
        binning = CustomBinningStrategy(binning_config['tenure_bins'])
        df = binning.bin_feature(df, 'tenure')
        logger.info("Feature binning completed successfully")
        logger.debug(f"Sample data after binning:\n{df.head()}")

        """
            05. Feature Encoding
        """
        ProjectLogger.log_step_header(logger, "STEP", "5: FEATURE ENCODING")
        
        logger.info(f"Nominal encoding for columns: {encoding_config['nominal_columns']}")
        nominal_encoder = NominalEncodingStrategy(encoding_config['nominal_columns'])
        df = nominal_encoder.encode(df)
        
        logger.info(f"Ordinal encoding with mappings: {encoding_config['ordinal_mappings']}")
        ordinal_encoder = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])
        df = ordinal_encoder.encode(df)

        logger.info(f"Feature encoding completed. Data shape: {df.shape}")
        logger.debug(f"Sample data after encoding:\n{df.head()}")

        """
            06. Feature Scaling
        """
        ProjectLogger.log_step_header(logger, "STEP", "6: FEATURE SCALING")
        
        logger.info(f"Applying MinMax scaling to columns: {scaling_config['columns_to_scale']}")
        scaling_strategy = MinMaxScalingStrategy()
        df = scaling_strategy.scale(df, scaling_config['columns_to_scale'])
        logger.info("Feature scaling completed successfully")
        logger.debug(f"Sample data after scaling:\n{df.head()}")

        """
            07. Post Processing
        """
        ProjectLogger.log_step_header(logger, "STEP", "7: POST PROCESSING")
        
        logger.info("Dropping 'customerID' column")
        df = df.drop("customerID", axis=1)
        logger.info(f"Data shape after post processing: {df.shape}")
        logger.debug(f"Final columns: {list(df.columns)}")

        """
            08. Data Splitting
        """
        ProjectLogger.log_step_header(logger, "STEP", "8: DATA SPLITTING")
        
        logger.info(f"Splitting data with test size: {splitting_config['test_size']}")
        splitting_strategy = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'])
        X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, 'Churn')

        # Save the split datasets
        logger.info("Saving split datasets to artifacts directory...")
        os.makedirs(artifacts_dir, exist_ok=True)
        X_train.to_csv(x_train_path, index=False)
        X_test.to_csv(x_test_path, index=False)
        Y_train.to_csv(y_train_path, index=False)
        Y_test.to_csv(y_test_path, index=False)

        # Log comprehensive metrics to MLflow using the correct format
        dataset_info = {
            'total_rows': len(X_train) + len(X_test),
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'num_features': len(X_train.columns),
            'missing_values': int(total_missing_values),
            'outliers_removed': outliers_removed,
            'test_size': splitting_config['test_size'],
            'random_state': 42,
            'missing_strategy': 'drop',
            'outlier_method': 'iqr',
            'encoding_applied': True,
            'scaling_applied': True,
            'feature_names': list(X_train.columns),
            'X_train': X_train,
            'Y_train': Y_train,
        }

        mlflow_tracker.log_data_pipeline_metrics(dataset_info)

        # Log additional pipeline parameters
        mlflow.log_params({
            'data_source': data_path,
            'target_column': target_column,
            'preprocessing_steps': ['data_ingestion', 'missing_values', 'outlier_detection', 'feature_binning', 
                                  'feature_encoding', 'feature_scaling', 'post_processing', 'data_splitting'],
            'data_pipeline_version': '1.0_pandas',
            'force_rebuild': force_rebuild
        })

        mlflow_tracker.end_run()
        
        # Add missing return statement
        logger.info("Data splitting completed successfully:")
        logger.info(f"  - X_train shape: {X_train.shape}")
        logger.info(f"  - X_test shape: {X_test.shape}")
        logger.info(f"  - Y_train shape: {Y_train.shape}")
        logger.info(f"  - Y_test shape: {Y_test.shape}")
        
        ProjectLogger.log_success_header(logger, "DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'Y_train': Y_train.values.ravel(),
            'Y_test': Y_test.values.ravel()
        }
        
    except Exception as e:
        logger.error(f"Error in data pipeline execution: {str(e)}")
        logger.error("Pipeline execution failed", exc_info=True)
        if 'mlflow_tracker' in locals(): # locals() -> returns a dictionary of the current local variables
            mlflow_tracker.end_run()
        raise

if __name__ == "__main__":
    try:
        result = data_pipeline()
        logger.info("Data pipeline completed successfully")
        logger.info(f"Returned data shapes:")
        logger.info(f"  - X_train: {result['X_train'].shape}")
        logger.info(f"  - X_test: {result['X_test'].shape}")
        logger.info(f"  - Y_train: {result['Y_train'].shape}")
        logger.info(f"  - Y_test: {result['Y_test'].shape}")
    except Exception as e:
        logger.error(f"Data pipeline execution failed: {str(e)}")
        raise
