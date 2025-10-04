"""
Unified Training Pipeline for Telco Customer Churn Prediction.

This module provides a unified interface for both PySpark MLlib and scikit-learn
implementations of the machine learning training pipeline. It supports multiple
algorithms with comprehensive model evaluation and experiment tracking.

Key Features:
- Dual Implementation Support - Choose between PySpark MLlib and scikit-learn
- Multiple ML Algorithms - XGBoost, RandomForest, GBT, LogisticRegression
- Hyperparameter Tuning - Automated parameter optimization
- Comprehensive Evaluation - Multiple metrics and model performance analysis
- MLflow Integration - Complete experiment tracking and model registry
- Production Ready - Model persistence and deployment preparation

Algorithm Support:
1. XGBoost Classifier (Default):
   - Excellent performance on tabular data
   - Built-in feature importance
   - Handles missing values automatically
   - Memory efficient gradient boosting

2. Random Forest:
   - Good interpretability and robustness
   - Handles overfitting well
   - Provides feature importance scores
   - Works well with mixed data types

3. Gradient Boosted Trees (PySpark only):
   - Distributed training capability
   - High accuracy for large datasets
   - Built-in regularization
   - Scalable to big data

Training Pipeline Features:
- Automatic data loading from preprocessing pipeline
- Model training with optimized hyperparameters
- Comprehensive evaluation with multiple metrics
- Feature importance analysis
- Model persistence for deployment
- Experiment tracking with MLflow
- Performance monitoring and logging

Usage:
    >>> # Train with PySpark for distributed processing
    >>> result = training_pipeline(use_pyspark=True)
    >>> 
    >>> # Train with custom parameters
    >>> custom_params = {
    ...     'n_estimators': 200,
    ...     'max_depth': 8,
    ...     'learning_rate': 0.1
    ... }
    >>> result = training_pipeline(model_params=custom_params)
    >>> 
    >>> # Use scikit-learn for local training
    >>> result = training_pipeline(use_pyspark=False)

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

import json
import os
import sys
import pickle
import pandas as pd
import mlflow
from typing import Dict, Any, Optional

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import pipeline implementations
from data_pipeline import data_pipeline
from training_pipeline_pyspark import training_pipeline_pyspark

# Import pandas implementation components (for fallback)
from model_building import RandomForestModelBuilder, XGBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from config import get_model_config, get_data_paths
from logger import get_logger, ProjectLogger, log_exceptions
from mlflow_utils import MLflowTracker, create_mlflow_run_tags

# Initialize logger
logger = get_logger(__name__)

@log_exceptions(logger)
def training_pipeline(
                        data_path: str = 'data/raw/TelcoCustomerChurnPrediction.csv',
                        model_params: Optional[Dict[str, Any]] = None,
                        test_size: float = 0.2, 
                        random_state: int = 42,
                        model_path: str = 'artifacts/models/telco_customer_churn_prediction.joblib',
                        use_pyspark: bool = True
                    ) -> Dict[str, Any]:
    """
    Execute the complete machine learning training pipeline for Telco Customer Churn prediction.
    
    This function provides a unified interface for both PySpark ML and traditional scikit-learn
    implementations, supporting multiple algorithms and comprehensive model evaluation.
    
    Training Pipeline Steps:
    1. Data Pipeline - Execute preprocessing pipeline to prepare training data
    2. Model Training - Train XGBoost classifier with optimal parameters
    3. Model Evaluation - Assess performance with accuracy, precision, recall, F1-score
    4. Model Persistence - Save trained model for deployment
    5. MLflow Logging - Track experiments, parameters, and model artifacts
    
    Args:
        data_path (str, optional): Path to the raw dataset CSV file.
                                 Defaults to 'data/raw/TelcoCustomerChurnPrediction.csv'.
        model_params (Optional[Dict[str, Any]], optional): Custom hyperparameters for the model.
                                                          If None, uses optimized default parameters.
                                                          Defaults to None.
        test_size (float, optional): Proportion of dataset reserved for testing (0.0 to 1.0).
                                    Defaults to 0.2 (20% for testing).
        random_state (int, optional): Random seed for reproducible results across runs.
                                    Defaults to 42.
        model_path (str, optional): File path where the trained model will be saved.
                                  Defaults to 'artifacts/models/telco_customer_churn_prediction.joblib'.
        use_pyspark (bool, optional): Whether to use PySpark ML for distributed training.
                                    If True, uses PySpark ML algorithms for scalability.
                                    If False, uses scikit-learn for local training.
                                    Defaults to True.
    
    Returns:
        Dict[str, Any]: Comprehensive training results containing:
            - 'model': Trained XGBoost classifier object
            - 'accuracy': Model accuracy score on test set
            - 'classification_report': Detailed precision, recall, F1 scores per class
            - 'confusion_matrix': Confusion matrix for error analysis
            - 'feature_importance': Feature importance scores from the model
            - 'training_time': Total time taken for training process
            - 'model_path': Path where the model artifact is saved
            - 'data_shapes': Training and testing data dimensions
    
    Raises:
        Exception: If training pipeline execution fails at any step
        FileNotFoundError: If the specified data_path file does not exist
        ValueError: If test_size is not between 0.0 and 1.0
        ImportError: If required ML libraries (XGBoost, PySpark) are not available
        MLflowException: If MLflow logging fails
    
    Example:
        >>> # Use PySpark for distributed training (recommended)
        >>> result = training_pipeline(use_pyspark=True)
        >>> model = result['model']
        >>> accuracy = result['accuracy']
        >>> print(f"Model accuracy: {accuracy:.3f}")
        
        >>> # Train with custom hyperparameters
        >>> custom_params = {
        ...     'n_estimators': 200,
        ...     'max_depth': 8,
        ...     'learning_rate': 0.1
        ... }
        >>> result = training_pipeline(model_params=custom_params)
        
        >>> # Use traditional scikit-learn for smaller datasets
        >>> result = training_pipeline(use_pyspark=False)
        
        >>> # Custom data split and model save location
        >>> result = training_pipeline(
        ...     test_size=0.3,
        ...     model_path='models/custom_model.joblib'
        ... )
    
    Note:
        - PySpark implementation provides better scalability for large datasets
        - XGBoost is used as the primary algorithm due to superior performance
        - All training experiments are automatically logged to MLflow
        - Model artifacts include both the trained model and preprocessing encoders
        - The function automatically handles data preprocessing if needed
        - Training results are cached for subsequent pipeline runs
    """
    
    if use_pyspark:
        logger.info("Using PySpark training pipeline implementation")
        return training_pipeline_pyspark(
            data_path=data_path,
            test_size=test_size
        )
    else:
        # Use original pandas implementation
        return training_pipeline_pandas_original(
            data_path=data_path,
            model_params=model_params,
            test_size=test_size,
            random_state=random_state,
            model_path=model_path
        )


@log_exceptions(logger)
def training_pipeline_pandas_original(
                        data_path: str = 'data/raw/TelcoCustomerChurnPrediction.csv',
                        model_params: Optional[Dict[str, Any]] = None,
                        test_size: float = 0.2, 
                        random_state: int = 42,
                        model_path: str = 'artifacts/models/telco_customer_churn_prediction.joblib',
                    ) -> Dict[str, Any]:
    """
    Execute the complete training pipeline for customer churn prediction.
    
    Args:
        data_path (str): Path to the raw data file
        model_params (Optional[Dict[str, Any]]): Model hyperparameters
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        model_path (str): Path to save the trained model
    """
    
    ProjectLogger.log_section_header(logger, "STARTING TRAINING PIPELINE EXECUTION")
    logger.info(f"Training Parameters:")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Model parameters: {model_params}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Random state: {random_state}")
    logger.info(f"  - Model save path: {model_path}")
    
    try:
        # Initialize model parameters if not provided
        if model_params is None:
            model_params = {}
            logger.info("Using default model parameters")

        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
                                        'training_pipeline', {
                                                            'model_type' : 'XGboost',
                                                            'training_strategy' : 'simple',
                                                            'data_path': data_path,
                                                            'model_path': model_path,
                                                            'processing_engine': 'scikit-learn',
                                                            }
                                                            )
        run = mlflow_tracker.start_run(run_name='01_training_pipeline_sklearn', tags=run_tags)
        run_artifacts_dir = os.path.join('artifacts', 'mlflow_training_artifacts', run.info.run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)
        
        # Step 1: Check and prepare data
        ProjectLogger.log_step_header(logger, "STEP", "1: DATA PREPARATION")
        
        if  (not os.path.exists(get_data_paths()['X_train'])) or \
            (not os.path.exists(get_data_paths()['X_test'])) or \
            (not os.path.exists(get_data_paths()['Y_train'])) or \
            (not os.path.exists(get_data_paths()['Y_test'])):
            logger.info("Preprocessed data not found. Running data pipeline...")
            data_pipeline()
            logger.info("Data pipeline executed successfully")
        else:
            logger.info("Preprocessed data already exists. Skipping data pipeline execution.")
        
        # Step 2: Load preprocessed data
        ProjectLogger.log_step_header(logger, "STEP", "2: LOADING PREPROCESSED DATA")
        
        logger.info("Loading training and test datasets...")
        X_train = pd.read_csv(get_data_paths()['X_train'])
        X_test = pd.read_csv(get_data_paths()['X_test'])
        Y_train = pd.read_csv(get_data_paths()['Y_train'])
        Y_test = pd.read_csv(get_data_paths()['Y_test'])
        
        # Validate loaded data
        logger.info("Validating loaded data...")
        if X_train.empty or X_test.empty or Y_train.empty or Y_test.empty:
            raise ValueError("One or more datasets are empty")
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  - X_train shape: {X_train.shape}")
        logger.info(f"  - X_test shape: {X_test.shape}")
        logger.info(f"  - Y_train shape: {Y_train.shape}")
        logger.info(f"  - Y_test shape: {Y_test.shape}")
        
        # Check for data consistency
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature mismatch: X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]}")
        
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(f"Sample mismatch: X_train has {X_train.shape[0]} samples, Y_train has {Y_train.shape[0]}")
            
        logger.info("Data validation completed successfully")

        mlflow.log_metrics({
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'num_features': X_train.shape[1],
                        'train_class_0': (Y_train == 0).sum().iloc[0],
                        'train_class_1': (Y_train == 1).sum().iloc[0],
                        'test_class_0': (Y_test == 0).sum().iloc[0],
                        'test_class_1': (Y_test == 1).sum().iloc[0]
                        })
        
        mlflow.log_param('feature_names', list(X_train.columns))
        
        # Step 3: Model building
        ProjectLogger.log_step_header(logger, "STEP", "3: MODEL BUILDING")
        
        logger.info(f"Building XGBoost model with parameters: {model_params}")
        model_builder = XGBoostModelBuilder(**model_params)
        model = model_builder.build_model()
        logger.info("Model built successfully")
        
        # Step 4: Model training
        ProjectLogger.log_step_header(logger, "STEP", "4: MODEL TRAINING")
        
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer()
        
        import time
        logger.info("Starting model training...")
        start_time = time.time()
        model, _ = trainer.train(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
        )
        training_time = time.time() - start_time
        logger.info(f"Model training completed successfully in {training_time:.2f} seconds")
        
        # Step 5: Model saving
        ProjectLogger.log_step_header(logger, "STEP", "5: MODEL SAVING")
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")
        
        logger.info(f"Saving trained model to: {model_path}")
        trainer.save_model(model, model_path)
        mlflow.log_artifact(model_path, "01_XGBoost_model")
                    

        
        # Verify model was saved
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Model saved successfully (Size: {model_size:.2f} MB)")
        else:
            raise RuntimeError(f"Failed to save model to {model_path}")
        
        # Step 6: Model evaluation
        ProjectLogger.log_step_header(logger, "STEP", "6: MODEL EVALUATION")
        
        logger.info("Initializing model evaluator...")
        evaluator = ModelEvaluator(model, 'XGBoost')
        eval_results = evaluator.evaluate(X_test, Y_test)
        
        # Log evaluation results
        eval_result_copy = eval_results.copy()
        if 'cm' in eval_result_copy:
            del eval_result_copy['cm']  # Remove confusion matrix for cleaner logging

        eval_result_copy.update({
            'training_time_seconds': training_time,
            'model_complexity': model.n_estimators if hasattr(model, 'n_estimators') else 0,
            'max_depth': model.max_depth if hasattr(model, 'max_depth') else 0
        })

        model_config = get_model_config()['model_params']
        
        training_metrics = {
            "num_features": X_train.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_class_0": (Y_train == 0).sum().iloc[0],
            "train_class_1": (Y_train == 1).sum().iloc[0],
            "test_class_0": (Y_test == 0).sum().iloc[0],
            "test_class_1": (Y_test == 1).sum().iloc[0],
        }

        mlflow_tracker.log_training_metrics(
                                            model, 
                                            training_metrics=training_metrics,
                                            model_params=model_config,
                                            X_train=X_train, Y_train=Y_train, 
                                            X_test=X_test,
                                            )
        
        # Log training summary
        training_summary = {
            'model_type': 'XGboost',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': X_train.shape[1],
            'training_time': training_time,
            'model_path': model_path,
            'timestamp': pd.Timestamp.now().isoformat(),
        }

        # Save training summary
        summary_path = os.path.join(run_artifacts_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        mlflow.log_artifact(summary_path, "training_summary")
        
        logger.info("Model evaluation completed:")
        for metric, value in eval_result_copy.items():
            if isinstance(value, float):
                logger.info(f"  - {metric}: {value:.4f}")
            else:
                logger.info(f"  - {metric}: {value}")

        logger.info("logging evaluation metrics to MLflow")
        eval_metrics = {
            'metrics': eval_result_copy
        }
        logger.info(f"logging evaluation metrics to MLflow {eval_metrics}")
        mlflow_tracker.log_evaluation_metrics(eval_metrics)
        
        mlflow_tracker.end_run()
        
        # Step 7: Training completion
        ProjectLogger.log_success_header(logger, "TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        
        
        logger.info("Training pipeline results prepared successfully")
        
    except FileNotFoundError as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - FILE NOT FOUND")
        logger.error(f"File not found error: {str(e)}")
        logger.error("Please ensure all required data files exist")
        raise
        
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - DATA VALIDATION ERROR")
        logger.error(f"Data validation error: {str(e)}")
        logger.error("Please check data integrity and preprocessing")
        raise
        
    except ImportError as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - MISSING DEPENDENCY")
        logger.error(f"Import error: {str(e)}")
        logger.error("Please ensure all required packages are installed")
        raise
        
    except MemoryError as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - INSUFFICIENT MEMORY")
        logger.error(f"Memory error: {str(e)}")
        logger.error("Consider reducing data size or using a machine with more RAM")
        raise
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - UNEXPECTED ERROR")
        logger.error(f"Unexpected error during training pipeline: {str(e)}")
        logger.error("Training pipeline failed", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting training pipeline execution")
    
    try:
        model_config = get_model_config()
        logger.info(f"Loaded model configuration: {model_config}")
        
        training_pipeline(model_params=model_config)
        ProjectLogger.log_success_header(logger, "TRAINING PIPELINE MAIN EXECUTION COMPLETED")
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE MAIN EXECUTION FAILED")
        raise  