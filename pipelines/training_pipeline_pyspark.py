"""
PySpark Training Pipeline for Telco Customer Churn Prediction.

This module implements a complete machine learning training pipeline using Apache Spark MLlib
for distributed model training and evaluation. It supports multiple algorithms with automatic
hyperparameter tuning and comprehensive performance evaluation.

Key Features:
- Distributed model training using PySpark MLlib
- Multiple ML algorithms (GBT, RandomForest, LogisticRegression, DecisionTree)
- Automatic hyperparameter tuning with cross-validation
- Comprehensive model evaluation and metrics
- MLflow integration for experiment tracking
- Scalable processing for large datasets
- Production-ready model persistence

Supported Algorithms:
- Gradient Boosted Trees (GBT) - Default, best performance
- Random Forest - Good for interpretability
- Logistic Regression - Fast, baseline model
- Decision Tree - Simple, interpretable

Training Pipeline Steps:
1. Data Loading - Load preprocessed data from data pipeline
2. Feature Preparation - Prepare features for ML algorithms
3. Model Training - Train selected algorithm with optimal parameters
4. Hyperparameter Tuning - Optimize model parameters (optional)
5. Model Evaluation - Assess performance with multiple metrics
6. Feature Importance - Extract feature importance scores
7. Model Persistence - Save trained model pipeline
8. MLflow Logging - Track experiments and model artifacts

Dependencies:
- Apache Spark 3.x with MLlib
- PySpark SQL and ML libraries
- MLflow for experiment tracking
- Custom data pipeline modules

Usage:
    >>> from training_pipeline_pyspark import training_pipeline_pyspark
    >>> result = training_pipeline_pyspark(
    ...     algorithm='gbt',
    ...     hyperparameter_tuning=True,
    ...     cross_validation_folds=3
    ... )
    >>> model = result['model']
    >>> metrics = result['metrics']

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

import json
import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_pipeline_pyspark import data_pipeline_pyspark
from model_building import ModelFactory, ModelType
from model_training import create_model_trainer
from model_evaluation import create_model_evaluator
from config import get_model_config, get_data_paths
from logger import get_logger, ProjectLogger, log_exceptions
from spark_utils import get_spark_session
from mlflow_utils import MLflowTracker, create_mlflow_run_tags, setup_mlflow_autolog

# Initialize logger
logger = get_logger(__name__)

# Setup MLflow autologging
setup_mlflow_autolog()


@log_exceptions(logger)
def training_pipeline_pyspark(
    data_path: str = 'data/raw/TelcoCustomerChurnPrediction.csv',
    model_type: str = 'gbt',
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    model_path: str = 'artifacts/models/telco_customer_churn_pyspark_model',
    target_column: str = 'Churn',
    spark: Optional[SparkSession] = None
) -> Dict[str, Any]:
    """
    Execute the complete training pipeline for customer churn prediction using PySpark ML.
    
    Args:
        data_path (str): Path to the raw data file
        model_type (str): Type of model to train ('gbt', 'random_forest', 'logistic_regression', 'decision_tree')
        model_params (Optional[Dict[str, Any]]): Model hyperparameters
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        model_path (str): Path to save the trained model
        target_column (str): Name of the target column
        spark: Optional SparkSession
        
    Returns:
        Dict[str, Any]: Training results and metrics
    """
    
    ProjectLogger.log_section_header(logger, "STARTING PYSPARK TRAINING PIPELINE EXECUTION")
    logger.info(f"Training Parameters:")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Model type: {model_type}")
    logger.info(f"  - Model parameters: {model_params}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Random state: {random_state}")
    logger.info(f"  - Model save path: {model_path}")
    logger.info(f"  - Target column: {target_column}")
    
    try:
        # Initialize Spark session
        spark = spark or get_spark_session()
        logger.info(f"Using Spark session: {spark.sparkContext.appName}")
        
        # Initialize model parameters if not provided
        if model_params is None:
            model_params = {}
            logger.info("Using default model parameters")

        # Create artifacts directory for this run
        run_artifacts_dir = os.path.join('artifacts', 'pyspark_training_artifacts', 
                                       f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_artifacts_dir, exist_ok=True)
        
        """
        Step 1: Data Preparation
        """
        ProjectLogger.log_step_header(logger, "STEP", "1: DATA PREPARATION")
        
        # Check if preprocessed data exists, otherwise run data pipeline
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'data'))
        parquet_train_path = os.path.join(artifacts_dir, 'train_data.parquet')
        parquet_test_path = os.path.join(artifacts_dir, 'test_data.parquet')
        
        if os.path.exists(parquet_train_path) and os.path.exists(parquet_test_path):
            logger.info("Preprocessed Parquet data found. Loading existing data...")
            train_df = spark.read.parquet(parquet_train_path)
            test_df = spark.read.parquet(parquet_test_path)
            logger.info("Existing data loaded successfully")
        else:
            logger.info("Preprocessed data not found. Running data pipeline...")
            pipeline_result = data_pipeline_pyspark(
                data_path=data_path,
                target_column=target_column,
                test_size=test_size,
                spark=spark
            )
            train_df = pipeline_result['train_df']
            test_df = pipeline_result['test_df']
            logger.info("Data pipeline executed successfully")
        
        # Cache DataFrames for performance
        train_df.cache()
        test_df.cache()
        
        # Get data statistics
        train_count = train_df.count()
        test_count = test_df.count()
        feature_columns = [col for col in train_df.columns if col != target_column]
        num_features = len(feature_columns)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  - Training samples: {train_count}")
        logger.info(f"  - Test samples: {test_count}")
        logger.info(f"  - Number of features: {num_features}")
        logger.info(f"  - Feature columns: {feature_columns[:5]}..." if len(feature_columns) > 5 else f"  - Feature columns: {feature_columns}")
        
        # Check target distribution
        logger.info("Target distribution in training data:")
        train_target_dist = train_df.groupBy(target_column).count().collect()
        for row in train_target_dist:
            count = row['count']
            percentage = (count / train_count) * 100
            logger.info(f"  - {row[target_column]}: {count} samples ({percentage:.2f}%)")
        
        """
        Step 2: Model Building
        """
        ProjectLogger.log_step_header(logger, "STEP", "2: MODEL BUILDING")
        
        logger.info(f"Building {model_type} model with parameters: {model_params}")
        
        # Map string model type to enum
        model_type_mapping = {
            'gbt': ModelType.GBT,
            'random_forest': ModelType.RANDOM_FOREST,
            'logistic_regression': ModelType.LOGISTIC_REGRESSION,
            'decision_tree': ModelType.DECISION_TREE
        }
        
        if model_type not in model_type_mapping:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(model_type_mapping.keys())}")
        
        model_factory = ModelFactory(spark=spark)
        model = model_factory.create_model(model_type_mapping[model_type], **model_params)
        logger.info(f"Model built successfully: {type(model).__name__}")
        
        """
        Step 3: Model Training
        """
        ProjectLogger.log_step_header(logger, "STEP", "3: MODEL TRAINING")
        
        logger.info("Initializing PySpark model trainer...")
        trainer = create_model_trainer(spark=spark)
        
        import time
        logger.info("Starting model training...")
        start_time = time.time()
        
        # Train the model
        pipeline_model, training_metrics = trainer.train(
            model=model,
            train_df=train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_save_path=model_path
        )
        
        training_time = time.time() - start_time
        logger.info(f"Model training completed successfully in {training_time:.2f} seconds")
        
        # Add training time to metrics
        training_metrics['training_time_seconds'] = training_time
        
        """
        Step 4: Model Evaluation
        """
        ProjectLogger.log_step_header(logger, "STEP", "4: MODEL EVALUATION")
        
        logger.info("Initializing PySpark model evaluator...")
        evaluator = create_model_evaluator(
            model=pipeline_model,
            model_name=f"PySpark_{model_type}",
            spark=spark
        )
        
        # Evaluate model on test data
        eval_results = evaluator.evaluate(test_df, target_column)
        
        logger.info("Model evaluation completed:")
        for metric, value in eval_results.items():
            if isinstance(value, float):
                logger.info(f"  - {metric}: {value:.4f}")
            elif metric not in ['confusion_matrix', 'per_class_metrics']:  # Skip complex objects for logging
                logger.info(f"  - {metric}: {value}")
        
        """
        Step 4.5: MLflow Logging
        """
        ProjectLogger.log_step_header(logger, "STEP", "4.5: MLFLOW LOGGING")
        
        try:
            # Initialize MLflow tracker
            mlflow_tracker = MLflowTracker()
            
            # Create run tags
            run_tags = create_mlflow_run_tags(
                pipeline_type="pyspark_training",
                engine="pyspark_ml",
                additional_tags={
                    'model_type': model_type,
                    'spark_version': spark.version,
                    'dataset_rows': str(train_count + test_count)
                }
            )
            
            # Start MLflow run
            run_name = f"PySpark {model_type} Training"
            with mlflow_tracker.start_run(run_name=run_name, tags=run_tags):
                
                # Log data pipeline information
                data_info = {
                    'total_rows': train_count + test_count,
                    'train_rows': train_count,
                    'test_rows': test_count,
                    'num_features': num_features,
                    'test_size': test_size,
                    'train_df': train_df,
                    'test_df': test_df,
                    'source': data_path,
                    'target_column': target_column,
                    'feature_encoding_applied': True,
                    'feature_scaling_applied': True,
                    'executor_memory': spark.conf.get('spark.executor.memory', 'unknown'),
                    'driver_memory': spark.conf.get('spark.driver.memory', 'unknown')
                }
                
                mlflow_tracker.log_pyspark_data_pipeline_metrics(data_info)
                
                # Prepare metrics for MLflow
                combined_metrics = {}
                combined_metrics.update(training_metrics)
                combined_metrics.update(eval_results)
                
                # Remove non-numeric metrics for MLflow
                numeric_metrics = {}
                for key, value in combined_metrics.items():
                    if isinstance(value, (int, float)) and key not in ['confusion_matrix', 'per_class_metrics']:
                        numeric_metrics[key] = value
                
                # Log PySpark model
                mlflow_tracker.log_pyspark_model(
                    model=pipeline_model,
                    training_metrics=numeric_metrics,
                    model_params=model_params,
                    train_df=train_df,
                    test_df=test_df,
                    model_name=f"pyspark_{model_type}_churn_model"
                )
                
                logger.info("✓ Model and metrics logged to MLflow successfully")
                
        except Exception as mlflow_error:
            logger.warning(f"MLflow logging failed: {str(mlflow_error)}")
            logger.warning("Continuing without MLflow logging...")
        
        """
        Step 5: Results Compilation
        """
        ProjectLogger.log_step_header(logger, "STEP", "5: RESULTS COMPILATION")
        
        # Compile comprehensive results
        results = {
            'model_info': {
                'model_type': model_type,
                'model_params': model_params,
                'model_path': model_path,
                'pipeline_stages': len(pipeline_model.stages)
            },
            'data_info': {
                'train_samples': train_count,
                'test_samples': test_count,
                'num_features': num_features,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'test_size': test_size
            },
            'training_metrics': training_metrics,
            'evaluation_metrics': eval_results,
            'execution_info': {
                'training_time_seconds': training_time,
                'timestamp': datetime.now().isoformat(),
                'spark_app_name': spark.sparkContext.appName,
                'spark_version': spark.version
            }
        }
        
        # Save training summary
        summary_path = os.path.join(run_artifacts_dir, 'pyspark_training_summary.json')
        with open(summary_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Save feature importance if available
        try:
            if hasattr(pipeline_model.stages[-1], 'featureImportances'):
                feature_importance = pipeline_model.stages[-1].featureImportances.toArray()
                importance_dict = dict(zip(feature_columns, feature_importance))
                
                # Sort by importance
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                logger.info("Top 10 most important features:")
                for i, (feature, importance) in enumerate(sorted_importance[:10]):
                    logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                
                # Save feature importance
                importance_path = os.path.join(run_artifacts_dir, 'feature_importance.json')
                with open(importance_path, 'w') as f:
                    json.dump(dict(sorted_importance), f, indent=2)
                
                results['feature_importance'] = dict(sorted_importance)
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        # Clean up cache
        train_df.unpersist()
        test_df.unpersist()
        
        ProjectLogger.log_success_header(logger, "PYSPARK TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        
        logger.info("Training pipeline results summary:")
        logger.info(f"  - Model type: {model_type}")
        logger.info(f"  - Training accuracy: {training_metrics.get('accuracy', 'N/A'):.4f}" if isinstance(training_metrics.get('accuracy'), float) else f"  - Training accuracy: {training_metrics.get('accuracy', 'N/A')}")
        logger.info(f"  - Test accuracy: {eval_results.get('accuracy', 'N/A'):.4f}" if isinstance(eval_results.get('accuracy'), float) else f"  - Test accuracy: {eval_results.get('accuracy', 'N/A')}")
        logger.info(f"  - Training time: {training_time:.2f} seconds")
        logger.info(f"  - Model saved to: {model_path}")
        
        return results
        
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
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE FAILED - UNEXPECTED ERROR")
        logger.error(f"Unexpected error during training pipeline: {str(e)}")
        logger.error("Training pipeline failed", exc_info=True)
        raise


def training_pipeline(
    data_path: str = 'data/raw/TelcoCustomerChurnPrediction.csv',
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    model_path: str = 'artifacts/models/telco_customer_churn_prediction.joblib'
) -> Dict[str, Any]:
    """
    Legacy function wrapper for backward compatibility.
    """
    # Convert legacy model path to PySpark model path
    if model_path.endswith('.joblib'):
        model_path = model_path.replace('.joblib', '_pyspark_model')
    
    return training_pipeline_pyspark(
        data_path=data_path,
        model_type='gbt',  # Default to GBT for backward compatibility
        model_params=model_params,
        test_size=test_size,
        random_state=random_state,
        model_path=model_path
    )


def compare_models_pyspark(
    data_path: str = 'data/raw/TelcoCustomerChurnPrediction.csv',
    model_types: List[str] = ['gbt', 'random_forest', 'logistic_regression'],
    test_size: float = 0.2,
    spark: Optional[SparkSession] = None
) -> Dict[str, Any]:
    """
    Compare multiple PySpark ML models and return results.
    
    Args:
        data_path (str): Path to the raw data file
        model_types (List[str]): List of model types to compare
        test_size (float): Proportion of data for testing
        spark: Optional SparkSession
        
    Returns:
        Dict[str, Any]: Comparison results for all models
    """
    ProjectLogger.log_section_header(logger, "STARTING PYSPARK MODEL COMPARISON")
    
    comparison_results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        
        try:
            model_path = f'artifacts/models/telco_churn_{model_type}_model'
            result = training_pipeline_pyspark(
                data_path=data_path,
                model_type=model_type,
                test_size=test_size,
                model_path=model_path,
                spark=spark
            )
            comparison_results[model_type] = result
            logger.info(f"{model_type} model training completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {str(e)}")
            comparison_results[model_type] = {'error': str(e)}
    
    # Generate comparison summary
    summary = {
        'comparison_timestamp': datetime.now().isoformat(),
        'models_compared': model_types,
        'results': comparison_results
    }
    
    # MLflow logging for model comparison
    try:
        mlflow_tracker = MLflowTracker()
        
        # Find best model
        best_model = None
        best_accuracy = 0
        for model_type, result in comparison_results.items():
            if 'error' not in result and 'evaluation_metrics' in result:
                accuracy = result['evaluation_metrics'].get('accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_type
        
        if best_model:
            # Create comparison run tags
            comparison_tags = create_mlflow_run_tags(
                pipeline_type="pyspark_model_comparison",
                engine="pyspark_ml",
                additional_tags={
                    'best_model': best_model,
                    'models_compared': ','.join(model_types),
                    'comparison_type': 'accuracy_based'
                }
            )
            
            # Start MLflow run for comparison
            with mlflow_tracker.start_run(run_name="PySpark Model Comparison", tags=comparison_tags):
                
                # Prepare comparison metrics
                comparison_metrics = {
                    'best_accuracy': best_accuracy,
                    'models_trained': len([r for r in comparison_results.values() if 'error' not in r]),
                    'models_failed': len([r for r in comparison_results.values() if 'error' in r])
                }
                
                # Log comparison results
                mlflow_tracker.log_pyspark_training_comparison(
                    models_results=comparison_results,
                    best_model_name=best_model,
                    comparison_metrics=comparison_metrics
                )
                
                logger.info("✓ Model comparison results logged to MLflow successfully")
        
    except Exception as mlflow_error:
        logger.warning(f"MLflow comparison logging failed: {str(mlflow_error)}")
        logger.warning("Continuing without MLflow comparison logging...")
    
    # Save comparison results
    comparison_dir = os.path.join('artifacts', 'model_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_path = os.path.join(comparison_dir, 'pyspark_model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    ProjectLogger.log_success_header(logger, "PYSPARK MODEL COMPARISON COMPLETED")
    
    return summary


if __name__ == "__main__":
    logger.info("Starting PySpark training pipeline execution")
    
    try:
        # Load model configuration
        model_config = get_model_config()
        logger.info(f"Loaded model configuration: {model_config}")
        
        # Execute training pipeline with GBT model
        results = training_pipeline_pyspark(
            model_type='gbt',
            model_params=model_config.get('model_params', {})
        )
        
        logger.info("Training pipeline execution completed successfully")
        logger.info(f"Final test accuracy: {results['evaluation_metrics'].get('accuracy', 'N/A')}")
        
        ProjectLogger.log_success_header(logger, "TRAINING PIPELINE MAIN EXECUTION COMPLETED")
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "TRAINING PIPELINE MAIN EXECUTION FAILED")
        logger.error(f"Error: {str(e)}")
        raise