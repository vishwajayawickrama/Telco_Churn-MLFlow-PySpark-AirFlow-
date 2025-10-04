import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.spark
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.pandas_dataset import from_pandas
from mlflow.data.spark_dataset import from_spark
from typing import Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from config import get_mlflow_config

# PySpark imports for MLflow support
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.ml import Model as SparkModel
    from pyspark.ml.base import Transformer
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow tracking utilities for experiment management and model versioning"""
    
    def __init__(self):
        self.config = get_mlflow_config()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Initialize MLflow tracking with configuration"""
        tracking_uri = self.config.get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = self.config.get('experiment_name', 'telco_customer_churn_experiment')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
                
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        # Format timestamp for run name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if run_name is None:
            run_name_prefix = self.config.get('run_name_prefix', 'run')
            # Remove underscores and format with timestamp
            run_name_prefix = run_name_prefix.replace('_', ' ')
            run_name = f"{run_name_prefix} | {timestamp}"
        else:
            # Remove underscores from provided run name and append timestamp
            run_name = run_name.replace('_', ' ')
            run_name = f"{run_name} | {timestamp}"
        
        # Merge default tags with provided tags
        default_tags = self.config.get('tags', {})
        if tags:
            default_tags.update(tags)
            
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        print(f"🎯 MLflow Run Name: {run_name}")
        return run
    
    def log_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        """Log data pipeline metrics, parameters, and dataset artifacts to MLflow"""
        try:
            # Log dataset metrics (Metrics = Things in dataset that can be measured numerically)
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
                'missing_values_count': dataset_info.get('missing_values', 0),
                'outliers_removed': dataset_info.get('outliers_removed', 0)
            })
            
            # Log dataset parameters (Parameters = Things that are set before running the experiment)
            mlflow.log_params({
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42),
                'missing_value_strategy': dataset_info.get('missing_strategy', 'unknown'),
                'outlier_detection_method': dataset_info.get('outlier_method', 'unknown'),
                'feature_encoding_applied': dataset_info.get('encoding_applied', False),
                'feature_scaling_applied': dataset_info.get('scaling_applied', False)
            })
            
            # Log feature names
            if 'feature_names' in dataset_info:
                mlflow.log_param('feature_names', str(dataset_info['feature_names']))
            
            # Log dataset to MLflow if dataset is provided
            if 'processed_dataset' in dataset_info and dataset_info['processed_dataset'] is not None:
                logger.info("Logging processed dataset to MLflow...")
                try:
                    dataset = dataset_info['processed_dataset']
                    
                    # Convert to DataFrame if needed
                    if hasattr(dataset, 'to_pandas'):
                        dataset_df = dataset.copy()
                        logger.debug("Dataset is already a DataFrame")
                    elif hasattr(dataset, 'values'):
                        dataset_df = dataset.copy()
                        logger.debug("Dataset converted from DataFrame with values")
                    else:
                        dataset_df = pd.DataFrame(dataset)
                        logger.debug("Dataset converted from array-like structure")
                    
                    # Generate dataset name with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dataset_name = dataset_info.get('dataset_name', f"telco_churn_processed_data_{timestamp}")
                    
                    # Create MLflow dataset object
                    mlflow_dataset = from_pandas(
                        dataset_df,
                        source=dataset_info.get('source', 'data_pipeline'),
                        name=dataset_name
                    )
                    
                    # Log the dataset with context
                    context = dataset_info.get('context', 'data_preprocessing')
                    mlflow.log_input(mlflow_dataset, context=context)
                    
                    logger.info(f"Successfully logged dataset '{dataset_name}' to MLflow")
                    logger.info(f"Dataset shape: {dataset_df.shape}")
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to log dataset to MLflow: {str(dataset_error)}")
                    logger.warning("Continuing data pipeline metrics logging without dataset...")
            else:
                logger.info("No processed dataset provided - skipping dataset logging")
            
            # Log training and test datasets separately if provided
            if 'X_train' in dataset_info and 'Y_train' in dataset_info:
                logger.info("Logging training dataset to MLflow...")
                try:
                    X_train = dataset_info['X_train']
                    Y_train = dataset_info['Y_train']
                    
                    # Convert to DataFrame format
                    if hasattr(X_train, 'copy'):
                        train_df = X_train.copy()
                    else:
                        train_df = pd.DataFrame(X_train)
                    
                    # Add target variable
                    if hasattr(Y_train, 'values'):
                        if hasattr(Y_train.values, 'ravel'):
                            train_df['target'] = Y_train.values.ravel()
                        else:
                            train_df['target'] = Y_train.values
                    elif hasattr(Y_train, 'ravel'):
                        train_df['target'] = Y_train.ravel()
                    else:
                        train_df['target'] = Y_train
                    
                    # Generate training dataset name
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    train_dataset_name = f"telco_churn_training_split_{timestamp}"
                    
                    # Create and log training dataset
                    train_dataset = from_pandas(
                        train_df,
                        source="data_pipeline_training_split",
                        name=train_dataset_name
                    )
                    
                    mlflow.log_input(train_dataset, context="training_split")
                    logger.info(f"Successfully logged training dataset: {train_dataset_name}")
                    
                except Exception as train_error:
                    logger.error(f"Failed to log training dataset: {str(train_error)}")
            
            
        except Exception as e:
            logger.error(f"Error logging data pipeline metrics: {e}")
            logger.error("Data pipeline metrics logging failed", exc_info=True)
    
    def log_training_metrics(
            self, 
            model, 
            training_metrics: Dict[str, Any], 
            model_params: Dict[str, Any],
            X_train=None, 
            Y_train=None, 
            X_test=None, 
            dataset_name: str = None,
            ):
        """
        Log comprehensive training metrics, parameters, model artifacts, and training dataset to MLflow.
        
        Args:
            model: Trained machine learning model
            training_metrics: Dictionary of training performance metrics
            model_params: Model hyperparameters and configuration
            X_train: Training features
            Y_train: Training target variable
            X_test: Test features for signature inference
            dataset_name: Name for the training dataset in MLflow
        """
        logger.info("Starting comprehensive training metrics logging to MLflow")
        
        try:

            # Step 1: Log model parameters
            mlflow.log_params(model_params)

            # Step 2: Log training metrics
            mlflow.log_metrics(training_metrics)

            # Step 3: Log training dataset
            if X_train is not None and Y_train is not None:
                try:
                    if hasattr(X_train, 'to_pandas'):  # Already a DataFrame
                        train_df = X_train.copy()
                    elif hasattr(X_train, 'values'):  # DataFrame with values attribute
                        train_df = X_train.copy()
                    else: 
                        train_df = pd.DataFrame(X_train)

                    # Add target variable to the training DataFrame
                    if hasattr(Y_train, 'values'):
                        if hasattr(Y_train.values, 'ravel'):
                            train_df['target'] = Y_train.values.ravel()
                        else:
                            train_df['target'] = Y_train.values
                    elif hasattr(Y_train, 'ravel'):
                        train_df['target'] = Y_train.ravel()
                    else:
                        train_df['target'] = Y_train

                    # Generate dataset name if not provided
                    if dataset_name is None:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dataset_name = f"telco_churn_training_data_{timestamp}"
                    
                    # Create MLflow dataset object
                    logger.debug("Creating MLflow dataset object...")
                    dataset = from_pandas(
                        train_df, 
                        source="training_pipeline",
                        name=dataset_name
                    )

                    # Log the dataset to MLflow
                    mlflow.log_input(dataset, context="training")
                    logger.info(f"Successfully logged training dataset: {dataset_name}")
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to log training dataset: {str(dataset_error)}")
                    logger.warning("Continuing without dataset logging...")
            else:
                logger.warning("Training data not provided - skipping dataset logging")

            # Step 4: Create model signature
            signature = None
            if X_test is not None:
                try:
                    from mlflow.models.signature import infer_signature
                    
                    # Use a small sample for signature inference
                    sample_size = min(5, len(X_test)) if hasattr(X_test, '__len__') else 5
                    X_sample = X_test[:sample_size] if hasattr(X_test, '__getitem__') else X_test
                    sample_predictions = model.predict(X_sample)
                    signature = infer_signature(X_sample, sample_predictions)
                    
                except Exception as sig_error:
                    logger.warning(f"Failed to create model signature: {str(sig_error)}")
                    logger.warning("Model will be logged without signature")
            else:
                logger.info("Test data not provided - skipping signature creation")

            # Step 5: Log the trained model
            logger.info("Logging trained model to MLflow...")
            try:
                artifact_path = self.config.get('artifact_path', 'model')
                model_registry_name = self.config.get('model_registry_name', 'churn_prediction_model')
                
                try:
                    mlflow.xgboost.log_model(
                                xgb_model=model,
                                artifact_path=artifact_path,
                                signature=signature,
                                registered_model_name=model_registry_name,
                            )
                    logger.info("Model logged as XGBoost model")
                except Exception:
                    mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=artifact_path,
                            signature=signature,
                            registered_model_name=model_registry_name
                            )
                    logger.info("Model logged as scikit-learn model")
                
            except Exception as model_error:
                logger.error(f"Failed to log model: {str(model_error)}")
                raise
            
            # Step 6: Log additional model metadata
            try:
                model_metadata = {
                    'model_type': type(model).__name__,
                    'model_framework': 'scikit-learn' if hasattr(model, 'fit') else 'unknown',
                    'has_predict_proba': hasattr(model, 'predict_proba'),
                    'training_timestamp': datetime.now().isoformat()
                }
                
                mlflow.log_params(model_metadata)
                logger.info("Successfully logged model metadata")
                
            except Exception as metadata_error:
                logger.warning(f"Failed to log model metadata: {str(metadata_error)}")
            
            logger.info("Training metrics logging completed successfully")
            
        except Exception as e:
            logger.error(f"Critical error in log_training_metrics: {str(e)}")
            logger.error("Training metrics logging failed", exc_info=True)
            raise
    
    def log_evaluation_metrics(self, evaluation_metrics: Dict[str, Any], confusion_matrix_path: Optional[str] = None):
        """Log evaluation metrics and artifacts"""
        try:
            # Log evaluation metrics
            if 'metrics' in evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics['metrics'])
            
            # Log confusion matrix if provided
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path, "evaluation")
            
            logger.info("Logged evaluation metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging evaluation metrics: {e}")
    
    def log_inference_metrics(self, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None, 
                            input_data_info: Optional[Dict[str, Any]] = None):
        """Log inference metrics and results"""
        try:
            # Log inference metrics
            inference_metrics = {
                'num_predictions': len(predictions),
                'avg_prediction': float(np.mean(predictions)),
                'prediction_distribution_churn': int(np.sum(predictions)),
                'prediction_distribution_retain': int(len(predictions) - np.sum(predictions))
            }
            
            if probabilities is not None:
                inference_metrics.update({
                    'avg_churn_probability': float(np.mean(probabilities)),
                    'high_risk_predictions': int(np.sum(probabilities > 0.7)),
                    'medium_risk_predictions': int(np.sum((probabilities > 0.5) & (probabilities <= 0.7))),
                    'low_risk_predictions': int(np.sum(probabilities <= 0.5))
                })
            
            mlflow.log_metrics(inference_metrics)
            
            # Log input data info if provided
            if input_data_info:
                mlflow.log_params(input_data_info)
            
            logger.info("Logged inference metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging inference metrics: {e}")
    
    def load_model_from_registry(self, model_name: Optional[str] = None, 
                               version: Optional[Union[int, str]] = None, 
                               stage: Optional[str] = None):
        """Load model from MLflow Model Registry"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow registry: {e}")
            return None
    
    def get_latest_model_version(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the latest version of a registered model"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            
            if latest_version:
                return latest_version[0].version
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    def transition_model_stage(self, model_name: Optional[str] = None, 
                             version: Optional[str] = None, 
                             stage: str = "Staging"):
        """Transition model to a specific stage"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            if version is None:
                version = self.get_latest_model_version(model_name)
            
            if version:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                logger.info(f"Transitioned model {model_name} version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    # PySpark MLflow Support Methods
    def log_pyspark_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        """Log PySpark data pipeline metrics and datasets to MLflow"""
        if not PYSPARK_AVAILABLE:
            logger.warning("PySpark not available. Falling back to pandas logging.")
            return self.log_data_pipeline_metrics(dataset_info)
        
        try:
            # Log dataset metrics
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
                'missing_values_count': dataset_info.get('missing_values', 0),
                'outliers_removed': dataset_info.get('outliers_removed', 0)
            })
            
            # Log dataset parameters
            mlflow.log_params({
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42),
                'missing_value_strategy': dataset_info.get('missing_strategy', 'unknown'),
                'outlier_detection_method': dataset_info.get('outlier_method', 'unknown'),
                'feature_encoding_applied': dataset_info.get('encoding_applied', False),
                'feature_scaling_applied': dataset_info.get('scaling_applied', False),
                'engine': 'pyspark',
                'spark_executor_memory': dataset_info.get('executor_memory', 'unknown'),
                'spark_driver_memory': dataset_info.get('driver_memory', 'unknown')
            })
            
            # Log PySpark DataFrame datasets
            if 'train_df' in dataset_info and isinstance(dataset_info['train_df'], SparkDataFrame):
                logger.info("Logging PySpark training dataset to MLflow...")
                try:
                    train_df = dataset_info['train_df']
                    
                    # Create MLflow Spark dataset
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dataset_name = f"pyspark_train_data_{timestamp}"
                    
                    mlflow_dataset = from_spark(
                        train_df,
                        source=dataset_info.get('source', 'pyspark_data_pipeline'),
                        name=dataset_name
                    )
                    
                    mlflow.log_input(mlflow_dataset, context="training")
                    logger.info(f"Successfully logged PySpark training dataset '{dataset_name}' to MLflow")
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to log PySpark training dataset: {str(dataset_error)}")
            
            if 'test_df' in dataset_info and isinstance(dataset_info['test_df'], SparkDataFrame):
                logger.info("Logging PySpark test dataset to MLflow...")
                try:
                    test_df = dataset_info['test_df']
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dataset_name = f"pyspark_test_data_{timestamp}"
                    
                    mlflow_dataset = from_spark(
                        test_df,
                        source=dataset_info.get('source', 'pyspark_data_pipeline'),
                        name=dataset_name
                    )
                    
                    mlflow.log_input(mlflow_dataset, context="testing")
                    logger.info(f"Successfully logged PySpark test dataset '{dataset_name}' to MLflow")
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to log PySpark test dataset: {str(dataset_error)}")
            
            logger.info("PySpark data pipeline metrics logged successfully to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging PySpark data pipeline metrics to MLflow: {e}")
            raise
    
    def log_pyspark_model(self, 
                         model: Union[SparkModel, Transformer],
                         training_metrics: Dict[str, Any],
                         model_params: Dict[str, Any],
                         train_df: Optional[SparkDataFrame] = None,
                         test_df: Optional[SparkDataFrame] = None,
                         model_name: Optional[str] = None):
        """
        Log PySpark ML model and metrics to MLflow.
        
        Args:
            model: Trained PySpark ML model or Pipeline
            training_metrics: Dictionary of training performance metrics
            model_params: Model hyperparameters and configuration
            train_df: Training PySpark DataFrame
            test_df: Test PySpark DataFrame  
            model_name: Name for model registration
        """
        if not PYSPARK_AVAILABLE:
            logger.warning("PySpark not available. Cannot log PySpark model.")
            return
        
        try:
            logger.info("Logging PySpark ML model to MLflow")
            
            # Log model parameters
            mlflow.log_params(model_params)
            
            # Log training metrics
            mlflow.log_metrics(training_metrics)
            
            # Log PySpark model
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'pyspark_churn_model')
            
            # Create model signature from training data
            signature = None
            if train_df is not None:
                try:
                    # Sample a small subset for signature inference
                    sample_df = train_df.limit(100).toPandas()
                    feature_columns = [col for col in sample_df.columns if col != 'Churn']
                    
                    from mlflow.models.signature import infer_signature
                    
                    # Make prediction on sample to infer signature
                    sample_spark_df = train_df.limit(100)
                    predictions = model.transform(sample_spark_df)
                    predictions_pandas = predictions.select('prediction').toPandas()
                    
                    signature = infer_signature(
                        sample_df[feature_columns], 
                        predictions_pandas['prediction']
                    )
                    logger.info("Model signature inferred successfully")
                    
                except Exception as sig_error:
                    logger.warning(f"Could not infer model signature: {sig_error}")
            
            # Log the PySpark model
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="pyspark_model",
                signature=signature,
                registered_model_name=model_name
            )
            
            logger.info(f"PySpark model logged successfully as '{model_name}'")
            
            # Log additional model artifacts
            if hasattr(model, 'stages'):
                # This is a Pipeline, log stage information
                stage_info = []
                for i, stage in enumerate(model.stages):
                    stage_info.append({
                        'stage_index': i,
                        'stage_type': type(stage).__name__,
                        'stage_params': stage.extractParamMap()
                    })
                
                mlflow.log_dict(stage_info, "pipeline_stages.json")
                logger.info("Pipeline stage information logged")
            
            # Log feature importance if available
            if hasattr(model, 'featureImportances'):
                try:
                    importance_dict = {}
                    for i, importance in enumerate(model.featureImportances):
                        importance_dict[f'feature_{i}'] = float(importance)
                    
                    mlflow.log_dict(importance_dict, "feature_importance.json")
                    logger.info("Feature importance logged")
                    
                except Exception as imp_error:
                    logger.warning(f"Could not log feature importance: {imp_error}")
            
        except Exception as e:
            logger.error(f"Error logging PySpark model to MLflow: {e}")
            raise
    
    def log_pyspark_training_comparison(self, 
                                       models_results: Dict[str, Dict],
                                       best_model_name: str,
                                       comparison_metrics: Dict[str, Any]):
        """
        Log PySpark model comparison results to MLflow.
        
        Args:
            models_results: Dictionary containing model results
            best_model_name: Name of the best performing model
            comparison_metrics: Overall comparison metrics
        """
        try:
            logger.info("Logging PySpark model comparison results to MLflow")
            
            # Log comparison metrics
            mlflow.log_metrics(comparison_metrics)
            
            # Log best model information
            mlflow.log_params({
                'best_model': best_model_name,
                'total_models_trained': len(models_results),
                'engine': 'pyspark_ml'
            })
            
            # Log detailed model comparison
            comparison_data = {}
            for model_name, results in models_results.items():
                if 'metrics' in results:
                    for metric_name, metric_value in results['metrics'].items():
                        comparison_data[f"{model_name}_{metric_name}"] = metric_value
            
            mlflow.log_dict(comparison_data, "model_comparison.json")
            
            # Log model rankings
            if 'accuracy' in comparison_metrics:
                model_rankings = []
                for model_name, results in models_results.items():
                    if 'metrics' in results and 'accuracy' in results['metrics']:
                        model_rankings.append({
                            'model': model_name,
                            'accuracy': results['metrics']['accuracy'],
                            'f1': results['metrics'].get('f1', 0),
                            'auc': results['metrics'].get('auc', 0)
                        })
                
                # Sort by accuracy
                model_rankings.sort(key=lambda x: x['accuracy'], reverse=True)
                mlflow.log_dict(model_rankings, "model_rankings.json")
            
            logger.info("PySpark model comparison logged successfully")
            
        except Exception as e:
            logger.error(f"Error logging PySpark model comparison: {e}")
            raise
    
    def load_pyspark_model(self, 
                          model_name: Optional[str] = None, 
                          version: Optional[str] = None,
                          stage: Optional[str] = None) -> Optional[SparkModel]:
        """
        Load a PySpark ML model from MLflow registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from (e.g., 'Production', 'Staging')
            
        Returns:
            Loaded PySpark ML model or None if failed
        """
        if not PYSPARK_AVAILABLE:
            logger.warning("PySpark not available. Cannot load PySpark model.")
            return None
        
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'pyspark_churn_model')
            
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            logger.info(f"Loading PySpark model from: {model_uri}")
            
            model = mlflow.spark.load_model(model_uri)
            logger.info(f"Successfully loaded PySpark model from MLflow registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading PySpark model from MLflow registry: {e}")
            return None

def setup_mlflow_autolog():
    """Setup MLflow autologging for supported frameworks"""
    mlflow_config = get_mlflow_config()
    if mlflow_config.get('autolog', True):
        # Enable sklearn autologging
        mlflow.sklearn.autolog()
        logger.info("MLflow autologging enabled for scikit-learn")
        
        # Enable PySpark autologging if available
        if PYSPARK_AVAILABLE:
            try:
                mlflow.spark.autolog()
                logger.info("MLflow autologging enabled for PySpark")
            except Exception as e:
                logger.warning(f"Could not enable PySpark autologging: {e}")

def create_mlflow_run_tags(pipeline_type: str, 
                          engine: str = "sklearn",
                          additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Create standardized tags for MLflow runs.
    
    Args:
        pipeline_type: Type of pipeline (training, inference, etc.)
        engine: ML engine being used (sklearn, pyspark, etc.)
        additional_tags: Additional custom tags
        
    Returns:
        Dictionary of MLflow tags
    """
    tags = {
        'pipeline_type': pipeline_type,
        'engine': engine,
        'project': 'telco_customer_churn',
        'timestamp': datetime.now().isoformat(),
        'environment': os.getenv("ENVIRONMENT", "development")
    }
    
    if additional_tags:
        tags.update(additional_tags)
    
    return tags
