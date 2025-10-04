"""
Airflow Utilities for Telco Churn ML Pipeline.

This module contains common utilities and helper functions used across
multiple DAGs in the Telco Customer Churn ML pipeline.

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

import os
import sys
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirflowMLPipelineUtils:
    """
    Utility class for ML pipeline operations in Airflow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize utility class with configuration.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate that all required components are available.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'spark_available': False,
            'mlflow_available': False,
            'data_available': False,
            'directories_exist': False
        }
        
        try:
            # Check Spark availability
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                spark.stop()
                validation_results['spark_available'] = True
            except Exception as e:
                self.logger.warning(f"Spark not available: {e}")
            
            # Check MLflow availability
            try:
                import mlflow
                validation_results['mlflow_available'] = True
            except Exception as e:
                self.logger.warning(f"MLflow not available: {e}")
            
            # Check data availability
            if os.path.exists(self.config.get('data_path', '')):
                validation_results['data_available'] = True
            
            # Check directories
            required_dirs = [
                self.config.get('model_path', ''),
                self.config.get('results_path', ''),
                '/opt/airflow/artifacts',
                '/opt/airflow/data'
            ]
            
            all_dirs_exist = all(os.path.exists(os.path.dirname(d)) for d in required_dirs if d)
            validation_results['directories_exist'] = all_dirs_exist
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
        
        return validation_results
    
    def setup_mlflow_experiment(self, experiment_name: str) -> str:
        """
        Setup MLflow experiment for tracking.
        
        Args:
            experiment_name: Name of the MLflow experiment
            
        Returns:
            Experiment ID
        """
        try:
            import mlflow
            
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(experiment_name)
                self.logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            # Set experiment
            mlflow.set_experiment(experiment_name)
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def create_run_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for the current pipeline run.
        
        Args:
            context: Airflow context
            
        Returns:
            Run metadata dictionary
        """
        return {
            'dag_id': context.get('dag').dag_id,
            'task_id': context.get('task').task_id,
            'execution_date': context.get('ds'),
            'run_id': context.get('run_id'),
            'try_number': context.get('task_instance').try_number,
            'hostname': os.uname().nodename,
            'user': os.environ.get('USER', 'airflow'),
            'pipeline_version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def log_pipeline_metrics(self, context: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Log pipeline metrics to MLflow.
        
        Args:
            context: Airflow context
            metrics: Metrics to log
        """
        try:
            import mlflow
            
            # Create run metadata
            run_metadata = self.create_run_metadata(context)
            
            with mlflow.start_run(run_name=f"{context['task'].task_id}_{context['ds']}"):
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))
                
                # Log metadata
                for key, value in run_metadata.items():
                    mlflow.log_param(f"run_{key}", value)
                
                self.logger.info(f"Logged {len(metrics)} metrics to MLflow")
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def save_task_results(self, task_id: str, results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Save task results to file system.
        
        Args:
            task_id: Task identifier
            results: Results to save
            context: Airflow context
            
        Returns:
            Path to saved results
        """
        try:
            # Create results directory
            results_dir = Path(self.config['results_path']) / context['ds'] / context['run_id']
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_file = results_dir / f"{task_id}_results.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved task results to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save task results: {e}")
            raise
    
    def load_task_results(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load task results from file system.
        
        Args:
            task_id: Task identifier
            context: Airflow context
            
        Returns:
            Loaded results
        """
        try:
            results_dir = Path(self.config['results_path']) / context['ds'] / context['run_id']
            results_file = results_dir / f"{task_id}_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                self.logger.info(f"Loaded task results from: {results_file}")
                return results
            else:
                self.logger.warning(f"Results file not found: {results_file}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load task results: {e}")
            return {}
    
    def check_data_freshness(self, data_path: str, max_age_hours: int = 24) -> bool:
        """
        Check if data is fresh enough for processing.
        
        Args:
            data_path: Path to data file
            max_age_hours: Maximum age in hours
            
        Returns:
            True if data is fresh enough
        """
        try:
            if not os.path.exists(data_path):
                return False
            
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(data_path))
            max_age = timedelta(hours=max_age_hours)
            
            is_fresh = file_age <= max_age
            self.logger.info(f"Data freshness check: {is_fresh} (age: {file_age}, max: {max_age})")
            
            return is_fresh
            
        except Exception as e:
            self.logger.error(f"Failed to check data freshness: {e}")
            return False
    
    def cleanup_old_artifacts(self, retention_days: int = 7) -> None:
        """
        Cleanup old artifacts and results.
        
        Args:
            retention_days: Number of days to retain artifacts
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Cleanup results
            results_dir = Path(self.config['results_path'])
            if results_dir.exists():
                for item in results_dir.iterdir():
                    if item.is_dir():
                        try:
                            item_date = datetime.strptime(item.name, '%Y-%m-%d')
                            if item_date < cutoff_date:
                                import shutil
                                shutil.rmtree(item)
                                self.logger.info(f"Cleaned up old results: {item}")
                        except (ValueError, OSError) as e:
                            self.logger.warning(f"Failed to cleanup {item}: {e}")
            
            # Cleanup old models (keep only latest N versions)
            model_dir = Path(self.config['model_path']).parent
            if model_dir.exists():
                model_files = sorted(model_dir.glob('telco_churn_model_*'), 
                                   key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Keep only 5 most recent models
                for old_model in model_files[5:]:
                    try:
                        if old_model.is_file():
                            old_model.unlink()
                        elif old_model.is_dir():
                            import shutil
                            shutil.rmtree(old_model)
                        self.logger.info(f"Cleaned up old model: {old_model}")
                    except OSError as e:
                        self.logger.warning(f"Failed to cleanup model {old_model}: {e}")
            
            self.logger.info(f"Cleanup completed (retention: {retention_days} days)")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

def create_notification_message(pipeline_report: Dict[str, Any]) -> str:
    """
    Create a formatted notification message for pipeline completion.
    
    Args:
        pipeline_report: Pipeline execution report
        
    Returns:
        Formatted notification message
    """
    status_emoji = "✅" if pipeline_report['execution_summary']['status'] == 'SUCCESS' else "❌"
    
    message = f"""
{status_emoji} **Telco Churn ML Pipeline Completed**

📊 **Execution Summary:**
• Date: {pipeline_report['pipeline_execution_date']}
• Run ID: {pipeline_report['pipeline_run_id'][:8]}...
• Status: {pipeline_report['execution_summary']['status']}
• Duration: {pipeline_report['execution_summary']['total_execution_time']:.2f}s

🤖 **Model Performance:**
• Best Algorithm: {pipeline_report['evaluation_summary']['best_model']}
• Test Accuracy: {pipeline_report['evaluation_summary']['test_accuracy']:.4f}
• F1 Score: {pipeline_report['evaluation_summary']['test_f1_score']:.4f}
• AUC: {pipeline_report['evaluation_summary']['test_auc']:.4f}

📈 **Data Processing:**
• Training Records: {pipeline_report['preprocessing_summary']['train_shape']:,}
• Test Records: {pipeline_report['preprocessing_summary']['test_shape']:,}
• Features: {pipeline_report['preprocessing_summary']['feature_count']}

🔮 **Predictions:**
• Generated: {pipeline_report['inference_summary']['predictions_generated']:,}
• Churn Predicted: {pipeline_report['inference_summary']['churn_predictions']:,}
• Avg Probability: {pipeline_report['inference_summary']['average_churn_probability']:.3f}

💡 **Recommendations:**
{chr(10).join(f"• {rec}" for rec in pipeline_report.get('recommendations', []))}
"""
    
    return message

def validate_model_performance(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate model performance against thresholds.
    
    Args:
        metrics: Model performance metrics
        thresholds: Performance thresholds
        
    Returns:
        Validation results
    """
    validation_results = {}
    
    for metric, threshold in thresholds.items():
        if metric in metrics:
            validation_results[metric] = metrics[metric] >= threshold
        else:
            validation_results[metric] = False
    
    return validation_results

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the ML pipeline.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data_path': '/opt/airflow/data/raw/TelcoCustomerChurnPrediction.csv',
        'model_path': '/opt/airflow/artifacts/models/telco_churn_model',
        'results_path': '/opt/airflow/artifacts/results',
        'target_column': 'Churn',
        'test_size': 0.2,
        'algorithms': ['gbt', 'randomforest', 'logisticregression'],
        'hyperparameter_tuning': True,
        'cross_validation_folds': 3,
        'mlflow_experiment': 'telco_churn_airflow_pipeline',
        'performance_thresholds': {
            'accuracy': 0.75,
            'f1_score': 0.70,
            'precision': 0.70,
            'recall': 0.70,
            'auc': 0.75
        },
        'data_freshness_hours': 24,
        'retention_days': 7
    }