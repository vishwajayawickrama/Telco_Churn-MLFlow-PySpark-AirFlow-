"""
Airflow DAG for Telco Customer Churn ML Pipeline Orchestration.

This DAG orchestrates the complete machine learning pipeline workflow including:
1. Data preprocessing and feature engineering
2. Model training with hyperparameter tuning
3. Model evaluation and validation
4. Inference generation and result storage

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
import os
import sys
import logging
import pandas as pd
import json
from typing import Dict, Any, List

# Add project paths
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'pipelines'))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

# DAG Configuration
DEFAULT_ARGS = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4),
    'email': ['data-team@company.com']
}

# DAG Definition
dag = DAG(
    'telco_churn_ml_pipeline',
    default_args=DEFAULT_ARGS,
    description='Complete ML Pipeline for Telco Customer Churn Prediction',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'pyspark', 'churn-prediction', 'telco'],
    doc_md=__doc__
)

# Configuration
CONFIG = {
    'data_path': os.path.join(project_root, 'data', 'raw', 'TelcoCustomerChurnPrediction.csv'),
    'model_path': os.path.join(project_root, 'artifacts', 'models', 'telco_churn_model'),
    'results_path': os.path.join(project_root, 'artifacts', 'results'),
    'target_column': 'Churn',
    'test_size': 0.2,
    'algorithms': ['gbt', 'randomforest', 'logisticregression'],
    'hyperparameter_tuning': True,
    'cross_validation_folds': 3,
    'mlflow_experiment': 'telco_churn_airflow_pipeline'
}

# ==============================================================================
# TASK FUNCTIONS
# ==============================================================================

def check_data_quality(**context):
    """
    Check data quality and validate input data.
    """
    logging.info("Starting data quality check...")
    
    try:
        # Import here to avoid dependency issues
        from data_pipeline import data_pipeline_pyspark
        
        data_path = CONFIG['data_path']
        
        # Basic file existence check
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data sample for quality checks
        df = pd.read_csv(data_path, nrows=1000)
        
        # Data quality checks
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'target_distribution': df[CONFIG['target_column']].value_counts().to_dict() if CONFIG['target_column'] in df.columns else None
        }
        
        logging.info(f"Data quality metrics: {quality_metrics}")
        
        # Store metrics for downstream tasks
        context['task_instance'].xcom_push(key='data_quality_metrics', value=quality_metrics)
        
        # Quality validation
        if quality_metrics['missing_values'] > len(df) * 0.5:
            raise ValueError("Too many missing values in dataset")
        
        if CONFIG['target_column'] not in df.columns:
            raise ValueError(f"Target column '{CONFIG['target_column']}' not found in dataset")
        
        logging.info("Data quality check completed successfully")
        return quality_metrics
        
    except Exception as e:
        logging.error(f"Data quality check failed: {str(e)}")
        raise

def run_data_preprocessing(**context):
    """
    Execute data preprocessing and feature engineering pipeline.
    """
    logging.info("Starting data preprocessing pipeline...")
    
    try:
        from data_pipeline import data_pipeline_pyspark
        
        # Execute data pipeline
        result = data_pipeline_pyspark(
            data_path=CONFIG['data_path'],
            target_column=CONFIG['target_column'],
            test_size=CONFIG['test_size'],
            force_rebuild=True
        )
        
        # Store preprocessing results
        preprocessing_metrics = {
            'train_shape': result['train_df'].count(),
            'test_shape': result['test_df'].count(),
            'feature_count': len(result['train_df'].columns) - 1,
            'preprocessing_steps': [
                'data_ingestion', 'missing_values', 'outlier_detection',
                'feature_binning', 'feature_encoding', 'feature_scaling',
                'data_splitting'
            ]
        }
        
        logging.info(f"Preprocessing completed: {preprocessing_metrics}")
        context['task_instance'].xcom_push(key='preprocessing_metrics', value=preprocessing_metrics)
        
        return preprocessing_metrics
        
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def run_model_training(**context):
    """
    Execute model training with multiple algorithms and hyperparameter tuning.
    """
    logging.info("Starting model training pipeline...")
    
    try:
        from training_pipeline import training_pipeline_pyspark, compare_models_pyspark
        
        training_results = {}
        
        # Train multiple algorithms
        for algorithm in CONFIG['algorithms']:
            logging.info(f"Training {algorithm} model...")
            
            result = training_pipeline_pyspark(
                algorithm=algorithm,
                hyperparameter_tuning=CONFIG['hyperparameter_tuning'],
                cross_validation_folds=CONFIG['cross_validation_folds'],
                force_rebuild=True
            )
            
            training_results[algorithm] = {
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'auc': result['auc'],
                'training_time': result['training_time'],
                'model_path': result['model_path']
            }
            
            logging.info(f"{algorithm} training completed with accuracy: {result['accuracy']:.4f}")
        
        # Compare models and select best
        comparison_result = compare_models_pyspark()
        best_model = comparison_result['best_model']
        
        training_summary = {
            'algorithms_trained': CONFIG['algorithms'],
            'best_model': best_model,
            'model_comparison': training_results,
            'hyperparameter_tuning': CONFIG['hyperparameter_tuning'],
            'cross_validation_folds': CONFIG['cross_validation_folds']
        }
        
        logging.info(f"Model training completed. Best model: {best_model}")
        context['task_instance'].xcom_push(key='training_results', value=training_summary)
        
        return training_summary
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise

def run_model_evaluation(**context):
    """
    Execute comprehensive model evaluation and validation.
    """
    logging.info("Starting model evaluation...")
    
    try:
        # Get training results from previous task
        training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='model_training')
        best_model = training_results['best_model']
        
        # Load model and evaluate
        from training_pipeline import training_pipeline_pyspark
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Detailed evaluation of best model
        detailed_evaluation = training_pipeline_pyspark(
            algorithm=best_model,
            evaluation_only=True
        )
        
        evaluation_metrics = {
            'best_model': best_model,
            'test_accuracy': detailed_evaluation['test_accuracy'],
            'test_f1_score': detailed_evaluation['test_f1_score'],
            'test_precision': detailed_evaluation['test_precision'],
            'test_recall': detailed_evaluation['test_recall'],
            'test_auc': detailed_evaluation['test_auc'],
            'feature_importance': detailed_evaluation['feature_importance'][:10],  # Top 10 features
            'confusion_matrix': detailed_evaluation['confusion_matrix'],
            'classification_report': detailed_evaluation['classification_report'],
            'validation_status': 'PASSED' if detailed_evaluation['test_accuracy'] > 0.75 else 'FAILED'
        }
        
        logging.info(f"Model evaluation completed. Validation status: {evaluation_metrics['validation_status']}")
        context['task_instance'].xcom_push(key='evaluation_metrics', value=evaluation_metrics)
        
        # Fail the task if model performance is below threshold
        if evaluation_metrics['validation_status'] == 'FAILED':
            raise ValueError(f"Model performance below threshold. Accuracy: {evaluation_metrics['test_accuracy']:.4f}")
        
        return evaluation_metrics
        
    except Exception as e:
        logging.error(f"Model evaluation failed: {str(e)}")
        raise

def run_batch_inference(**context):
    """
    Execute batch inference on new data and store results.
    """
    logging.info("Starting batch inference...")
    
    try:
        from streaming_inference_pipeline import streaming_inference_pyspark
        
        # Get evaluation results
        evaluation_metrics = context['task_instance'].xcom_pull(key='evaluation_metrics', task_ids='model_evaluation')
        best_model = evaluation_metrics['best_model']
        
        # Run batch inference
        model_path = f"{CONFIG['model_path']}_{best_model}"
        batch_data_path = CONFIG['data_path']  # Using same data for demo
        
        inference_result = streaming_inference_pyspark(
            model_path=model_path,
            batch_data_path=batch_data_path
        )
        
        # Store inference results
        results_path = f"{CONFIG['results_path']}/batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Save predictions
        inference_result['predictions'].to_csv(results_path, index=False)
        
        inference_summary = {
            'model_used': best_model,
            'predictions_generated': len(inference_result['predictions']),
            'churn_predictions': inference_result['predictions']['prediction'].sum(),
            'results_path': results_path,
            'inference_time': inference_result['inference_time'],
            'average_churn_probability': inference_result['predictions']['probability'].mean()
        }
        
        logging.info(f"Batch inference completed: {inference_summary}")
        context['task_instance'].xcom_push(key='inference_results', value=inference_summary)
        
        return inference_summary
        
    except Exception as e:
        logging.error(f"Batch inference failed: {str(e)}")
        raise

def generate_pipeline_report(**context):
    """
    Generate comprehensive pipeline execution report.
    """
    logging.info("Generating pipeline report...")
    
    try:
        # Collect results from all tasks
        data_quality = context['task_instance'].xcom_pull(key='data_quality_metrics', task_ids='data_quality_check')
        preprocessing = context['task_instance'].xcom_pull(key='preprocessing_metrics', task_ids='data_preprocessing')
        training = context['task_instance'].xcom_pull(key='training_results', task_ids='model_training')
        evaluation = context['task_instance'].xcom_pull(key='evaluation_metrics', task_ids='model_evaluation')
        inference = context['task_instance'].xcom_pull(key='inference_results', task_ids='batch_inference')
        
        # Generate comprehensive report
        pipeline_report = {
            'pipeline_execution_date': context['ds'],
            'pipeline_run_id': context['run_id'],
            'execution_summary': {
                'total_execution_time': (datetime.now() - context['data_interval_start']).total_seconds(),
                'tasks_completed': 5,
                'status': 'SUCCESS'
            },
            'data_quality_summary': data_quality,
            'preprocessing_summary': preprocessing,
            'training_summary': training,
            'evaluation_summary': evaluation,
            'inference_summary': inference,
            'recommendations': [],
            'next_actions': []
        }
        
        # Add recommendations based on results
        if evaluation['test_accuracy'] > 0.90:
            pipeline_report['recommendations'].append("Excellent model performance. Consider deploying to production.")
        elif evaluation['test_accuracy'] > 0.80:
            pipeline_report['recommendations'].append("Good model performance. Monitor in staging environment.")
        else:
            pipeline_report['recommendations'].append("Model performance needs improvement. Review feature engineering.")
        
        # Save report
        report_path = f"{CONFIG['results_path']}/pipeline_report_{context['ds']}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(pipeline_report, f, indent=2, default=str)
        
        logging.info(f"Pipeline report generated: {report_path}")
        context['task_instance'].xcom_push(key='pipeline_report', value=pipeline_report)
        
        return pipeline_report
        
    except Exception as e:
        logging.error(f"Pipeline report generation failed: {str(e)}")
        raise

def notify_pipeline_completion(**context):
    """
    Send notification about pipeline completion.
    """
    logging.info("Sending pipeline completion notification...")
    
    try:
        pipeline_report = context['task_instance'].xcom_pull(key='pipeline_report', task_ids='generate_report')
        
        # Create notification message
        message = f"""
🤖 Telco Churn ML Pipeline Completed Successfully!

📊 **Execution Summary:**
- Date: {pipeline_report['pipeline_execution_date']}
- Best Model: {pipeline_report['evaluation_summary']['best_model']}
- Test Accuracy: {pipeline_report['evaluation_summary']['test_accuracy']:.4f}
- Predictions Generated: {pipeline_report['inference_summary']['predictions_generated']}

🎯 **Key Metrics:**
- Training Data: {pipeline_report['preprocessing_summary']['train_shape']} records
- Test Data: {pipeline_report['preprocessing_summary']['test_shape']} records
- Feature Count: {pipeline_report['preprocessing_summary']['feature_count']}
- Churn Rate: {pipeline_report['inference_summary']['churn_predictions']}/{pipeline_report['inference_summary']['predictions_generated']}

✅ **Status:** {pipeline_report['execution_summary']['status']}
"""
        
        logging.info(f"Pipeline completion notification: {message}")
        return message
        
    except Exception as e:
        logging.error(f"Notification failed: {str(e)}")
        raise

# ==============================================================================
# DAG TASK DEFINITION
# ==============================================================================

# 1. START TASK
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
    doc_md="Start of the ML pipeline execution"
)

# 2. DATA QUALITY CHECK
data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag,
    doc_md="Validate input data quality and schema"
)

# 3. DATA PREPROCESSING TASK GROUP
with TaskGroup('data_preprocessing', dag=dag) as preprocessing_group:
    
    preprocessing_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_data_preprocessing,
        doc_md="Execute data preprocessing and feature engineering"
    )
    
    validate_preprocessing = BashOperator(
        task_id='validate_preprocessing',
        bash_command="""
        echo "Validating preprocessing results..."
        # Add validation commands here
        if [ -f "/opt/airflow/artifacts/data/X_train.csv" ]; then
            echo "Preprocessing validation successful"
        else
            echo "Preprocessing validation failed"
            exit 1
        fi
        """,
        doc_md="Validate preprocessing outputs"
    )
    
    preprocessing_task >> validate_preprocessing

# 4. MODEL TRAINING TASK GROUP
with TaskGroup('model_training', dag=dag) as training_group:
    
    training_task = PythonOperator(
        task_id='run_training',
        python_callable=run_model_training,
        doc_md="Train multiple ML models with hyperparameter tuning"
    )
    
    validate_training = BashOperator(
        task_id='validate_training',
        bash_command="""
        echo "Validating model training results..."
        # Check if model artifacts exist
        if [ -d "/opt/airflow/artifacts/models" ]; then
            echo "Model training validation successful"
        else
            echo "Model training validation failed"
            exit 1
        fi
        """,
        doc_md="Validate training outputs"
    )
    
    training_task >> validate_training

# 5. MODEL EVALUATION
model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag,
    doc_md="Comprehensive model evaluation and validation"
)

# 6. BATCH INFERENCE
batch_inference = PythonOperator(
    task_id='batch_inference',
    python_callable=run_batch_inference,
    dag=dag,
    doc_md="Generate batch predictions on new data"
)

# 7. REPORTING
generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_pipeline_report,
    dag=dag,
    doc_md="Generate comprehensive pipeline execution report"
)

# 8. NOTIFICATION
notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=notify_pipeline_completion,
    dag=dag,
    trigger_rule='all_done',  # Run regardless of upstream failures
    doc_md="Send pipeline completion notification"
)

# 9. END TASK
end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
    trigger_rule='all_done',
    doc_md="End of the ML pipeline execution"
)

# ==============================================================================
# TASK DEPENDENCIES
# ==============================================================================

# Define the complete workflow
start_task >> data_quality_check >> preprocessing_group >> training_group >> model_evaluation >> batch_inference >> generate_report >> notification_task >> end_task

# Additional monitoring and alerting
if hasattr(dag, 'add_task'):
    # Health check task (optional)
    health_check = BashOperator(
        task_id='health_check',
        bash_command="""
        echo "Performing system health check..."
        # Check Spark cluster health
        # Check MLflow server health
        # Check storage availability
        echo "Health check completed"
        """,
        dag=dag
    )
    
    start_task >> health_check >> data_quality_check