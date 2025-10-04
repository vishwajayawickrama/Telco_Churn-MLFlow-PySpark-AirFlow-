"""
Model Monitoring and Drift Detection DAG for Telco Customer Churn ML Pipeline.

This DAG monitors deployed models for performance degradation and data drift,
providing alerts and recommendations for model retraining.

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
import sys
import os
import logging
import json
import pandas as pd
import numpy as np
import glob
from typing import Dict, Any, List

# Add project paths
import os
# Get the parent directory of dags folder (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'pipelines'))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'utils'))

from airflow_utils import AirflowMLPipelineUtils, get_default_config

# DAG Configuration
DEFAULT_ARGS = {
    'owner': 'ml-monitoring-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
    'execution_timeout': timedelta(hours=2),
    'email': ['ml-monitoring@company.com']
}

# DAG Definition
dag = DAG(
    'telco_churn_model_monitoring',
    default_args=DEFAULT_ARGS,
    description='Model Performance Monitoring and Data Drift Detection',
    schedule_interval='@daily',  # Run daily for monitoring
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'monitoring', 'drift-detection', 'performance', 'telco'],
    doc_md=__doc__
)

# Extended configuration for monitoring
CONFIG = get_default_config()
CONFIG.update({
    'monitoring': {
        'performance_threshold': {
            'accuracy': 0.85,
            'f1_score': 0.80,
            'precision': 0.78,
            'recall': 0.82
        },
        'drift_threshold': {
            'psi_threshold': 0.2,  # Population Stability Index
            'ks_threshold': 0.3,   # Kolmogorov-Smirnov test
            'chi2_threshold': 0.05  # Chi-square test p-value
        },
        'alert_channels': ['email', 'slack'],
        'monitoring_window_days': 7,
        'baseline_window_days': 30
    },
    'data_sources': {
        'production_predictions': os.path.join(project_root, 'data', 'predictions'),
        'ground_truth': os.path.join(project_root, 'data', 'ground_truth'),
        'reference_data': os.path.join(project_root, 'data', 'reference')
    }
})

# Initialize utilities
utils = AirflowMLPipelineUtils(CONFIG)

# ==============================================================================
# MONITORING FUNCTIONS
# ==============================================================================

def collect_production_data(**context):
    """
    Collect production predictions and ground truth data for monitoring.
    """
    logging.info("Collecting production data for monitoring...")
    
    try:
        import glob
        from datetime import datetime, timedelta
        
        # Calculate monitoring window
        end_date = datetime.strptime(context['ds'], '%Y-%m-%d')
        start_date = end_date - timedelta(days=CONFIG['monitoring']['monitoring_window_days'])
        
        # Collect prediction files
        prediction_files = glob.glob(
            f"{CONFIG['data_sources']['production_predictions']}/predictions_*.csv"
        )
        
        # Filter files by date range
        production_data = []
        for file_path in prediction_files:
            file_date_str = os.path.basename(file_path).split('_')[1].split('.')[0]
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if start_date <= file_date <= end_date:
                    df = pd.read_csv(file_path)
                    df['prediction_date'] = file_date
                    production_data.append(df)
            except ValueError:
                logging.warning(f"Could not parse date from filename: {file_path}")
                continue
        
        if not production_data:
            logging.warning("No production data found for monitoring window")
            return None
        
        # Combine all data
        combined_data = pd.concat(production_data, ignore_index=True)
        
        # Collect ground truth data if available
        ground_truth_files = glob.glob(
            f"{CONFIG['data_sources']['ground_truth']}/ground_truth_*.csv"
        )
        
        ground_truth_data = []
        for file_path in ground_truth_files:
            file_date_str = os.path.basename(file_path).split('_')[2].split('.')[0]
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if start_date <= file_date <= end_date:
                    df = pd.read_csv(file_path)
                    df['ground_truth_date'] = file_date
                    ground_truth_data.append(df)
            except ValueError:
                continue
        
        ground_truth_df = pd.concat(ground_truth_data, ignore_index=True) if ground_truth_data else None
        
        # Save collected data
        monitoring_data = {
            'predictions': combined_data.to_dict('records'),
            'ground_truth': ground_truth_df.to_dict('records') if ground_truth_df is not None else None,
            'monitoring_window': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'num_predictions': len(combined_data)
            }
        }
        
        results_file = utils.save_task_results('production_data_collection', monitoring_data, context)
        
        logging.info(f"Collected {len(combined_data)} predictions for monitoring")
        context['task_instance'].xcom_push(key='monitoring_data', value=monitoring_data)
        
        return monitoring_data
        
    except Exception as e:
        logging.error(f"Failed to collect production data: {e}")
        raise

def detect_data_drift(**context):
    """
    Detect data drift using statistical tests.
    """
    logging.info("Detecting data drift...")
    
    try:
        from scipy import stats
        
        # Get monitoring data
        monitoring_data = context['task_instance'].xcom_pull(key='monitoring_data', task_ids='collect_production_data')
        
        if not monitoring_data or not monitoring_data['predictions']:
            logging.warning("No monitoring data available for drift detection")
            return None
        
        # Load current data
        current_df = pd.DataFrame(monitoring_data['predictions'])
        
        # Load reference/baseline data
        reference_files = glob.glob(f"{CONFIG['data_sources']['reference_data']}/reference_*.csv")
        if not reference_files:
            logging.warning("No reference data found for drift detection")
            return None
        
        reference_df = pd.read_csv(reference_files[0])  # Use most recent reference
        
        # Perform drift detection for numerical features
        numerical_features = current_df.select_dtypes(include=[np.number]).columns
        categorical_features = current_df.select_dtypes(include=['object']).columns
        
        drift_results = {
            'numerical_drift': {},
            'categorical_drift': {},
            'overall_drift_detected': False,
            'drift_score': 0.0
        }
        
        # Kolmogorov-Smirnov test for numerical features
        for feature in numerical_features:
            if feature in reference_df.columns:
                current_values = current_df[feature].dropna()
                reference_values = reference_df[feature].dropna()
                
                if len(current_values) > 0 and len(reference_values) > 0:
                    ks_stat, p_value = stats.ks_2samp(reference_values, current_values)
                    
                    drift_detected = ks_stat > CONFIG['monitoring']['drift_threshold']['ks_threshold']
                    
                    drift_results['numerical_drift'][feature] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected,
                        'drift_magnitude': 'high' if ks_stat > 0.5 else 'medium' if ks_stat > 0.3 else 'low'
                    }
                    
                    if drift_detected:
                        drift_results['overall_drift_detected'] = True
                        drift_results['drift_score'] += ks_stat
        
        # Chi-square test for categorical features
        for feature in categorical_features:
            if feature in reference_df.columns:
                current_counts = current_df[feature].value_counts()
                reference_counts = reference_df[feature].value_counts()
                
                # Align categories
                all_categories = set(current_counts.index) | set(reference_counts.index)
                current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                
                if sum(current_aligned) > 0 and sum(reference_aligned) > 0:
                    chi2_stat, p_value = stats.chisquare(current_aligned, reference_aligned)
                    
                    drift_detected = p_value < CONFIG['monitoring']['drift_threshold']['chi2_threshold']
                    
                    drift_results['categorical_drift'][feature] = {
                        'chi2_statistic': chi2_stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected,
                        'drift_magnitude': 'high' if p_value < 0.01 else 'medium' if p_value < 0.03 else 'low'
                    }
                    
                    if drift_detected:
                        drift_results['overall_drift_detected'] = True
                        drift_results['drift_score'] += (1 - p_value)
        
        # Normalize drift score
        total_features = len(numerical_features) + len(categorical_features)
        if total_features > 0:
            drift_results['drift_score'] /= total_features
        
        # Save drift detection results
        results_file = utils.save_task_results('drift_detection', drift_results, context)
        
        logging.info(f"Drift detection completed. Overall drift detected: {drift_results['overall_drift_detected']}")
        context['task_instance'].xcom_push(key='drift_results', value=drift_results)
        
        return drift_results
        
    except Exception as e:
        logging.error(f"Data drift detection failed: {e}")
        raise

def evaluate_model_performance(**context):
    """
    Evaluate current model performance against ground truth.
    """
    logging.info("Evaluating model performance...")
    
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Get monitoring data
        monitoring_data = context['task_instance'].xcom_pull(key='monitoring_data', task_ids='collect_production_data')
        
        if not monitoring_data or not monitoring_data['ground_truth']:
            logging.warning("No ground truth data available for performance evaluation")
            return None
        
        # Convert to DataFrames
        predictions_df = pd.DataFrame(monitoring_data['predictions'])
        ground_truth_df = pd.DataFrame(monitoring_data['ground_truth'])
        
        # Merge predictions with ground truth
        merged_df = pd.merge(
            predictions_df, 
            ground_truth_df, 
            on='customer_id', 
            how='inner'
        )
        
        if len(merged_df) == 0:
            logging.warning("No matching records between predictions and ground truth")
            return None
        
        # Calculate performance metrics
        y_true = merged_df['actual_churn']
        y_pred = merged_df['predicted_churn']
        y_pred_proba = merged_df['churn_probability']
        
        performance_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'num_samples': len(merged_df)
        }
        
        # Check against thresholds
        performance_issues = {}
        thresholds = CONFIG['monitoring']['performance_threshold']
        
        for metric, value in performance_metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                if value < threshold:
                    performance_issues[metric] = {
                        'current_value': value,
                        'threshold': threshold,
                        'degradation': threshold - value
                    }
        
        performance_evaluation = {
            'current_performance': performance_metrics,
            'performance_issues': performance_issues,
            'performance_degraded': len(performance_issues) > 0,
            'evaluation_date': context['ds']
        }
        
        # Save performance evaluation results
        results_file = utils.save_task_results('performance_evaluation', performance_evaluation, context)
        
        logging.info(f"Performance evaluation completed. Issues detected: {len(performance_issues)}")
        context['task_instance'].xcom_push(key='performance_results', value=performance_evaluation)
        
        return performance_evaluation
        
    except Exception as e:
        logging.error(f"Model performance evaluation failed: {e}")
        raise

def generate_monitoring_report(**context):
    """
    Generate comprehensive monitoring report.
    """
    logging.info("Generating monitoring report...")
    
    try:
        # Collect all monitoring results
        drift_results = context['task_instance'].xcom_pull(key='drift_results', task_ids='drift_detection.detect_drift')
        performance_results = context['task_instance'].xcom_pull(key='performance_results', task_ids='performance_monitoring.evaluate_performance')
        monitoring_data = context['task_instance'].xcom_pull(key='monitoring_data', task_ids='collect_production_data')
        
        # Create comprehensive report
        monitoring_report = {
            'report_date': context['ds'],
            'monitoring_window': monitoring_data.get('monitoring_window', {}) if monitoring_data else {},
            'data_drift': drift_results or {},
            'performance_evaluation': performance_results or {},
            'recommendations': [],
            'alert_level': 'info'
        }
        
        # Generate recommendations based on findings
        if drift_results and drift_results.get('overall_drift_detected'):
            monitoring_report['recommendations'].append({
                'type': 'data_drift',
                'priority': 'high',
                'message': 'Significant data drift detected. Consider retraining the model with recent data.',
                'action': 'retrain_model'
            })
            monitoring_report['alert_level'] = 'warning'
        
        if performance_results and performance_results.get('performance_degraded'):
            monitoring_report['recommendations'].append({
                'type': 'performance_degradation',
                'priority': 'high',
                'message': 'Model performance has degraded below acceptable thresholds.',
                'action': 'investigate_and_retrain'
            })
            monitoring_report['alert_level'] = 'critical'
        
        if not monitoring_report['recommendations']:
            monitoring_report['recommendations'].append({
                'type': 'status',
                'priority': 'info',
                'message': 'Model is performing within acceptable parameters.',
                'action': 'continue_monitoring'
            })
        
        # Save monitoring report
        report_file = os.path.join(project_root, 'reports', f'monitoring_report_{context["ds"]}.json')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(monitoring_report, f, indent=2)
        
        logging.info(f"Monitoring report generated: {report_file}")
        context['task_instance'].xcom_push(key='monitoring_report', value=monitoring_report)
        
        return monitoring_report
        
    except Exception as e:
        logging.error(f"Failed to generate monitoring report: {e}")
        raise

def send_monitoring_alerts(**context):
    """
    Send monitoring alerts based on findings.
    """
    logging.info("Processing monitoring alerts...")
    
    try:
        # Get monitoring report
        monitoring_report = context['task_instance'].xcom_pull(key='monitoring_report', task_ids='generate_report')
        
        if not monitoring_report:
            logging.warning("No monitoring report available for alerting")
            return
        
        alert_level = monitoring_report.get('alert_level', 'info')
        
        if alert_level in ['warning', 'critical']:
            # Prepare alert message
            alert_message = f"""
            Model Monitoring Alert - {alert_level.upper()}
            
            Report Date: {monitoring_report['report_date']}
            Alert Level: {alert_level}
            
            Findings:
            """
            
            for recommendation in monitoring_report['recommendations']:
                alert_message += f"\n- {recommendation['message']} (Priority: {recommendation['priority']})"
            
            # Log alert (in production, this would send to Slack, email, etc.)
            logging.warning(f"MONITORING ALERT: {alert_message}")
            
            # Store alert for email task
            context['task_instance'].xcom_push(key='alert_message', value=alert_message)
            context['task_instance'].xcom_push(key='send_alert', value=True)
        else:
            logging.info("No alerts required - model performing normally")
            context['task_instance'].xcom_push(key='send_alert', value=False)
        
    except Exception as e:
        logging.error(f"Failed to process monitoring alerts: {e}")
        raise

# ==============================================================================
# DAG TASK DEFINITION
# ==============================================================================

# Start task
start_task = DummyOperator(
    task_id='start_monitoring',
    dag=dag
)

# Data collection
collect_data = PythonOperator(
    task_id='collect_production_data',
    python_callable=collect_production_data,
    dag=dag,
    doc_md="Collect production predictions and ground truth data"
)

# Drift detection task group
with TaskGroup('drift_detection', dag=dag) as drift_group:
    
    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_data_drift,
        doc_md="Detect data drift using statistical tests"
    )

# Performance monitoring task group
with TaskGroup('performance_monitoring', dag=dag) as performance_group:
    
    evaluate_performance = PythonOperator(
        task_id='evaluate_performance',
        python_callable=evaluate_model_performance,
        doc_md="Evaluate model performance against ground truth"
    )

# Reporting and alerting
generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_monitoring_report,
    dag=dag,
    doc_md="Generate comprehensive monitoring report"
)

process_alerts = PythonOperator(
    task_id='process_alerts',
    python_callable=send_monitoring_alerts,
    dag=dag,
    doc_md="Process and send monitoring alerts"
)

# Conditional email alert
send_email_alert = EmailOperator(
    task_id='send_email_alert',
    to=['ml-team@company.com'],
    subject='ML Model Monitoring Alert - {{ ti.xcom_pull(key="alert_level") }}',
    html_content='{{ ti.xcom_pull(key="alert_message") }}',
    dag=dag
)

# End task
end_task = DummyOperator(
    task_id='end_monitoring',
    dag=dag
)

# Task dependencies
start_task >> collect_data >> [drift_group, performance_group]
[drift_group, performance_group] >> generate_report >> process_alerts
process_alerts >> send_email_alert >> end_task