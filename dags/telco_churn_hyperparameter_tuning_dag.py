"""
Hyperparameter Tuning DAG for Telco Customer Churn ML Pipeline.

This DAG focuses on comprehensive hyperparameter tuning for multiple algorithms
to find the optimal model configuration for customer churn prediction.

Author: Data Science Team
Version: 2.0.0
Last Updated: 2024
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import sys
import os
import logging
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
    'owner': 'ml-engineering-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=8),
    'email': ['ml-team@company.com']
}

# DAG Definition
dag = DAG(
    'telco_churn_hyperparameter_tuning',
    default_args=DEFAULT_ARGS,
    description='Comprehensive Hyperparameter Tuning for Telco Churn Models',
    schedule_interval='@weekly',  # Run weekly for hyperparameter optimization
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'hyperparameter-tuning', 'optimization', 'telco'],
    doc_md=__doc__
)

# Extended configuration for hyperparameter tuning
CONFIG = get_default_config()
CONFIG.update({
    'hyperparameter_search_space': {
        'gbt': {
            'maxDepth': [5, 7, 10, 15],
            'numTrees': [50, 100, 200, 300],
            'stepSize': [0.01, 0.05, 0.1, 0.2],
            'subsamplingRate': [0.7, 0.8, 0.9, 1.0]
        },
        'randomforest': {
            'maxDepth': [5, 10, 15, 20],
            'numTrees': [50, 100, 200, 500],
            'subsamplingRate': [0.7, 0.8, 0.9, 1.0],
            'featureSubsetStrategy': ['auto', 'sqrt', 'log2']
        },
        'logisticregression': {
            'regParam': [0.01, 0.1, 1.0, 10.0],
            'elasticNetParam': [0.0, 0.5, 1.0],
            'maxIter': [100, 200, 500]
        }
    },
    'cross_validation_folds': 5,
    'hyperparameter_trials': 50,
    'optimization_metric': 'f1',
    'early_stopping_rounds': 10
})

# Initialize utilities
utils = AirflowMLPipelineUtils(CONFIG)

# ==============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# ==============================================================================

def run_grid_search_gbt(**context):
    """
    Perform grid search for Gradient Boosted Trees.
    """
    logging.info("Starting GBT hyperparameter tuning...")
    
    try:
        from training_pipeline import training_pipeline_pyspark
        import itertools
        import mlflow
        
        # Setup MLflow experiment
        experiment_id = utils.setup_mlflow_experiment("gbt_hyperparameter_tuning")
        
        # Get parameter space
        param_space = CONFIG['hyperparameter_search_space']['gbt']
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = 0.0
        best_params = {}
        results = []
        
        logging.info(f"Testing {len(param_combinations)} parameter combinations for GBT")
        
        for i, param_combo in enumerate(param_combinations[:CONFIG['hyperparameter_trials']]):
            params = dict(zip(param_names, param_combo))
            
            logging.info(f"Trial {i+1}/{CONFIG['hyperparameter_trials']}: {params}")
            
            try:
                with mlflow.start_run(run_name=f"gbt_trial_{i+1}"):
                    # Train model with specific parameters
                    result = training_pipeline_pyspark(
                        algorithm='gbt',
                        model_params=params,
                        cross_validation_folds=CONFIG['cross_validation_folds']
                    )
                    
                    # Log parameters and metrics
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'auc': result['auc']
                    })
                    
                    # Track best performance
                    score = result[CONFIG['optimization_metric']]
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    results.append({
                        'trial': i + 1,
                        'params': params,
                        'score': score,
                        'metrics': {
                            'accuracy': result['accuracy'],
                            'f1_score': result['f1_score'],
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'auc': result['auc']
                        }
                    })
                    
            except Exception as e:
                logging.warning(f"Trial {i+1} failed: {e}")
                continue
        
        # Summary results
        tuning_summary = {
            'algorithm': 'gbt',
            'total_trials': len(results),
            'best_score': best_score,
            'best_params': best_params,
            'optimization_metric': CONFIG['optimization_metric'],
            'all_results': results
        }
        
        logging.info(f"GBT tuning completed. Best {CONFIG['optimization_metric']}: {best_score:.4f}")
        context['task_instance'].xcom_push(key='gbt_tuning_results', value=tuning_summary)
        
        return tuning_summary
        
    except Exception as e:
        logging.error(f"GBT hyperparameter tuning failed: {e}")
        raise

def run_grid_search_randomforest(**context):
    """
    Perform grid search for Random Forest.
    """
    logging.info("Starting Random Forest hyperparameter tuning...")
    
    try:
        from training_pipeline import training_pipeline_pyspark
        import itertools
        import mlflow
        
        # Setup MLflow experiment
        experiment_id = utils.setup_mlflow_experiment("rf_hyperparameter_tuning")
        
        # Get parameter space
        param_space = CONFIG['hyperparameter_search_space']['randomforest']
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = 0.0
        best_params = {}
        results = []
        
        logging.info(f"Testing {len(param_combinations)} parameter combinations for Random Forest")
        
        for i, param_combo in enumerate(param_combinations[:CONFIG['hyperparameter_trials']]):
            params = dict(zip(param_names, param_combo))
            
            logging.info(f"Trial {i+1}/{CONFIG['hyperparameter_trials']}: {params}")
            
            try:
                with mlflow.start_run(run_name=f"rf_trial_{i+1}"):
                    # Train model with specific parameters
                    result = training_pipeline_pyspark(
                        algorithm='randomforest',
                        model_params=params,
                        cross_validation_folds=CONFIG['cross_validation_folds']
                    )
                    
                    # Log parameters and metrics
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'auc': result['auc']
                    })
                    
                    # Track best performance
                    score = result[CONFIG['optimization_metric']]
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    results.append({
                        'trial': i + 1,
                        'params': params,
                        'score': score,
                        'metrics': {
                            'accuracy': result['accuracy'],
                            'f1_score': result['f1_score'],
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'auc': result['auc']
                        }
                    })
                    
            except Exception as e:
                logging.warning(f"Trial {i+1} failed: {e}")
                continue
        
        # Summary results
        tuning_summary = {
            'algorithm': 'randomforest',
            'total_trials': len(results),
            'best_score': best_score,
            'best_params': best_params,
            'optimization_metric': CONFIG['optimization_metric'],
            'all_results': results
        }
        
        logging.info(f"Random Forest tuning completed. Best {CONFIG['optimization_metric']}: {best_score:.4f}")
        context['task_instance'].xcom_push(key='rf_tuning_results', value=tuning_summary)
        
        return tuning_summary
        
    except Exception as e:
        logging.error(f"Random Forest hyperparameter tuning failed: {e}")
        raise

def run_grid_search_logistic(**context):
    """
    Perform grid search for Logistic Regression.
    """
    logging.info("Starting Logistic Regression hyperparameter tuning...")
    
    try:
        from training_pipeline import training_pipeline_pyspark
        import itertools
        import mlflow
        
        # Setup MLflow experiment
        experiment_id = utils.setup_mlflow_experiment("lr_hyperparameter_tuning")
        
        # Get parameter space
        param_space = CONFIG['hyperparameter_search_space']['logisticregression']
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = 0.0
        best_params = {}
        results = []
        
        logging.info(f"Testing {len(param_combinations)} parameter combinations for Logistic Regression")
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            logging.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                with mlflow.start_run(run_name=f"lr_trial_{i+1}"):
                    # Train model with specific parameters
                    result = training_pipeline_pyspark(
                        algorithm='logisticregression',
                        model_params=params,
                        cross_validation_folds=CONFIG['cross_validation_folds']
                    )
                    
                    # Log parameters and metrics
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'auc': result['auc']
                    })
                    
                    # Track best performance
                    score = result[CONFIG['optimization_metric']]
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    results.append({
                        'trial': i + 1,
                        'params': params,
                        'score': score,
                        'metrics': {
                            'accuracy': result['accuracy'],
                            'f1_score': result['f1_score'],
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'auc': result['auc']
                        }
                    })
                    
            except Exception as e:
                logging.warning(f"Trial {i+1} failed: {e}")
                continue
        
        # Summary results
        tuning_summary = {
            'algorithm': 'logisticregression',
            'total_trials': len(results),
            'best_score': best_score,
            'best_params': best_params,
            'optimization_metric': CONFIG['optimization_metric'],
            'all_results': results
        }
        
        logging.info(f"Logistic Regression tuning completed. Best {CONFIG['optimization_metric']}: {best_score:.4f}")
        context['task_instance'].xcom_push(key='lr_tuning_results', value=tuning_summary)
        
        return tuning_summary
        
    except Exception as e:
        logging.error(f"Logistic Regression hyperparameter tuning failed: {e}")
        raise

def compare_tuning_results(**context):
    """
    Compare hyperparameter tuning results across all algorithms.
    """
    logging.info("Comparing hyperparameter tuning results...")
    
    try:
        # Collect results from all tuning tasks
        gbt_results = context['task_instance'].xcom_pull(key='gbt_tuning_results', task_ids='hyperparameter_tuning.tune_gbt')
        rf_results = context['task_instance'].xcom_pull(key='rf_tuning_results', task_ids='hyperparameter_tuning.tune_randomforest')
        lr_results = context['task_instance'].xcom_pull(key='lr_tuning_results', task_ids='hyperparameter_tuning.tune_logistic')
        
        all_results = [gbt_results, rf_results, lr_results]
        
        # Find overall best model
        best_overall = max(all_results, key=lambda x: x['best_score'])
        
        # Create comparison summary
        comparison_summary = {
            'tuning_date': context['ds'],
            'optimization_metric': CONFIG['optimization_metric'],
            'best_overall': {
                'algorithm': best_overall['algorithm'],
                'score': best_overall['best_score'],
                'params': best_overall['best_params']
            },
            'algorithm_results': {
                'gbt': {
                    'best_score': gbt_results['best_score'],
                    'best_params': gbt_results['best_params'],
                    'trials_completed': gbt_results['total_trials']
                },
                'randomforest': {
                    'best_score': rf_results['best_score'],
                    'best_params': rf_results['best_params'],
                    'trials_completed': rf_results['total_trials']
                },
                'logisticregression': {
                    'best_score': lr_results['best_score'],
                    'best_params': lr_results['best_params'],
                    'trials_completed': lr_results['total_trials']
                }
            }
        }
        
        # Save comparison results
        results_file = utils.save_task_results('hyperparameter_comparison', comparison_summary, context)
        
        logging.info(f"Hyperparameter tuning comparison completed. Best: {best_overall['algorithm']} with {CONFIG['optimization_metric']}: {best_overall['best_score']:.4f}")
        context['task_instance'].xcom_push(key='tuning_comparison', value=comparison_summary)
        
        return comparison_summary
        
    except Exception as e:
        logging.error(f"Hyperparameter tuning comparison failed: {e}")
        raise

def update_production_config(**context):
    """
    Update production configuration with best hyperparameters.
    """
    logging.info("Updating production configuration...")
    
    try:
        # Get comparison results
        comparison = context['task_instance'].xcom_pull(key='tuning_comparison', task_ids='compare_results')
        
        # Create production configuration
        production_config = {
            'model_selection': {
                'algorithm': comparison['best_overall']['algorithm'],
                'hyperparameters': comparison['best_overall']['params'],
                'expected_performance': comparison['best_overall']['score'],
                'tuning_date': context['ds'],
                'validation_metric': CONFIG['optimization_metric']
            },
            'alternative_models': comparison['algorithm_results'],
            'deployment_ready': True
        }
        
        # Save production configuration
        import json
        config_path = os.path.join(project_root, 'artifacts', 'config', f'production_model_config_{context["ds"]}.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(production_config, f, indent=2)
        
        logging.info(f"Production configuration updated: {config_path}")
        context['task_instance'].xcom_push(key='production_config', value=production_config)
        
        return production_config
        
    except Exception as e:
        logging.error(f"Failed to update production configuration: {e}")
        raise

# ==============================================================================
# DAG TASK DEFINITION
# ==============================================================================

# Start task
start_task = DummyOperator(
    task_id='start_hyperparameter_tuning',
    dag=dag
)

# Hyperparameter tuning task group
with TaskGroup('hyperparameter_tuning', dag=dag) as tuning_group:
    
    tune_gbt = PythonOperator(
        task_id='tune_gbt',
        python_callable=run_grid_search_gbt,
        doc_md="Hyperparameter tuning for Gradient Boosted Trees"
    )
    
    tune_randomforest = PythonOperator(
        task_id='tune_randomforest',
        python_callable=run_grid_search_randomforest,
        doc_md="Hyperparameter tuning for Random Forest"
    )
    
    tune_logistic = PythonOperator(
        task_id='tune_logistic',
        python_callable=run_grid_search_logistic,
        doc_md="Hyperparameter tuning for Logistic Regression"
    )

# Comparison and configuration update
compare_results = PythonOperator(
    task_id='compare_results',
    python_callable=compare_tuning_results,
    dag=dag,
    doc_md="Compare hyperparameter tuning results across algorithms"
)

update_config = PythonOperator(
    task_id='update_production_config',
    python_callable=update_production_config,
    dag=dag,
    doc_md="Update production configuration with best hyperparameters"
)

# End task
end_task = DummyOperator(
    task_id='end_hyperparameter_tuning',
    dag=dag
)

# Task dependencies
start_task >> tuning_group >> compare_results >> update_config >> end_task