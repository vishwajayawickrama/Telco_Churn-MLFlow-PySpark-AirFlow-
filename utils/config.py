import os
import sys
import yaml
from typing import Dict, Any, List

sys.path.append(os.path.dirname(__file__))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

@log_exceptions(logger)
def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
        
        logger.debug(f"Loading configuration from: {CONFIG_FILE}")
        
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            logger.warning("Configuration file is empty or invalid")
            return {}
        
        # Log configuration sections
        config_sections = list(config.keys())
        logger.debug(f"Configuration sections loaded: {config_sections}")
        
        return config
        
    except FileNotFoundError as e:
        ProjectLogger.log_error_header(logger, "CONFIGURATION FILE NOT FOUND")
        logger.error(f"File error: {str(e)}")
        raise
        
    except yaml.YAMLError as e:
        ProjectLogger.log_error_header(logger, "YAML PARSING ERROR")
        logger.error(f"YAML error: {str(e)}")
        raise
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN CONFIG LOADING")
        logger.error(f"Unexpected error: {str(e)}")
        return {}

@log_exceptions(logger)
def get_data_paths() -> Dict[str, str]:
    """
    Get data paths configuration.
    
    Returns:
        Dict[str, str]: Data paths configuration
    """
    logger.debug("Retrieving data paths configuration")
    config = load_config()
    data_paths = config.get('data_paths', {})
    
    if not data_paths:
        logger.warning("No data paths configuration found")
    else:
        logger.debug(f"Data paths loaded: {len(data_paths)} paths")
    
    return data_paths

@log_exceptions(logger)
def get_columns() -> Dict[str, List[str]]:
    """
    Get columns configuration.
    
    Returns:
        Dict[str, List[str]]: Columns configuration
    """
    logger.debug("Retrieving columns configuration")
    config = load_config()
    columns = config.get('columns', {})
    
    if not columns:
        logger.warning("No columns configuration found")
    else:
        logger.debug(f"Columns configuration loaded: {list(columns.keys())}")
    
    return columns


@log_exceptions(logger)
def get_missing_values_config() -> Dict[str, Any]:
    """
    Get missing values configuration.
    
    Returns:
        Dict[str, Any]: Missing values configuration
    """
    logger.debug("Retrieving missing values configuration")
    config = load_config()
    missing_values_config = config.get('missing_values', {})
    
    if not missing_values_config:
        logger.warning("No missing values configuration found")
    
    return missing_values_config

@log_exceptions(logger)
def get_outlier_config() -> Dict[str, Any]:
    """
    Get outlier detection configuration.
    
    Returns:
        Dict[str, Any]: Outlier detection configuration
    """
    logger.debug("Retrieving outlier detection configuration")
    config = load_config()
    outlier_config = config.get('outlier_detection', {})
    
    if not outlier_config:
        logger.warning("No outlier detection configuration found")
    
    return outlier_config

@log_exceptions(logger)
def get_binning_config() -> Dict[str, Any]:
    """
    Get feature binning configuration.
    
    Returns:
        Dict[str, Any]: Feature binning configuration
    """
    logger.debug("Retrieving feature binning configuration")
    config = load_config()
    binning_config = config.get('feature_binning', {})
    
    if not binning_config:
        logger.warning("No feature binning configuration found")
    
    return binning_config

@log_exceptions(logger)
def get_encoding_config() -> Dict[str, Any]:
    """
    Get feature encoding configuration.
    
    Returns:
        Dict[str, Any]: Feature encoding configuration
    """
    logger.debug("Retrieving feature encoding configuration")
    config = load_config()
    encoding_config = config.get('feature_encoding', {})
    
    if not encoding_config:
        logger.warning("No feature encoding configuration found")
    
    return encoding_config

@log_exceptions(logger)
def get_scaling_config() -> Dict[str, Any]:
    """
    Get feature scaling configuration.
    
    Returns:
        Dict[str, Any]: Feature scaling configuration
    """
    logger.debug("Retrieving feature scaling configuration")
    config = load_config()
    scaling_config = config.get('feature_scaling', {})
    
    if not scaling_config:
        logger.warning("No feature scaling configuration found")
    
    return scaling_config

@log_exceptions(logger)
def get_splitting_config() -> Dict[str, Any]:
    """
    Get data splitting configuration.
    
    Returns:
        Dict[str, Any]: Data splitting configuration
    """
    logger.debug("Retrieving data splitting configuration")
    config = load_config()
    splitting_config = config.get('data_splitting', {})
    
    if not splitting_config:
        logger.warning("No data splitting configuration found")
    
    return splitting_config

@log_exceptions(logger)
def get_training_config() -> Dict[str, Any]:
    """
    Get training configuration.
    
    Returns:
        Dict[str, Any]: Training configuration
    """
    logger.debug("Retrieving training configuration")
    config = load_config()
    training_config = config.get('training', {})
    
    if not training_config:
        logger.warning("No training configuration found")
    
    return training_config

@log_exceptions(logger)
def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration.
    
    Returns:
        Dict[str, Any]: Model configuration
    """
    logger.debug("Retrieving model configuration")
    config = load_config()
    model_config = config.get('model', {})
    
    if not model_config:
        logger.warning("No model configuration found")
    
    return model_config

@log_exceptions(logger)
def get_evaluation_config() -> Dict[str, Any]:
    """
    Get evaluation configuration.
    
    Returns:
        Dict[str, Any]: Evaluation configuration
    """
    logger.debug("Retrieving evaluation configuration")
    config = load_config()
    evaluation_config = config.get('evaluation', {})
    
    if not evaluation_config:
        logger.warning("No evaluation configuration found")
    
    return evaluation_config

@log_exceptions(logger)
def get_deployment_config() -> Dict[str, Any]:
    """
    Get deployment configuration.
    
    Returns:
        Dict[str, Any]: Deployment configuration
    """
    logger.debug("Retrieving deployment configuration")
    config = load_config()
    deployment_config = config.get('deployment', {})
    
    if not deployment_config:
        logger.warning("No deployment configuration found")
    
    return deployment_config

@log_exceptions(logger)
def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration.
    
    Returns:
        Dict[str, Any]: Logging configuration
    """
    logger.debug("Retrieving logging configuration")
    config = load_config()
    logging_config = config.get('logging', {})
    
    if not logging_config:
        logger.warning("No logging configuration found")
    
    return logging_config

@log_exceptions(logger)
def get_environment_config() -> Dict[str, Any]:
    """
    Get environment configuration.
    
    Returns:
        Dict[str, Any]: Environment configuration
    """
    logger.debug("Retrieving environment configuration")
    config = load_config()
    environment_config = config.get('environment', {})
    
    if not environment_config:
        logger.warning("No environment configuration found")
    
    return environment_config

@log_exceptions(logger)
def get_pipeline_config() -> Dict[str, Any]:
    """
    Get pipeline configuration.
    
    Returns:
        Dict[str, Any]: Pipeline configuration
    """
    logger.debug("Retrieving pipeline configuration")
    config = load_config()
    pipeline_config = config.get('pipeline', {})
    
    if not pipeline_config:
        logger.warning("No pipeline configuration found")
    
    return pipeline_config


def get_inference_config():
    config = load_config()
    return config.get('inference', {})


def get_config() ->Dict[str, Any]:
    return load_config()


def get_data_config() ->Dict[str, Any]:
    config = get_config()
    return config.get('data', {})


def get_preprocessing_config() ->Dict[str, Any]:
    config = get_config()
    return config.get('preprocessing', {})

def get_mlflow_config():
    config = load_config()
    return config.get('mlflow', {})


def get_selected_model_config() ->Dict[str, Any]:
    training_config = get_training_config()
    selected_model = training_config.get('selected_model', 'random_forest')
    model_types = training_config.get('model_types', {})
    return {'model_type': selected_model, 'model_config': model_types.get(
        selected_model, {}), 'training_strategy': training_config.get(
        'training_strategy', 'cv'), 'cv_folds': training_config.get(
        'cv_folds', 5), 'random_state': training_config.get('random_state', 42)
        }


def get_available_models() ->List[str]:
    training_config = get_training_config()
    return list(training_config.get('model_types', {}).keys())


def update_config(updates: Dict[str, Any]) ->None:
    config_path = CONFIG_FILE
    config = get_config()
    for key, value in updates.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def create_default_config() ->None:
    config_path = CONFIG_FILE
    if not os.path.exists(config_path):
        default_config = {'data': {'file_path':
            'data/raw/ChurnModelling.csv', 'target_column': 'Exited',
            'test_size': 0.2, 'random_state': 42}, 'preprocessing': {
            'handle_missing_values': True, 'handle_outliers': True,
            'feature_binning': True, 'feature_encoding': True,
            'feature_scaling': True}, 'training': {'selected_model':
            'random_forest', 'training_strategy': 'cv', 'cv_folds': 5,
            'random_state': 42}}
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        logger.info(f'Created default configuration file: {config_path}')


create_default_config()
