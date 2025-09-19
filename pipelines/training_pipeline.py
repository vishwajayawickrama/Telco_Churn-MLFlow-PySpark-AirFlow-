import os
import sys
import pandas as pd
import pickle
from data_pipeline import data_pipeline
from typing import Dict, Any, Tuple, Optional

# Add src and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import src modules
from model_building import RandomForestModelBuilder, XGBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

# Import utils
from config import get_model_config, get_data_paths
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)

@log_exceptions(logger)
def training_pipeline(
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
        
        logger.info("Starting model training...")
        model, _ = trainer.train(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
        )
        logger.info("Model training completed successfully")
        
        # Step 5: Model saving
        ProjectLogger.log_step_header(logger, "STEP", "5: MODEL SAVING")
        
        # Ensure model directory exists
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")
        
        logger.info(f"Saving trained model to: {model_path}")
        trainer.save_model(model, model_path)
        
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
        
        logger.info("Evaluating model performance...")
        eval_results = evaluator.evaluate(X_test, Y_test)
        
        # Log evaluation results
        eval_result_copy = eval_results.copy()
        if 'cm' in eval_result_copy:
            del eval_result_copy['cm']  # Remove confusion matrix for cleaner logging
        
        logger.info("Model evaluation completed:")
        for metric, value in eval_result_copy.items():
            if isinstance(value, float):
                logger.info(f"  - {metric}: {value:.4f}")
            else:
                logger.info(f"  - {metric}: {value}")
        
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