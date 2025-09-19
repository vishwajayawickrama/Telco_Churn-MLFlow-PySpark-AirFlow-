import os
import sys
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation class with logging and visualization.
    """
    
    def __init__(self, model: BaseEstimator, model_name: str):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            model_name (str): Name of the model for identification
        """
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        
        ProjectLogger.log_section_header(logger, f"INITIALIZING MODEL EVALUATOR FOR {model_name.upper()}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info("Evaluator ready for performance assessment")

    @log_exceptions(logger)
    def evaluate(
        self, 
        X_test,
        Y_test,
    ) :
        """
        Comprehensive model evaluation.
        
        Args:
            X_test (pd.DataFrame): Test features
            Y_test (pd.Series): Test targets
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        ProjectLogger.log_step_header(logger, "STEP", f"EVALUATING {self.model_name.upper()} MODEL")
        
        try:
            # Validate inputs
            if X_test.empty:
                raise ValueError("Test features (X_test) is empty")
            
            if Y_test.empty:
                raise ValueError("Test targets (Y_test) is empty")
            
            if len(X_test) != len(Y_test):
                raise ValueError(f"Feature and target lengths don't match: {len(X_test)} vs {len(Y_test)}")
            
            logger.info(f"Test data shape: {X_test.shape}")
            logger.info(f"Test target shape: {Y_test.shape}")
            
            # Make predictions
            logger.info("Generating predictions...")
            Y_pred = self.model.predict(X_test)
            
            # Core metrics
            cm = confusion_matrix(Y_test, Y_pred)
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred)
            recall = recall_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred)
            
            # Log results
            logger.info("Evaluation metrics:")
            logger.info(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1-Score: {f1:.4f}")
            
            # Log confusion matrix
            logger.info("Confusion Matrix:")
            logger.info(f"  {cm}")

            self.evaluation_result = {
                                    'cm': cm,
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1
                                }

            
            ProjectLogger.log_success_header(logger, "MODEL EVALUATION COMPLETED")
            
            return self.evaluation_results
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL EVALUATION")
            logger.error(f"Unexpected error: {str(e)}")
            raise