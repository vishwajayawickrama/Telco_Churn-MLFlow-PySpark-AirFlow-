"""
Model evaluation module for PySpark ML operations.
Provides comprehensive evaluation functionality with logging and metrics calculation.
"""

import logging
from typing import Dict, Optional, List, Tuple
import warnings
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, 
    MulticlassClassificationEvaluator
)
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from utils.logger import ProjectLogger, log_exceptions
from utils.spark_utils import get_spark_session

# Configure logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation class for PySpark ML with logging and metrics.
    """
    
    def __init__(self, model: PipelineModel, model_name: str, spark: Optional[SparkSession] = None):
        """
        Initialize model evaluator.
        
        Args:
            model (PipelineModel): Trained PySpark ML Pipeline model to evaluate
            model_name (str): Name of the model for identification
            spark: Optional SparkSession
        """
        self.model = model
        self.model_name = model_name
        self.spark = spark or get_spark_session()
        self.evaluation_results = {}
        
        ProjectLogger.log_section_header(logger, f"INITIALIZING PYSPARK MODEL EVALUATOR FOR {model_name.upper()}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Pipeline stages: {len(model.stages)}")
        logger.info("PySpark evaluator ready for performance assessment")

    @log_exceptions(logger)
    def evaluate(
        self, 
        test_df: DataFrame,
        target_column: str = "label"
    ) -> Dict:
        """
        Comprehensive model evaluation using PySpark ML evaluators.
        
        Args:
            test_df (DataFrame): Test DataFrame with features and target
            target_column (str): Name of the target column (default: "label")
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        ProjectLogger.log_step_header(logger, "STEP", f"EVALUATING {self.model_name.upper()} MODEL")
        
        try:
            # Validate inputs
            test_count = test_df.count()
            if test_count == 0:
                raise ValueError("Test DataFrame is empty")
            
            if target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in test DataFrame")
            
            logger.info(f"Test data size: {test_count} samples")
            logger.info(f"Test data columns: {len(test_df.columns)}")
            
            # Check target distribution in test data
            logger.info("Target distribution in test data:")
            target_dist = test_df.groupBy(target_column).count().collect()
            for row in target_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / test_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            # Make predictions
            logger.info("Generating predictions...")
            predictions_df = self.model.transform(test_df)
            
            # Cache predictions for performance
            predictions_df.cache()
            
            # Validate predictions were generated
            prediction_count = predictions_df.count()
            if prediction_count != test_count:
                logger.warning(f"Prediction count ({prediction_count}) differs from test count ({test_count})")
            
            # Check if we have required columns for evaluation
            required_cols = ["prediction", target_column]
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for evaluation: {missing_cols}")
            
            logger.info("Calculating evaluation metrics...")
            
            # Initialize evaluators
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=target_column,
                rawPredictionCol="rawPrediction" if "rawPrediction" in predictions_df.columns else "prediction"
            )
            
            multiclass_evaluator = MulticlassClassificationEvaluator(
                labelCol=target_column,
                predictionCol="prediction"
            )
            
            # Calculate binary classification metrics (if applicable)
            metrics = {}
            
            try:
                # AUC-ROC
                auc_roc = binary_evaluator.evaluate(predictions_df, {binary_evaluator.metricName: "areaUnderROC"})
                metrics['auc_roc'] = float(auc_roc)
                logger.info(f"  - AUC-ROC: {auc_roc:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
                metrics['auc_roc'] = None
            
            try:
                # AUC-PR
                auc_pr = binary_evaluator.evaluate(predictions_df, {binary_evaluator.metricName: "areaUnderPR"})
                metrics['auc_pr'] = float(auc_pr)
                logger.info(f"  - AUC-PR: {auc_pr:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate AUC-PR: {str(e)}")
                metrics['auc_pr'] = None
            
            # Calculate multiclass metrics
            accuracy = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "accuracy"})
            precision = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "weightedPrecision"})
            recall = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "weightedRecall"})
            f1 = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "f1"})
            
            metrics.update({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
            
            # Log multiclass metrics
            logger.info("Multiclass evaluation metrics:")
            logger.info(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            logger.info(f"  - Weighted Precision: {precision:.4f}")
            logger.info(f"  - Weighted Recall: {recall:.4f}")
            logger.info(f"  - F1-Score: {f1:.4f}")
            
            # Calculate confusion matrix
            logger.info("Calculating confusion matrix...")
            confusion_matrix = self._calculate_confusion_matrix(predictions_df, target_column)
            metrics['confusion_matrix'] = confusion_matrix
            
            # Log confusion matrix
            logger.info("Confusion Matrix:")
            for i, row in enumerate(confusion_matrix):
                logger.info(f"  Class {i}: {row}")
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(predictions_df, target_column)
            metrics['per_class_metrics'] = per_class_metrics
            
            # Log per-class metrics
            if per_class_metrics:
                logger.info("Per-class metrics:")
                for class_val, class_metrics in per_class_metrics.items():
                    logger.info(f"  Class {class_val}:")
                    logger.info(f"    - Precision: {class_metrics.get('precision', 'N/A'):.4f}")
                    logger.info(f"    - Recall: {class_metrics.get('recall', 'N/A'):.4f}")
                    logger.info(f"    - F1: {class_metrics.get('f1', 'N/A'):.4f}")
            
            # Additional statistics
            metrics.update({
                'test_samples': test_count,
                'model_name': self.model_name,
                'pipeline_stages': len(self.model.stages)
            })
            
            # Store results
            self.evaluation_results = metrics
            
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL EVALUATION COMPLETED")
            logger.info(f"Evaluation results stored with {len(self.evaluation_results)} metrics")
            
            # Clean up cache
            predictions_df.unpersist()
            
            return self.evaluation_results
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL EVALUATION")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _calculate_confusion_matrix(self, predictions_df: DataFrame, target_column: str) -> List[List[int]]:
        """
        Calculate confusion matrix from predictions DataFrame.
        
        Args:
            predictions_df (DataFrame): DataFrame with predictions and actual labels
            target_column (str): Name of the target column
            
        Returns:
            List[List[int]]: Confusion matrix as 2D list
        """
        try:
            # Get unique classes
            unique_labels = predictions_df.select(target_column).distinct().rdd.map(lambda row: row[0]).collect()
            unique_predictions = predictions_df.select("prediction").distinct().rdd.map(lambda row: row[0]).collect()
            
            # Get all unique classes (union of labels and predictions)
            all_classes = sorted(set(unique_labels + unique_predictions))
            n_classes = len(all_classes)
            
            logger.info(f"Confusion matrix classes: {all_classes}")
            
            # Initialize confusion matrix
            confusion_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
            
            # Calculate confusion matrix
            for i, true_class in enumerate(all_classes):
                for j, pred_class in enumerate(all_classes):
                    count = predictions_df.filter(
                        (F.col(target_column) == true_class) & 
                        (F.col("prediction") == pred_class)
                    ).count()
                    confusion_matrix[i][j] = count
            
            return confusion_matrix
            
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {str(e)}")
            return []

    def _calculate_per_class_metrics(self, predictions_df: DataFrame, target_column: str) -> Dict:
        """
        Calculate precision, recall, and F1 for each class.
        
        Args:
            predictions_df (DataFrame): DataFrame with predictions and actual labels
            target_column (str): Name of the target column
            
        Returns:
            Dict: Per-class metrics
        """
        try:
            # Get unique classes
            unique_classes = predictions_df.select(target_column).distinct().rdd.map(lambda row: row[0]).collect()
            per_class_metrics = {}
            
            for class_val in unique_classes:
                # True Positives
                tp = predictions_df.filter(
                    (F.col(target_column) == class_val) & 
                    (F.col("prediction") == class_val)
                ).count()
                
                # False Positives
                fp = predictions_df.filter(
                    (F.col(target_column) != class_val) & 
                    (F.col("prediction") == class_val)
                ).count()
                
                # False Negatives
                fn = predictions_df.filter(
                    (F.col(target_column) == class_val) & 
                    (F.col("prediction") != class_val)
                ).count()
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_class_metrics[class_val] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
            
            return per_class_metrics
            
        except Exception as e:
            logger.error(f"Error calculating per-class metrics: {str(e)}")
            return {}

    @log_exceptions(logger)
    def evaluate_with_predictions_df(self, predictions_df: DataFrame, target_column: str = "label") -> Dict:
        """
        Evaluate model using a DataFrame that already contains predictions.
        
        Args:
            predictions_df (DataFrame): DataFrame with both actual labels and predictions
            target_column (str): Name of the target column
            
        Returns:
            Dict: Evaluation results
        """
        ProjectLogger.log_step_header(logger, "STEP", f"EVALUATING {self.model_name.upper()} WITH PREDICTIONS DF")
        
        try:
            # Validate required columns
            required_cols = ["prediction", target_column]
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            test_count = predictions_df.count()
            logger.info(f"Evaluating {test_count} predictions")
            
            # Use the existing evaluation logic but skip the prediction step
            return self._calculate_metrics_from_predictions(predictions_df, target_column)
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "ERROR IN PREDICTIONS DF EVALUATION")
            logger.error(f"Error: {str(e)}")
            raise

    def _calculate_metrics_from_predictions(self, predictions_df: DataFrame, target_column: str) -> Dict:
        """Helper method to calculate metrics from a predictions DataFrame."""
        # This would use the same logic as in the main evaluate method
        # but without the model.transform step
        # For brevity, I'll reference the main evaluation logic
        
        # Initialize evaluators
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=target_column,
            rawPredictionCol="rawPrediction" if "rawPrediction" in predictions_df.columns else "prediction"
        )
        
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol=target_column,
            predictionCol="prediction"
        )
        
        # Calculate metrics (same as main evaluate method)
        metrics = {}
        
        try:
            auc_roc = binary_evaluator.evaluate(predictions_df, {binary_evaluator.metricName: "areaUnderROC"})
            metrics['auc_roc'] = float(auc_roc)
        except:
            metrics['auc_roc'] = None
        
        accuracy = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "accuracy"})
        precision = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "weightedPrecision"})
        recall = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "weightedRecall"})
        f1 = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "f1"})
        
        metrics.update({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': self._calculate_confusion_matrix(predictions_df, target_column),
            'per_class_metrics': self._calculate_per_class_metrics(predictions_df, target_column)
        })
        
        return metrics

    def get_evaluation_results(self) -> Dict:
        """
        Get stored evaluation results.
        
        Returns:
            Dict: Evaluation results
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate() first.")
            return {}
        
        return self.evaluation_results.copy()


def create_model_evaluator(model: PipelineModel, model_name: str, spark: Optional[SparkSession] = None) -> ModelEvaluator:
    """
    Factory function to create a ModelEvaluator instance.
    
    Args:
        model (PipelineModel): Trained PySpark ML Pipeline model
        model_name (str): Name of the model for identification
        spark: Optional SparkSession
        
    Returns:
        ModelEvaluator: Configured model evaluator
    """
    return ModelEvaluator(model, model_name, spark)