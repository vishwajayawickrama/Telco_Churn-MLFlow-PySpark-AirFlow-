"""
Model training module for PySpark ML Pipeline operations.
Provides comprehensive training functionality with logging and validation.
"""

import os
import logging
from typing import Tuple, Any, Optional, Dict, List, Union
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import (
    GBTClassifier, RandomForestClassifier, 
    LogisticRegression, DecisionTreeClassifier
)

from utils.logger import ProjectLogger, log_exceptions
from utils.spark_utils import get_spark_session

# Configure logger
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training class for PySpark ML with logging and validation.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize model trainer.
        
        Args:
            spark: Optional SparkSession
        """
        self.spark = spark or get_spark_session()
        ProjectLogger.log_section_header(logger, "INITIALIZING PYSPARK MODEL TRAINER")
        logger.info("PySpark model trainer ready for training operations")

    @log_exceptions(logger)
    def train(
        self, 
        model,
        train_df: DataFrame,
        feature_columns: List[str],
        target_column: str,
        model_save_path: Optional[str] = None
    ) -> Tuple[PipelineModel, Dict[str, float]]:
        """
        Train a PySpark ML model with pipeline approach.
        
        Args:
            model: PySpark ML classifier
            train_df (DataFrame): Training DataFrame
            feature_columns (List[str]): List of feature column names
            target_column (str): Name of the target column
            model_save_path (Optional[str]): Path to save the trained model
            
        Returns:
            Tuple: (trained_pipeline_model, metrics)
        """
        ProjectLogger.log_step_header(logger, "STEP", "TRAINING MODEL WITH PYSPARK ML PIPELINE")
        
        try:
            # Validate inputs
            if train_df.count() == 0:
                raise ValueError("Training DataFrame is empty")
            
            if not feature_columns:
                raise ValueError("Feature columns list is empty")
            
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            # Validate feature columns exist
            missing_features = [col for col in feature_columns if col not in train_df.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")
            
            train_count = train_df.count()
            logger.info(f"Training data size: {train_count} samples")
            logger.info(f"Feature columns: {len(feature_columns)} features")
            logger.info(f"Target column: {target_column}")
            logger.info(f"Model type: {type(model).__name__}")
            
            # Check target distribution
            logger.info("Target distribution in training data:")
            target_dist = train_df.groupBy(target_column).count().collect()
            for row in target_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / train_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            # Build ML Pipeline
            logger.info("Building ML Pipeline...")
            
            # Vector assembler to combine features
            vector_assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol="features",
                handleInvalid="keep"
            )
            
            # String indexer for target if needed (convert string labels to numeric)
            target_indexer = StringIndexer(
                inputCol=target_column,
                outputCol="label",
                handleInvalid="keep"
            )
            
            # Set up model with proper input/output columns
            model.setFeaturesCol("features")
            model.setLabelCol("label")
            model.setPredictionCol("prediction")
            
            # Create pipeline
            pipeline_stages = [vector_assembler, target_indexer, model]
            pipeline = Pipeline(stages=pipeline_stages)
            
            logger.info(f"Pipeline stages: {len(pipeline_stages)}")
            logger.info("  1. VectorAssembler - Combine features")
            logger.info("  2. StringIndexer - Convert target to numeric labels")
            logger.info(f"  3. {type(model).__name__} - Classification model")
            
            # Start training
            training_start = datetime.now()
            logger.info(f"Starting model training at: {training_start}")
            
            # Fit the pipeline
            logger.info("Fitting ML Pipeline to training data...")
            pipeline_model = pipeline.fit(train_df)
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Make predictions on training data for evaluation
            logger.info("Making predictions on training data for evaluation...")
            train_predictions = pipeline_model.transform(train_df)
            
            # Calculate training metrics
            logger.info("Calculating training metrics...")
            
            # Binary classification evaluator
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            
            # Multiclass evaluator
            multiclass_evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction"
            )
            
            # Calculate various metrics
            auc = binary_evaluator.evaluate(train_predictions)
            accuracy = multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "accuracy"})
            precision = multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
            recall = multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
            f1 = multiclass_evaluator.evaluate(train_predictions, {multiclass_evaluator.metricName: "f1"})
            
            metrics = {
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'training_duration_seconds': training_duration,
                'training_samples': train_count
            }
            
            logger.info("Training metrics:")
            logger.info(f"  - AUC: {auc:.4f}")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1 Score: {f1:.4f}")
            
            # Save model if path provided
            if model_save_path:
                logger.info(f"Saving model to: {model_save_path}")
                self.save_model(pipeline_model, model_save_path)
                metrics['model_save_path'] = model_save_path
            
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL TRAINING COMPLETED")
            
            return pipeline_model, metrics
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL TRAINING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def train_with_cross_validation(
        self,
        model,
        train_df: DataFrame,
        feature_columns: List[str],
        target_column: str,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 3,
        metric_name: str = "areaUnderROC",
        model_save_path: Optional[str] = None
    ) -> Tuple[PipelineModel, Dict[str, Any]]:
        """
        Train a model with cross-validation and hyperparameter tuning.
        
        Args:
            model: PySpark ML classifier
            train_df (DataFrame): Training DataFrame
            feature_columns (List[str]): List of feature column names
            target_column (str): Name of the target column
            param_grid (Optional[Dict]): Parameter grid for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            metric_name (str): Evaluation metric name
            model_save_path (Optional[str]): Path to save the best model
            
        Returns:
            Tuple: (best_model, cv_results)
        """
        ProjectLogger.log_step_header(logger, "STEP", "TRAINING MODEL WITH CROSS-VALIDATION")
        
        try:
            logger.info(f"Cross-validation folds: {cv_folds}")
            logger.info(f"Evaluation metric: {metric_name}")
            
            # Build pipeline (same as regular training)
            vector_assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol="features",
                handleInvalid="keep"
            )
            
            target_indexer = StringIndexer(
                inputCol=target_column,
                outputCol="label",
                handleInvalid="keep"
            )
            
            model.setFeaturesCol("features")
            model.setLabelCol("label")
            model.setPredictionCol("prediction")
            
            pipeline = Pipeline(stages=[vector_assembler, target_indexer, model])
            
            # Set up parameter grid
            if param_grid is None:
                # Default parameter grid based on model type
                param_grid = self._get_default_param_grid(model, pipeline)
            else:
                # Convert user param grid to ParamGridBuilder format
                param_grid_builder = ParamGridBuilder()
                for param_name, values in param_grid.items():
                    # This would need to be adapted based on actual parameter structure
                    param_grid_builder = param_grid_builder.addGrid(getattr(model, param_name), values)
                param_grid = param_grid_builder.build()
            
            logger.info(f"Parameter grid size: {len(param_grid)} combinations")
            
            # Set up evaluator
            if metric_name in ["areaUnderROC", "areaUnderPR"]:
                evaluator = BinaryClassificationEvaluator(
                    labelCol="label",
                    rawPredictionCol="rawPrediction",
                    metricName=metric_name
                )
            else:
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName=metric_name
                )
            
            # Set up cross-validator
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=cv_folds,
                seed=42
            )
            
            # Train with cross-validation
            training_start = datetime.now()
            logger.info("Starting cross-validation training...")
            
            cv_model = cv.fit(train_df)
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            logger.info(f"Cross-validation completed in {training_duration:.2f} seconds")
            
            # Get best model and results
            best_model = cv_model.bestModel
            best_metric = max(cv_model.avgMetrics)
            
            cv_results = {
                'best_metric_value': best_metric,
                'metric_name': metric_name,
                'cv_folds': cv_folds,
                'avg_metrics': cv_model.avgMetrics,
                'training_duration_seconds': training_duration,
                'param_grid_size': len(param_grid)
            }
            
            logger.info("Cross-validation results:")
            logger.info(f"  - Best {metric_name}: {best_metric:.4f}")
            logger.info(f"  - Parameter combinations tested: {len(param_grid)}")
            
            # Save best model if path provided
            if model_save_path:
                logger.info(f"Saving best model to: {model_save_path}")
                self.save_model(best_model, model_save_path)
                cv_results['model_save_path'] = model_save_path
            
            ProjectLogger.log_success_header(logger, "CROSS-VALIDATION TRAINING COMPLETED")
            
            return best_model, cv_results
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN CV TRAINING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _get_default_param_grid(self, model, pipeline) -> List:
        """Get default parameter grid for common models."""
        param_grid_builder = ParamGridBuilder()
        
        if isinstance(model, RandomForestClassifier):
            param_grid_builder = param_grid_builder \
                .addGrid(model.numTrees, [10, 20, 50]) \
                .addGrid(model.maxDepth, [5, 10, 15])
        elif isinstance(model, GBTClassifier):
            param_grid_builder = param_grid_builder \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .addGrid(model.maxDepth, [5, 10])
        elif isinstance(model, LogisticRegression):
            param_grid_builder = param_grid_builder \
                .addGrid(model.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0])
        elif isinstance(model, DecisionTreeClassifier):
            param_grid_builder = param_grid_builder \
                .addGrid(model.maxDepth, [5, 10, 15, 20])
        
        return param_grid_builder.build()

    @log_exceptions(logger)
    def save_model(self, model: PipelineModel, filepath: str) -> None:
        """
        Save a trained PySpark ML Pipeline model to file.
        
        Args:
            model (PipelineModel): Trained PySpark ML Pipeline model to save
            filepath (str): Path to save the model
        """
        ProjectLogger.log_step_header(logger, "STEP", "SAVING TRAINED PYSPARK MODEL")
        
        try:
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            if not filepath:
                raise ValueError("Filepath cannot be empty")
            
            if not isinstance(model, PipelineModel):
                raise ValueError("Model must be a PySpark ML PipelineModel")
            
            # Create directory if it doesn't exist
            os.makedirs(filepath, exist_ok=True)
            logger.info(f"Model directory created/verified: {filepath}")
            
            logger.info(f"Saving PySpark ML Pipeline model")
            logger.info(f"Model type: {type(model).__name__}")
            logger.info(f"Save path: {filepath}")
            logger.info(f"Pipeline stages: {len(model.stages)}")
            
            # Save the model
            model.write().overwrite().save(filepath)
            
            # Verify save
            if os.path.exists(filepath):
                # Calculate directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(filepath):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                
                logger.info(f"Model saved successfully")
                logger.info(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
            else:
                raise Exception("Model directory was not created")
            
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL SAVED SUCCESSFULLY")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL SAVE VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL SAVING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_model(self, filepath: str) -> PipelineModel:
        """
        Load a trained PySpark ML Pipeline model from file.
        
        Args:
            filepath (str): Path to load the model from
            
        Returns:
            PipelineModel: Loaded PySpark ML Pipeline model
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING TRAINED PYSPARK MODEL")
        
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                raise ValueError(f"Model directory not found: {filepath}")
            
            # Get directory info
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(filepath):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            logger.info(f"Loading model from: {filepath}")
            logger.info(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
            
            # Load model
            model = PipelineModel.load(filepath)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(model).__name__}")
            logger.info(f"Pipeline stages: {len(model.stages)}")
            
            ProjectLogger.log_success_header(logger, "PYSPARK MODEL LOADED SUCCESSFULLY")
            
            return model
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOAD VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


def create_model_trainer(spark: Optional[SparkSession] = None) -> ModelTrainer:
    """
    Factory function to create a ModelTrainer instance.
    
    Args:
        spark: Optional SparkSession
        
    Returns:
        ModelTrainer: Configured model trainer
    """
    return ModelTrainer(spark)