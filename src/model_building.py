import os
import sys
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from abc import ABC, abstractmethod

# PySpark ML imports
from pyspark.ml.classification import (
    GBTClassifier, 
    RandomForestClassifier, 
    LogisticRegression,
    DecisionTreeClassifier
)
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions
from spark_utils import get_spark_session

# Initialize logger
logger = get_logger(__name__)


class BaseModelBuilder(ABC):
    """
    Abstract base class for PySpark ML model builders.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize base model builder.
        
        Args:
            model_name (str): Name of the model
            **kwargs: Model parameters
        """
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        self.build_timestamp = None
        self.spark = get_spark_session()
        
        ProjectLogger.log_section_header(logger, f"INITIALIZING {model_name.upper()} MODEL BUILDER")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Model parameters ({len(self.model_params)}):")
        
        for param, value in self.model_params.items():
            logger.info(f"  - {param}: {value}")

    @abstractmethod
    def build_model(self):
        """
        Abstract method to build the PySpark ML model.
        
        Returns:
            PySpark ML Classifier: Built model instance
        """
        pass
    
    @log_exceptions(logger)
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the built model.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        if self.model is None:
            logger.warning("Model not built yet. Call build_model() first.")
            return {}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'build_timestamp': self.build_timestamp,
            'parameters': self.model_params
        }
        
        logger.info(f"Model info retrieved for {self.model_name}")
        return info
    
    @log_exceptions(logger)
    def save_model(self, filepath: str):
        """
        Save the PySpark ML model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Call build_model() first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save PySpark ML model
            self.model.write().overwrite().save(filepath)
            
            logger.info(f"Model saved successfully to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    @log_exceptions(logger) 
    def load_model(self, filepath: str):
        """
        Load a previously saved PySpark ML model.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            # This is a placeholder - actual loading depends on model type
            # In practice, you'd need to know the model type to load correctly
            logger.warning("Model loading requires specific implementation per model type")
            logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise


class GBTModelBuilder(BaseModelBuilder):
    """
    Gradient Boosted Trees model builder for PySpark ML.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize GBT model builder.
        
        Args:
            **kwargs: GBT model parameters
        """
        # Set default parameters for GBT
        default_params = {
            'max_iter': 100,
            'max_depth': 10,
            'step_size': 0.1,
            'seed': 42,
            'features_col': 'features',
            'label_col': 'label',
            'prediction_col': 'prediction'
        }
        default_params.update(kwargs)
        
        super().__init__("GradientBoostedTrees", **default_params)
    
    @log_exceptions(logger)
    def build_model(self):
        """
        Build Gradient Boosted Trees classifier.
        
        Returns:
            GBTClassifier: Built GBT model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING GRADIENT BOOSTED TREES MODEL")
        
        try:
            self.model = GBTClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("GBT model built successfully")
            logger.info(f"Max iterations: {self.model.getMaxIter()}")
            logger.info(f"Max depth: {self.model.getMaxDepth()}")
            logger.info(f"Step size: {self.model.getStepSize()}")
            
            ProjectLogger.log_success_header(logger, "GBT MODEL BUILD COMPLETED")
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "GBT MODEL BUILD FAILED")
            logger.error(f"Error building GBT model: {str(e)}")
            raise


class RandomForestModelBuilder(BaseModelBuilder):
    """
    Random Forest model builder for PySpark ML.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model builder.
        
        Args:
            **kwargs: Random Forest model parameters
        """
        # Set default parameters for Random Forest
        default_params = {
            'num_trees': 100,
            'max_depth': 10,
            'seed': 42,
            'feature_subset_strategy': 'auto',
            'features_col': 'features',
            'label_col': 'label',
            'prediction_col': 'prediction'
        }
        default_params.update(kwargs)
        
        super().__init__("RandomForest", **default_params)
    
    @log_exceptions(logger)
    def build_model(self):
        """
        Build Random Forest classifier.
        
        Returns:
            RandomForestClassifier: Built Random Forest model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING RANDOM FOREST MODEL")
        
        try:
            self.model = RandomForestClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("Random Forest model built successfully")
            logger.info(f"Number of trees: {self.model.getNumTrees()}")
            logger.info(f"Max depth: {self.model.getMaxDepth()}")
            logger.info(f"Feature subset strategy: {self.model.getFeatureSubsetStrategy()}")
            
            ProjectLogger.log_success_header(logger, "RANDOM FOREST MODEL BUILD COMPLETED")
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "RANDOM FOREST MODEL BUILD FAILED")
            logger.error(f"Error building Random Forest model: {str(e)}")
            raise


class LogisticRegressionModelBuilder(BaseModelBuilder):
    """
    Logistic Regression model builder for PySpark ML.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model builder.
        
        Args:
            **kwargs: Logistic Regression model parameters
        """
        # Set default parameters for Logistic Regression
        default_params = {
            'max_iter': 1000,
            'reg_param': 0.01,
            'elastic_net_param': 0.0,
            'family': 'binomial',
            'features_col': 'features',
            'label_col': 'label',
            'prediction_col': 'prediction'
        }
        default_params.update(kwargs)
        
        super().__init__("LogisticRegression", **default_params)
    
    @log_exceptions(logger)
    def build_model(self):
        """
        Build Logistic Regression classifier.
        
        Returns:
            LogisticRegression: Built Logistic Regression model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING LOGISTIC REGRESSION MODEL")
        
        try:
            self.model = LogisticRegression(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("Logistic Regression model built successfully")
            logger.info(f"Max iterations: {self.model.getMaxIter()}")
            logger.info(f"Regularization parameter: {self.model.getRegParam()}")
            logger.info(f"Family: {self.model.getFamily()}")
            
            ProjectLogger.log_success_header(logger, "LOGISTIC REGRESSION MODEL BUILD COMPLETED")
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "LOGISTIC REGRESSION MODEL BUILD FAILED")
            logger.error(f"Error building Logistic Regression model: {str(e)}")
            raise


class DecisionTreeModelBuilder(BaseModelBuilder):
    """
    Decision Tree model builder for PySpark ML.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Decision Tree model builder.
        
        Args:
            **kwargs: Decision Tree model parameters
        """
        # Set default parameters for Decision Tree
        default_params = {
            'max_depth': 10,
            'seed': 42,
            'impurity': 'gini',
            'features_col': 'features',
            'label_col': 'label',
            'prediction_col': 'prediction'
        }
        default_params.update(kwargs)
        
        super().__init__("DecisionTree", **default_params)
    
    @log_exceptions(logger)
    def build_model(self):
        """
        Build Decision Tree classifier.
        
        Returns:
            DecisionTreeClassifier: Built Decision Tree model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING DECISION TREE MODEL")
        
        try:
            self.model = DecisionTreeClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("Decision Tree model built successfully")
            logger.info(f"Max depth: {self.model.getMaxDepth()}")
            logger.info(f"Impurity: {self.model.getImpurity()}")
            
            ProjectLogger.log_success_header(logger, "DECISION TREE MODEL BUILD COMPLETED")
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "DECISION TREE MODEL BUILD FAILED")
            logger.error(f"Error building Decision Tree model: {str(e)}")
            raise


class ModelFactory:
    """
    Factory class for creating PySpark ML model builders.
    """
    
    _builders = {
        'gbt_classifier': GBTModelBuilder,
        'random_forest_classifier': RandomForestModelBuilder, 
        'logistic_regression': LogisticRegressionModelBuilder,
        'decision_tree_classifier': DecisionTreeModelBuilder
    }
    
    @staticmethod
    @log_exceptions(logger)
    def create_model_builder(model_type: str, **kwargs) -> BaseModelBuilder:
        """
        Create a model builder based on the specified type.
        
        Args:
            model_type (str): Type of model to build
            **kwargs: Model parameters
            
        Returns:
            BaseModelBuilder: Model builder instance
        """
        ProjectLogger.log_section_header(logger, f"CREATING MODEL BUILDER: {model_type.upper()}")
        
        if model_type not in ModelFactory._builders:
            available_types = list(ModelFactory._builders.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
        
        builder_class = ModelFactory._builders[model_type]
        builder = builder_class(**kwargs)
        
        logger.info(f"Model builder created: {type(builder).__name__}")
        return builder
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List[str]: Available model types
        """
        return list(ModelFactory._builders.keys())


# Convenience functions for backward compatibility
def GBTModelBuilder(**kwargs):
    """Create GBT model builder.""" 
    return ModelFactory.create_model_builder('gbt_classifier', **kwargs)

def RandomForestModelBuilder(**kwargs):
    """Create Random Forest model builder."""
    return ModelFactory.create_model_builder('random_forest_classifier', **kwargs)

def LogisticRegressionModelBuilder(**kwargs):
    """Create Logistic Regression model builder."""
    return ModelFactory.create_model_builder('logistic_regression', **kwargs)

def DecisionTreeModelBuilder(**kwargs):
    """Create Decision Tree model builder."""
    return ModelFactory.create_model_builder('decision_tree_classifier', **kwargs)


# For legacy compatibility
XGBoostModelBuilder = GBTModelBuilder  # Map XGBoost to GBT

    @log_exceptions(logger)
    def save_model(self, filepath: str, create_dirs: bool = True) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
            create_dirs (bool): Whether to create directories if they don't exist
            
        Raises:
            ValueError: If no model is built or filepath is invalid
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "SAVING MODEL")
        
        try:
            # Validate model exists
            if self.model is None:
                raise ValueError("No model to save. Build the model first using build_model()")
            
            # Validate filepath
            if not filepath:
                raise ValueError("Filepath cannot be empty")
            
            # Create directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                logger.info(f"Directory created/verified: {os.path.dirname(filepath)}")
            
            # Get file size before saving (if exists)
            file_existed = os.path.exists(filepath)
            old_size = os.path.getsize(filepath) if file_existed else 0
            
            logger.info(f"Saving {self.model_name} model to: {filepath}")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Verify save and get file info
            if os.path.exists(filepath):
                new_size = os.path.getsize(filepath)
                logger.info(f"Model saved successfully")
                logger.info(f"File size: {new_size:,} bytes ({new_size / 1024 / 1024:.2f} MB)")
                
                if file_existed:
                    size_diff = new_size - old_size
                    logger.info(f"Size change: {size_diff:+,} bytes")
                
                # Save metadata
                self._save_model_metadata(filepath)
                
            else:
                raise Exception("Model file was not created successfully")
            
            ProjectLogger.log_success_header(logger, "MODEL SAVED SUCCESSFULLY")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL SAVE VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL SAVING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @log_exceptions(logger)
    def load_model(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to load the model from
            
        Raises:
            ValueError: If file doesn't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "LOADING MODEL")
        
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                raise ValueError(f"Model file not found: {filepath}")
            
            # Get file info
            file_size = os.path.getsize(filepath)
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            logger.info(f"Loading model from: {filepath}")
            logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"Last modified: {file_modified}")
            
            # Load model
            self.model = joblib.load(filepath)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Load metadata if available
            self._load_model_metadata(filepath)
            
            ProjectLogger.log_success_header(logger, "MODEL LOADED SUCCESSFULLY")
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "MODEL LOAD VALIDATION ERROR")
            logger.error(f"Validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MODEL LOADING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def _save_model_metadata(self, filepath: str) -> None:
        """Save model metadata to a companion file."""
        try:
            metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            metadata = {
                'model_name': self.model_name,
                'model_type': type(self.model).__name__,
                'model_params': self.model_params,
                'build_timestamp': self.build_timestamp,
                'save_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Could not save model metadata: {str(e)}")
    
    def _load_model_metadata(self, filepath: str) -> None:
        """Load model metadata from companion file."""
        try:
            metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                logger.info("Model metadata loaded:")
                logger.info(f"  - Build timestamp: {metadata.get('build_timestamp', 'Unknown')}")
                logger.info(f"  - Save timestamp: {metadata.get('save_timestamp', 'Unknown')}")
                
        except Exception as e:
            logger.warning(f"Could not load model metadata: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {'model_name': self.model_name, 'model': None, 'status': 'Not built'}
        
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'model_params': self.model_params,
            'build_timestamp': self.build_timestamp,
            'status': 'Built'
        }


class RandomForestModelBuilder(BaseModelBuilder):
    """
    Random Forest model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model builder.
        
        Args:
            **kwargs: Random Forest parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build Random Forest classifier.
        
        Returns:
            RandomForestClassifier: Built model
            
        Raises:
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING RANDOM FOREST MODEL")
        
        try:
            logger.info("Creating Random Forest classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = RandomForestClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("Random Forest model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Min samples split: {self.model.min_samples_split}")
            logger.info(f"  - Min samples leaf: {self.model.min_samples_leaf}")
            logger.info(f"  - Random state: {self.model.random_state}")
            
            ProjectLogger.log_success_header(logger, "RANDOM FOREST MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN RANDOM FOREST BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class XGBoostModelBuilder(BaseModelBuilder):
    """
    XGBoost model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model builder.
        
        Args:
            **kwargs: XGBoost parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build XGBoost classifier.
        
        Returns:
            XGBClassifier: Built model
            
        Raises:
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING XGBOOST MODEL")
        
        try:
            logger.info("Creating XGBoost classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = XGBClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("XGBoost model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Learning rate: {self.model.learning_rate}")
            logger.info(f"  - Random state: {self.model.random_state}")
            
            ProjectLogger.log_success_header(logger, "XGBOOST MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN XGBOOST BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class LightGBMModelBuilder(BaseModelBuilder):
    """
    LightGBM model builder.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM model builder.
        
        Args:
            **kwargs: LightGBM parameters
        """
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'verbose': -1  # Suppress LightGBM warnings
        }
        default_params.update(kwargs)
        super().__init__('LightGBM', **default_params)

    @log_exceptions(logger)
    def build_model(self):
        """
        Build LightGBM classifier.
        
        Returns:
            LGBMClassifier: Built model
        """
        ProjectLogger.log_step_header(logger, "STEP", "BUILDING LIGHTGBM MODEL")
        
        try:
            logger.info("Creating LightGBM classifier...")
            logger.info(f"Parameters: {self.model_params}")
            
            # Build model
            self.model = LGBMClassifier(**self.model_params)
            self.build_timestamp = datetime.now().isoformat()
            
            logger.info("LightGBM model created successfully")
            logger.info(f"Model details:")
            logger.info(f"  - Max depth: {self.model.max_depth}")
            logger.info(f"  - N estimators: {self.model.n_estimators}")
            logger.info(f"  - Learning rate: {self.model.learning_rate}")
            
            ProjectLogger.log_success_header(logger, "LIGHTGBM MODEL BUILT")
            
            return self.model
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN LIGHTGBM BUILDING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
