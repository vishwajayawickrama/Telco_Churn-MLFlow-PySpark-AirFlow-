import os
import sys
import pandas as pd
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class SplitType(str, Enum):
    """Enumeration for data splitting types."""
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'


class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies.
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_train, X_test, Y_train, Y_test)
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Simple train-test split strategy using sklearn's train_test_split.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: Optional[int] = 42):
        """
        Initialize simple train-test split strategy.
        
        Args:
            test_size (float): Proportion of dataset for test set (default: 0.2)
            random_state (Optional[int]): Random state for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        
        ProjectLogger.log_section_header(logger, "INITIALIZING SIMPLE TRAIN-TEST SPLIT STRATEGY")
        logger.info(f"Test size: {self.test_size} ({self.test_size * 100}%)")
        logger.info(f"Train size: {1 - self.test_size} ({(1 - self.test_size) * 100}%)")
        logger.info(f"Random state: {self.random_state}")

    @log_exceptions(logger)
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using simple random sampling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_train, X_test, Y_train, Y_test)
            
        Raises:
            ValueError: If DataFrame is empty or target column doesn't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "SPLITTING DATA WITH SIMPLE STRATEGY")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            logger.info(f"Total dataset size: {len(df)} samples")
            logger.info(f"Features: {len(df.columns) - 1} columns")
            logger.info(f"Target column: {target_column}")
            
            # Check target column distribution
            target_distribution = df[target_column].value_counts()
            logger.info(f"Target distribution:")
            for value, count in target_distribution.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            # Check for missing values
            missing_in_features = df.drop(columns=[target_column]).isnull().sum().sum()
            missing_in_target = df[target_column].isnull().sum()
            
            if missing_in_features > 0:
                logger.warning(f"Found {missing_in_features} missing values in features")
            
            if missing_in_target > 0:
                logger.warning(f"Found {missing_in_target} missing values in target column")
                logger.warning("Missing target values may affect splitting")
            
            # Prepare features and target
            Y = df[target_column]
            X = df.drop(columns=[target_column])
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {Y.shape}")
            
            # Perform split
            logger.info("Performing train-test split...")
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, 
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Log split results
            logger.info("Split completed successfully:")
            logger.info(f"  - Training set: X_train {X_train.shape}, Y_train {Y_train.shape}")
            logger.info(f"  - Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")
            
            # Verify split proportions
            actual_test_proportion = len(X_test) / len(df)
            logger.info(f"  - Actual test proportion: {actual_test_proportion:.3f} (target: {self.test_size})")
            
            # Check target distribution in splits
            logger.info("Target distribution in training set:")
            train_distribution = Y_train.value_counts()
            for value, count in train_distribution.items():
                percentage = (count / len(Y_train)) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            logger.info("Target distribution in test set:")
            test_distribution = Y_test.value_counts()
            for value, count in test_distribution.items():
                percentage = (count / len(Y_test)) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            ProjectLogger.log_success_header(logger, "SIMPLE DATA SPLITTING COMPLETED")
            
            return X_train, X_test, Y_train, Y_test
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN DATA SPLITTING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Stratified train-test split strategy to maintain target distribution.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: Optional[int] = 42):
        """
        Initialize stratified train-test split strategy.
        
        Args:
            test_size (float): Proportion of dataset for test set (default: 0.2)
            random_state (Optional[int]): Random state for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        
        ProjectLogger.log_section_header(logger, "INITIALIZING STRATIFIED TRAIN-TEST SPLIT STRATEGY")
        logger.info(f"Test size: {self.test_size} ({self.test_size * 100}%)")
        logger.info(f"Train size: {1 - self.test_size} ({(1 - self.test_size) * 100}%)")
        logger.info(f"Random state: {self.random_state}")
        logger.info("Method: Maintains target distribution across train/test splits")

    @log_exceptions(logger)
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using stratified sampling to maintain target distribution.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_train, X_test, Y_train, Y_test)
        """
        ProjectLogger.log_step_header(logger, "STEP", "SPLITTING DATA WITH STRATIFIED STRATEGY")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            logger.info(f"Total dataset size: {len(df)} samples")
            
            # Prepare features and target
            Y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Check if stratification is possible
            target_counts = Y.value_counts()
            min_class_size = target_counts.min()
            min_samples_needed = int(1 / self.test_size) + 1
            
            if min_class_size < min_samples_needed:
                logger.warning(f"Smallest class has {min_class_size} samples, which may be too small for stratification")
                logger.warning(f"Recommended minimum: {min_samples_needed} samples per class")
            
            # Perform stratified split
            logger.info("Performing stratified train-test split...")
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, 
                test_size=self.test_size,
                stratify=Y,
                random_state=self.random_state
            )
            
            # Log results similar to SimpleTrainTestSplitStrategy
            logger.info("Stratified split completed successfully")
            logger.info(f"  - Training set: X_train {X_train.shape}, Y_train {Y_train.shape}")
            logger.info(f"  - Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")
            
            ProjectLogger.log_success_header(logger, "STRATIFIED DATA SPLITTING COMPLETED")
            
            return X_train, X_test, Y_train, Y_test
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN STRATIFIED SPLITTING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class DataSplitter:
    """
    Main class for handling data splitting operations.
    """
    
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialize data splitter with a specific strategy.
        
        Args:
            strategy (DataSplittingStrategy): Splitting strategy to use
        """
        self.strategy = strategy
        
        ProjectLogger.log_section_header(logger, "INITIALIZING DATA SPLITTER")
        logger.info(f"Using strategy: {strategy.__class__.__name__}")

    @log_exceptions(logger)
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using the configured strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_train, X_test, Y_train, Y_test)
        """
        return self.strategy.split_data(df, target_column)