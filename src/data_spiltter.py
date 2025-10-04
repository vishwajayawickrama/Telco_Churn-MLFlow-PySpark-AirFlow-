"""
Data splitting module for PySpark DataFrame operations.
Provides various splitting strategies including simple and stratified splitting.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pyspark.sql.utils

from utils.logger import ProjectLogger, log_exceptions
from utils.spark_utils import get_spark_session

# Configure logger
logger = logging.getLogger(__name__)


class SplitType(str, Enum):
    """Enumeration for data splitting types."""
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'


class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies using PySpark.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_spark_session()

    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (train_df, test_df)
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Simple train-test split strategy using PySpark's randomSplit.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: Optional[int] = 42, spark: Optional[SparkSession] = None):
        """
        Initialize simple train-test split strategy.
        
        Args:
            test_size (float): Proportion of dataset for test set (default: 0.2)
            random_state (Optional[int]): Random state for reproducibility (default: 42)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_state = random_state
        
        ProjectLogger.log_section_header(logger, "INITIALIZING SIMPLE TRAIN-TEST SPLIT STRATEGY")
        logger.info(f"Test size: {self.test_size} ({self.test_size * 100}%)")
        logger.info(f"Train size: {self.train_size} ({self.train_size * 100}%)")
        logger.info(f"Random state: {self.random_state}")

    @log_exceptions(logger)
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame]:
        """
        Split data using simple random sampling.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (train_df, test_df)
        """
        ProjectLogger.log_step_header(logger, "STEP", "SPLITTING DATA WITH SIMPLE STRATEGY")
        
        try:
            # Validate input
            total_count = df.count()
            if total_count == 0:
                raise ValueError("Input DataFrame is empty")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            logger.info(f"Total dataset size: {total_count} samples")
            logger.info(f"Features: {len(df.columns) - 1} columns")
            logger.info(f"Target column: {target_column}")
            
            # Check target column distribution
            logger.info("Target distribution:")
            target_distribution = df.groupBy(target_column).count().collect()
            for row in target_distribution:
                value = row[target_column]
                count = row['count']
                percentage = (count / total_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            # Check for missing values
            logger.info("Checking for missing values...")
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Count nulls in features
            null_counts = []
            for col in feature_columns:
                null_count = df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    null_counts.append((col, null_count))
            
            if null_counts:
                logger.warning("Missing values found in features:")
                for col, count in null_counts:
                    logger.warning(f"  - {col}: {count} missing values")
            
            # Check for nulls in target
            target_nulls = df.filter(F.col(target_column).isNull()).count()
            if target_nulls > 0:
                logger.warning(f"Found {target_nulls} missing values in target column")
                logger.warning("Missing target values may affect splitting")
            
            logger.info(f"Features columns: {len(feature_columns)}")
            
            # Perform split using randomSplit
            logger.info("Performing train-test split...")
            train_df, test_df = df.randomSplit([self.train_size, self.test_size], seed=self.random_state)
            
            # Cache the splits for performance
            train_df.cache()
            test_df.cache()
            
            # Get actual counts
            train_count = train_df.count()
            test_count = test_df.count()
            
            # Log split results
            logger.info("Split completed successfully:")
            logger.info(f"  - Training set: {train_count} samples")
            logger.info(f"  - Test set: {test_count} samples")
            
            # Verify split proportions
            actual_test_proportion = test_count / total_count
            logger.info(f"  - Actual test proportion: {actual_test_proportion:.3f} (target: {self.test_size})")
            
            # Check target distribution in splits
            logger.info("Target distribution in training set:")
            train_target_dist = train_df.groupBy(target_column).count().collect()
            for row in train_target_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / train_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            logger.info("Target distribution in test set:")
            test_target_dist = test_df.groupBy(target_column).count().collect()
            for row in test_target_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / test_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            ProjectLogger.log_success_header(logger, "SIMPLE DATA SPLITTING COMPLETED")
            
            return train_df, test_df
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            logger.error(f"Available columns: {df.columns}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN DATA SPLITTING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Stratified train-test split strategy to maintain target distribution using PySpark.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: Optional[int] = 42, spark: Optional[SparkSession] = None):
        """
        Initialize stratified train-test split strategy.
        
        Args:
            test_size (float): Proportion of dataset for test set (default: 0.2)
            random_state (Optional[int]): Random state for reproducibility (default: 42)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_state = random_state
        
        ProjectLogger.log_section_header(logger, "INITIALIZING STRATIFIED TRAIN-TEST SPLIT STRATEGY")
        logger.info(f"Test size: {self.test_size} ({self.test_size * 100}%)")
        logger.info(f"Train size: {self.train_size} ({self.train_size * 100}%)")
        logger.info(f"Random state: {self.random_state}")
        logger.info("Method: Maintains target distribution across train/test splits")

    @log_exceptions(logger)
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame]:
        """
        Split data using stratified sampling to maintain target distribution.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (train_df, test_df)
        """
        ProjectLogger.log_step_header(logger, "STEP", "SPLITTING DATA WITH STRATIFIED STRATEGY")
        
        try:
            # Validate input
            total_count = df.count()
            if total_count == 0:
                raise ValueError("Input DataFrame is empty")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            logger.info(f"Total dataset size: {total_count} samples")
            
            # Get target class distribution
            logger.info("Analyzing target distribution for stratification...")
            target_counts = df.groupBy(target_column).count().collect()
            
            # Check if stratification is feasible
            min_class_size = min([row['count'] for row in target_counts])
            min_samples_needed = max(1, int(1 / self.test_size))
            
            if min_class_size < min_samples_needed:
                logger.warning(f"Smallest class has {min_class_size} samples, which may be too small for stratification")
                logger.warning(f"Recommended minimum: {min_samples_needed} samples per class")
                logger.warning("Falling back to simple random split")
                return SimpleTrainTestSplitStrategy(self.test_size, self.random_state, self.spark).split_data(df, target_column)
            
            # Perform stratified split by sampling each class separately
            logger.info("Performing stratified sampling for each class...")
            
            train_dfs = []
            test_dfs = []
            
            for row in target_counts:
                class_value = row[target_column]
                class_count = row['count']
                
                logger.info(f"Splitting class '{class_value}' ({class_count} samples)")
                
                # Filter data for this class
                class_df = df.filter(F.col(target_column) == class_value)
                
                # Calculate expected test samples for this class
                expected_test_samples = int(class_count * self.test_size)
                expected_train_samples = class_count - expected_test_samples
                
                logger.info(f"  Expected split: {expected_train_samples} train, {expected_test_samples} test")
                
                # Use sampleBy for stratified sampling
                # Create fractions dict for this class
                fractions = {class_value: self.test_size}
                
                # Sample test set
                test_class_df = class_df.sample(withReplacement=False, fraction=self.test_size, seed=self.random_state)
                
                # Get training set by anti-joining
                train_class_df = class_df.subtract(test_class_df)
                
                # Add to lists
                train_dfs.append(train_class_df)
                test_dfs.append(test_class_df)
                
                # Log actual split
                actual_train = train_class_df.count()
                actual_test = test_class_df.count()
                logger.info(f"  Actual split: {actual_train} train, {actual_test} test")
            
            # Union all class splits
            logger.info("Combining stratified splits...")
            train_df = train_dfs[0]
            for i in range(1, len(train_dfs)):
                train_df = train_df.union(train_dfs[i])
            
            test_df = test_dfs[0]
            for i in range(1, len(test_dfs)):
                test_df = test_df.union(test_dfs[i])
            
            # Cache the results
            train_df.cache()
            test_df.cache()
            
            # Get final counts
            train_count = train_df.count()
            test_count = test_df.count()
            
            # Log results
            logger.info("Stratified split completed successfully")
            logger.info(f"  - Training set: {train_count} samples")
            logger.info(f"  - Test set: {test_count} samples")
            
            # Verify proportions
            actual_test_proportion = test_count / total_count
            logger.info(f"  - Actual test proportion: {actual_test_proportion:.3f} (target: {self.test_size})")
            
            # Check final target distributions
            logger.info("Final target distribution in training set:")
            final_train_dist = train_df.groupBy(target_column).count().collect()
            for row in final_train_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / train_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            logger.info("Final target distribution in test set:")
            final_test_dist = test_df.groupBy(target_column).count().collect()
            for row in final_test_dist:
                value = row[target_column]
                count = row['count']
                percentage = (count / test_count) * 100
                logger.info(f"  - {value}: {count} samples ({percentage:.2f}%)")
            
            ProjectLogger.log_success_header(logger, "STRATIFIED DATA SPLITTING COMPLETED")
            
            return train_df, test_df
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN STRATIFIED SPLITTING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class DataSplitter:
    """
    Main class for handling data splitting operations with PySpark.
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
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame]:
        """
        Split data using the configured strategy.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (train_df, test_df)
        """
        return self.strategy.split_data(df, target_column)
    
    @log_exceptions(logger)
    def split_and_separate_features_target(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data and separate features from target.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_train, X_test, Y_train, Y_test) where Y DataFrames contain only target column
        """
        train_df, test_df = self.split_data(df, target_column)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        
        X_train = train_df.select(*feature_columns)
        X_test = test_df.select(*feature_columns)
        Y_train = train_df.select(target_column)
        Y_test = test_df.select(target_column)
        
        return X_train, X_test, Y_train, Y_test


def create_data_splitter(split_type: SplitType = SplitType.STRATIFIED, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = 42,
                        spark: Optional[SparkSession] = None) -> DataSplitter:
    """
    Factory function to create data splitter with specified configuration.
    
    Args:
        split_type (SplitType): Type of splitting strategy
        test_size (float): Proportion of dataset for test set
        random_state (Optional[int]): Random state for reproducibility
        spark: Optional SparkSession
        
    Returns:
        DataSplitter: Configured data splitter
    """
    if split_type == SplitType.SIMPLE:
        strategy = SimpleTrainTestSplitStrategy(test_size, random_state, spark)
    elif split_type == SplitType.STRATIFIED:
        strategy = StratifiedTrainTestSplitStrategy(test_size, random_state, spark)
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    return DataSplitter(strategy)