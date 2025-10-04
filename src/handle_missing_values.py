import os
import sys
import pandas as pd
from enum import Enum
from typing import List, Optional, Union
from abc import ABC, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from spark_utils import get_spark_session

# Initialize logger
logger = get_logger(__name__)


class MissingValueHandlingStrategy(ABC):
    """
    Abstract base class for missing value handling strategies.
    """
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_spark_session()
    
    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values in the DataFrame.
        """
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to handle missing values by dropping rows with missing values in critical columns.
    """
    
    def __init__(self, critical_columns: List[str] = None, spark: Optional[SparkSession] = None):
        """
        Initialize the drop missing values strategy.
        
        Args:
            critical_columns (list): List of column names that are critical for analysis
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.critical_columns = critical_columns or []
        logger.info(f"Initialized DropMissingValuesStrategy for columns: {self.critical_columns}")

    @log_exceptions(logger)
    def handle(self, df: DataFrame) -> DataFrame:
        """
        Drop rows with missing values in critical columns.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: DataFrame with rows containing missing critical values removed

        """
        ProjectLogger.log_step_header(logger, "STEP", "DROPPING MISSING VALUES")
        
        try:
            initial_count = df.count()

            if self.critical_columns:
                # Drop rows with nulls in critical columns
                df_cleaned = df.dropna(subset=self.critical_columns)
            else:
                # If no critical columns specified, drop rows with any nulls
                df_cleaned = df.dropna()

            final_count = df_cleaned.count()
            n_dropped = initial_count - final_count
        
            logger.info(f"Dropped {n_dropped} rows with missing values")
            logger.info(f"Initial rows: {initial_count}")
            logger.info(f"Final rows: {final_count}")

            ProjectLogger.log_success_header(logger, "MISSING VALUES HANDLING COMPLETED")
            
            return df_cleaned
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MISSING VALUES HANDLING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to handle missing values by filling them with specific values or statistical measures.
    """
    
    def __init__(
        self, 
        method: str = 'mean',
        fill_value: Optional[Union[str, float, int]] = None, 
        relevant_column: Optional[str] = None, 
        is_custom_imputer: bool = False,
        custom_imputer: Optional[object] = None,
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize the fill missing values strategy.
        
        Args:
            method: Method to use ('mean', 'median', 'mode', 'constant')
            fill_value: Value to use for constant filling
            relevant_column: Column to fill (if None, fills all numeric columns)
            is_custom_imputer: Whether to use a custom imputer
            custom_imputer: Custom imputer object (must have impute method)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer
        
        ProjectLogger.log_section_header(logger, "INITIALIZING FILL MISSING VALUES STRATEGY")
        logger.info(f"Fill method: {self.method}")
        if self.fill_value:
            logger.info(f"Fill value: {self.fill_value}")
        logger.info(f"Relevant column: {self.relevant_column}")
        if self.is_custom_imputer:
            logger.info(f"Using custom imputer: {self.is_custom_imputer}")

    @log_exceptions(logger)
    def handle(self, df: DataFrame) -> DataFrame:
        """
        Fill missing values in the DataFrame.
        
        Args:
            df (DataFrame): Input DataFrame with missing values
            
        Returns:
            DataFrame: DataFrame with missing values filled

        """
        ProjectLogger.log_step_header(logger, "STEP", "FILLING MISSING VALUES")
        
        try:
            if self.relevant_column:
                if self.method == 'mean':
                    # Calculate mean for the column
                    mean_value = df.select(F.mean(F.col(self.relevant_column))).collect()[0][0]
                    df_filled = df.fillna({self.relevant_column: mean_value})
                    logger.info(f"Filled missing values in '{self.relevant_column}' with mean: {mean_value}")

                elif self.method == 'median':
                    
                    # Calculate median for the column
                    median_value = df.approxQuantile(self.relevant_column, [0.5], 0.01)[0]
                    df_filled = df.fillna({self.relevant_column: median_value})
                    logger.info(f"Filled missing values in '{self.relevant_column}' with median: {median_value}")

                elif self.method == 'mode':
                
                    # Calculate mode for the column
                    mode_value = df.groupBy(self.relevant_column).count().orderBy(F.desc('count')).first()[0]
                    df_filled = df.fillna({self.relevant_column: mode_value})
                    logger.info(f"Filled missing values in '{self.relevant_column}' with mode: {mode_value}")

                elif self.method == 'constant':
                    # Fill with constant value
                    df_filled = df.fillna({self.relevant_column: self.fill_value})
                    logger.info(f"Filled missing values in '{self.relevant_column}' with constant: {self.fill_value}")
                else:
                    raise ValueError(f"Invalid method '{self.method}' or missing fill_value")
            else:
                # Fill all columns based on method
                if self.method == 'constant' and self.fill_value is not None:
                    df_filled = df.fillna(self.fill_value)
                    logger.info(f"Filled all missing values with constant: {self.fill_value}")
                else: 
                    # Use Spark ML Imputer for mean/median on all numeric columns
                    numeric_cols = [field.name for field in df.schema.fields 
                              if field.dataType.typeName() in ['integer', 'long', 'float', 'double']]
                    
                    if numeric_cols:
                        imputer = Imputer(
                            inputCols=numeric_cols,
                            outputCols=[f"{col}_imputed" for col in numeric_cols],
                            strategy=self.method if self.method in ['mean', 'median'] else 'mean'
                        )

                        model = imputer.fit(df)
                        df_imputed = model.transform(df)

                        for col in numeric_cols:
                            df_imputed = df_imputed.withColumn(col, F.col(f"{col}_imputed")).drop(f"{col}_imputed")
                        logger.info(f'✓ Filled missing values in numeric columns using {self.method}')
                    
                        df_filled = df_imputed

                    else:
                        df_filled = df
                        logger.warning('No numeric columns found for imputation')

            ProjectLogger.log_success_header(logger, "MISSING VALUES FILLING COMPLETED")
            return df_filled  
          
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MISSING VALUES FILLING")
            logger.error(f"Unexpected error: {str(e)}")
            raise