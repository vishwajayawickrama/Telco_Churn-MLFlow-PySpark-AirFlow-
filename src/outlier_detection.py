import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from spark_utils import get_spark_session

# Initialize logger
logger = get_logger(__name__)


class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for outlier detection strategies.
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_spark_session()

    @abstractmethod
    def detect_outliers(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df (DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            
        Returns:
            DataFrame: Boolean DataFrame with additional column indicating outliers
        """
        pass

    @abstractmethod
    def get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Get outlier bounds for specified columns.
        
        Args:
            df: DataFrame (PySpark or pandas)
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """
    Outlier detection using Interquartile Range (IQR) method.
    """
    
    def __init__(self, threshold: float = 1.5, spark: Optional[SparkSession] = None):
        """
        Initialize IQR outlier detection.
        
        Args:
            threshold (float): IQR threshold for outlier detection (default: 1.5)
            spark: Optional SparkSession
        """

        super().__init__(spark)
        self.threshold = threshold

        ProjectLogger.log_section_header(logger, "INITIALIZING IQR OUTLIER DETECTION")

    def get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate outlier bounds using IQR method.
        
        Args:
            df: PySpark DataFrame
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        ProjectLogger.log_step_header(logger, "STEP", "CALCULATING OUTLIER BOUNDS USING IQR METHOD")
        bounds = {}

        for col in columns:

            # Calculate Q1 and Q3
            """
                approxQuantile: - calculates the approximate quantiles of a numeric column in a DataFrame. 
                                - It is a more efficient and scalable alternative to exact quantile 
                                    calculation for large datasets.
                                - It uses an algorithm based on the Greenwald-Khanna algorithm to compute 
                                    quantiles with a specified level of error. Instead of sorting the 
                                    entire dataset, it maintains a summary of the data in a more compact form.

                Parameters
                    df.approxQuantile(col, probabilities, relativeError)
                    - col: The name of the numeric column you want to analyze.
                    - probabilities: A list of float values between 0.0 and 1.0 representing the 
                                        quantiles to compute (e.g., [0.5] for the median).
                    - relativeError: A float value between 0.0 and 1.0 that determines the acceptable error. 
                                        A value of 0.0 provides the exact quantile, while larger values lead 
                                        to faster computation at the cost of precision.
            """
            quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
            Q1 = quantiles[0]
            Q3 = quantiles[1]
            IQR = Q3 - Q1

            upper_bound = Q3 + self.threshold * IQR
            lower_bound = Q1 - self.threshold * IQR

            bounds[col] = (lower_bound, upper_bound)

            logger.info(f"  Column '{col}': Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
            logger.info(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return bounds

    @log_exceptions(logger)
    def detect_outliers(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df (DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            
        Returns:
            DataFrame with additional boolean columns '{col}_outlier' indicating outliers
        """
        ProjectLogger.log_step_header(logger, "STEP", "DETECTING OUTLIERS USING IQR METHOD")
        
        try:
            ProjectLogger.log_success_header(logger, "OUTLIER DETECTION COMPLETED")
            bounds = self.get_outlier_bounds(df, columns)

            result_df = df
            total_outliers = 0

            for col in columns:
                logger.info(f"----Processing column '{col}' for outlier detection----")

                # get bounds for column
                lower_bound, upper_bound = bounds[col]

                outlier_col = f"{col}_outlier"

                """
                    withColumn: - is a method in PySpark's DataFrame API that allows you to create a new 
                                    column or replace an existing column in a DataFrame. It is commonly 
                                    used for data transformation and feature engineering tasks.
                    .withColumn(colName, expr)
                    Parameters:
                        - The first argument is the name of the new or existing column.
                        - The second argument is an expression that defines the values for the new column.
                        - The expression can be a literal value, a column reference, or a more complex 
                            expression involving functions and operations on existing columns.
                """

                result_df = result_df.withColumn(
                                            outlier_col,
                                            (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
                                            )
                """
                    filter: - is a method in PySpark's DataFrame API that allows you to filter rows in 
                                a DataFrame based on a specified condition. It is used to create a new 
                                DataFrame that contains only the rows that satisfy the given condition.
                    df.filter(condition)
                    Parameters:
                        - The condition is typically expressed using PySpark's Column expressions, 
                            which can involve comparisons, logical operations, and other DataFrame 
                            column manipulations.
                """
                outlier_count = result_df.filter(F.col(outlier_col)).count()

                total_rows = result_df.count()

                outlier_percentage = (outlier_count / total_rows * 100) if total_rows > 0 else 0
                logger.info(f"  ✓ Found {outlier_count} outliers ({outlier_percentage:.2f}%)")
                total_outliers += outlier_count

                ProjectLogger.log_success_header(logger, f"OUTLIER DETECTION FOR COLUMN '{col}' COMPLETED")

            ProjectLogger.log_success_header(logger, "OUTLIER DETECTION FOR ALL COLUMNS COMPLETED")
            return result_df
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN OUTLIER DETECTION")
            logger.error(f"Unexpected error: {str(e)}")
            raise

class OutlierDetector:
    """Main outlier detector class that uses different strategies."""

    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initialize outlier detector with a specific strategy.
        
        Args:
            strategy: OutlierDetectionStrategy instance
        """
        self.strategy = strategy

    def detect_outliers(self, df: DataFrame, selected_columns: List[str]) -> DataFrame:
            """
            Detect outliers in specified columns using the chosen strategy.
            
            Args:
                df (DataFrame): Input DataFrame
                selected_columns (List[str]): List of columns to check for outliers

            Returns:
                DataFrame: DataFrame with outlier information
            """
            return self._strategy.detect_outliers(df, selected_columns)

    def handle_outliers(self, df: DataFrame, selected_columns: List[str], 
                        method: str = 'remove', min_outliers: int = 2) -> DataFrame:
        """
        Handle outliers in specified columns using specified method.
        
        Args:
            df: DataFrame (PySpark or pandas)
            selected_columns: List of column names to check
            method: Method to handle outliers ('remove' or 'cap')
            min_outliers: Minimum number of outlier columns to remove a row

        Returns:
            DataFrame: DataFrame with outliers handled
        """
        ProjectLogger.log_step_header(logger, "STEP", f"HANDLING OUTLIERS USING METHOD: {method.upper()}")

        try:
            initial_rows = df.count()

            if self.method == 'remove':
                # Add outlier indicator columns
                df_with_outliers = self.detect_outliers(df, selected_columns)

                # Count outliers per row
                outlier_columns = [f"{col}_outlier" for col in selected_columns]

                """
                    cast: - is a method in PySpark's Column class that allows you to convert the data type of a column
                                to a different data type. It is commonly used when you need to change the type of
                                a column for various operations, such as filtering, aggregations, or transformations.
                    col.cast(dataType)
                    Parameters:
                        - dataType: The target data type to which you want to convert the column.  
                    however, in this case we are using it to convert boolean to integer (True to 1 and False to 0) 
                """
                """
                    sum: - is a function in PySpark's functions module that calculates the sum of values in a column or 
                            expression across all rows in a DataFrame. It is commonly used in aggregation operations 
                            to compute the total or sum of numeric values.
                    sum(expr)
                    Parameters:
                        - expr: The expression or column for which you want to calculate the sum.
                    in this case we are using it to sum up the integer values (0s and 1s) of outlier indicator columns to get the total count of outliers per row.
                """
                # Create expression to count outliers
                outlier_count_expr = sum(F.col(col).cast("int") for col in outlier_columns)
            
                # Add outlier count column
                df_with_count = df_with_outliers.withColumn("outlier_count", outlier_count_expr)

                # Filter rows with fewer outliers than threshold
                cleaned_df = df_with_count.filter(F.col("outlier_count") < min_outliers)

                # Remove temporary columns
                cleaned_df = cleaned_df.drop("outlier_count")
                for col in outlier_columns:
                    cleaned_df = cleaned_df.drop(col)

                rows_removed = initial_rows - cleaned_df.count()
                removal_percentage = (rows_removed / initial_rows * 100) if initial_rows > 0 else 0

                logger.info(f"Removed {rows_removed} rows with {min_outliers}+ outliers ({removal_percentage:.2f}%)")
                logger.info(f"Remaining rows: {cleaned_df.count()} ({(cleaned_df.count()/initial_rows*100):.2f}%)")
            
            if self.method =='cap':
                bounds = self.strategy.get_outlier_bounds(df, selected_columns)
                cleaned_df = df

                for col in selected_columns:
                    lower_bound, upper_bound = bounds[col]
                    """
                        when: - is a function in PySpark's functions module that allows you to create conditional 
                                    expressions similar to SQL's CASE WHEN statements. It is used to evaluate 
                                    conditions and return different values based on those conditions.
                        when(condition, value)
                        Parameters:
                            - condition: The condition to evaluate (a Column expression).
                            - value: The value to return if the condition is true.
                        otherwise(value)
                            - value: The value to return if none of the when conditions are true.
                    """
                    cleaned_df = cleaned_df.withColumn(
                        col,
                        F.when(F.col(col) < lower_bound, lower_bound)
                        .when(F.col(col) > upper_bound, upper_bound)
                        .otherwise(F.col(col))
                        )
                    logger.info(f"Capped outliers at IQR bounds for {len(selected_columns)} columns")
                else:
                    raise ValueError(f"Unknown outlier handling method: {method}")
                
            ProjectLogger.log_success_header(logger, "OUTLIER HANDLING COMPLETED")
            return cleaned_df

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
