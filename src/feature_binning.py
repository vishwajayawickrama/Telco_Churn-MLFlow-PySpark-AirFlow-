import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from typing import Dict, List, Union, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

logger = get_logger(__name__)


class FeatureBinningStrategy(ABC):
    """
    Abstract base class for feature binning strategies.
    """
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Bin a feature in the DataFrame.
        
        Args:
            df (DataFrame): Input DataFrame
            column (str): Column to bin
            
        Returns:
            DataFrame: DataFrame with binned feature
        """
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """
    Custom feature binning strategy using user-defined bin definitions.
    """

    def __init__(self, bin_definitions: Dict[str, List[float]], spark: Optional[SparkSession] = None):
        """
        Initialize custom binning strategy.
        
        Args:
            bin_definitions: Dictionary mapping bin names to [min, max] ranges
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.bin_definitions = bin_definitions
        
        ProjectLogger.log_section_header(logger, "INITIALIZING CUSTOM BINNING STRATEGY")
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}")
        

    @log_exceptions(logger)
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Apply custom binning to a feature.
        
        Args:
            df (DataFrame): Input DataFrame
            column (str): Column to bin
            
        Returns:
            DataFrame: DataFrame with binned feature
        """
        ProjectLogger.log_step_header(logger, "STEP", f"APPLYING CUSTOM BINNING TO COLUMN: {column}")
        
        try:
            """
                select: select is used to select specific columns from a DataFrame. It can also be used 
                        to perform operations on columns, such as renaming or applying functions.
                select()
                Parameters:
                    *cols: str or Column - Column names (as strings) or Column expressions to select
                Returns:
                    DataFrame - A new DataFrame with the selected columns
            """
            # Get column statistics
            stats = df.select(
                                F.count(F.col(column)).alias('count'),
                                F.countDistinct(F.col(column)).alias('unique'),
                                F.min(F.col(column)).alias('min'),
                                F.max(F.col(column)).alias('max')
                            ).collect()[0]
            
            logger.info(f"  Unique values: {stats['unique']}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            # Create binning expression
            bin_column = f"{column}Bins"
            
            
                
            # Check each bin definition
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    # Range bin: [min, max]
                    case_expr = case_expr.when(
                        (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1]),
                        bin_label
                    )
                elif len(bin_range) == 1:
                    # Open-ended bin: >= min
                    case_expr = case_expr.when(
                        (F.col(column) >= bin_range[0]),
                        bin_label
                    )
            
            # Apply binning
            case_expr = case_expr.otherwise("Invalid")
            df_binned = df.withColumn(bin_column, case_expr)

            """
                collect: collect is used to retrieve all the rows of a DataFrame as a list of Row objects. 
                         It is an action operation that triggers the execution of the DataFrame transformations.
                collect()
                Parameters: None
                Returns:
                    list - A list of Row objects representing the DataFrame rows    
            """

            """
                groupBy: groupBy is used to group the rows of a DataFrame based on one or more columns. 
                         It is often used in conjunction with aggregation functions to perform calculations on grouped data.
                groupBy(*cols)
                Parameters:
                    *cols: str or Column - Column names (as strings) or Column expressions to group by
                Returns:
                    GroupedData - An object that can be used to perform aggregations on the grouped data
            """

            # Log binning results
            bin_counts = df_binned.groupBy(bin_column).count().orderBy(F.desc('count')).collect()

            logger.info(f"\nBinning Results:")
            total_count = df_binned.count()
            for row in bin_counts:
                bin_name = row[bin_column]
                count = row['count']
                percentage = (count / total_count * 100)
                logger.info(f"  ✓ {bin_name}: {count} ({percentage:.2f}%)")
        
            # Check for invalid values
            invalid_count = df_binned.filter(F.col(bin_column) == "Invalid").count()
            if invalid_count > 0:
                logger.warning(f"  ⚠ Found {invalid_count} invalid values in column '{column}'")
        
             # Drop original column
            df_binned = df_binned.drop(column)
            
            logger.info(f"✓ Original column '{column}' removed, replaced with '{bin_column}'")
            
            ProjectLogger.log_success_header(logger, "CUSTOM BINNING COMPLETED")

            return df_binned
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN FEATURE BINNING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

class BucketizerBinningStrategy(FeatureBinningStrategy):
    """Binning strategy using PySpark's Bucketizer."""

    def __init__(self, splits: List[float], labels: Optional[List[str]] = None, 
                 handle_invalid: str = "keep", spark: Optional[SparkSession] = None):
        """
        Initialize Bucketizer binning strategy.
        
        Args:
            splits: List of split points for binning (must be monotonically increasing)
            labels: Optional list of bin labels (length should be len(splits) - 1)
            handle_invalid: How to handle values outside splits ("keep", "skip", "error")
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.splits = splits
        self.labels = labels
        self.handle_invalid = handle_invalid
        logger.info(f"BucketizerBinningStrategy initialized with {len(splits)-1} bins")

    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Apply Bucketizer binning to a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with binned feature
        """

        ProjectLogger.log_step_header(logger, "STEP", f"APPLYING BUCKETIZER BINNING TO COLUMN: {column}")

        try:
            # Create output column name
            bin_column = f"{column}Bins"
            temp_bin_column = f"{column}_bin_index"

            """
                Bucketizer: Bucketizer is a feature transformer in PySpark that is used to bin 
                            continuous features into discrete bins. It maps a column of continuous 
                            features to a column of feature buckets, where each bucket represents a 
                            range of values.
                Bucketizer(splits, inputCol=None, outputCol=None, handleInvalid='error')
                Parameters:
                    splits: list - A list of split points for binning (must be monotonically increasing 
                    inputCol: str - Name of the input column to be binned
                    outputCol: str - Name of the output column to store binned values
                    handleInvalid: str - How to handle values outside splits ("keep", "skip", "error")
                Returns:
                    Bucketizer - An instance of the Bucketizer transformer
            """

            # Create and apply Bucketizer
            bucketizer = Bucketizer(
                splits=self.splits,
                inputCol=column,
                outputCol=temp_bin_column,
                handleInvalid=self.handle_invalid
            )

            df_binned = bucketizer.transform(df)

            # If labels are provided, map indices to labels
            if self.labels:
                # Create mapping expression
                label_expr = F.when(F.col(temp_bin_column) == 0, self.labels[0])
                for i in range(1, len(self.labels)):
                    label_expr = label_expr.when(F.col(temp_bin_column) == i, self.labels[i])
                label_expr = label_expr.otherwise("Unknown")
            
                df_binned = df_binned.withColumn(bin_column, label_expr)
                df_binned = df_binned.drop(temp_bin_column)
            else:
                # Use numeric bin indices
                df_binned = df_binned.withColumnRenamed(temp_bin_column, bin_column)
            
            # Log binning results
            bin_dist = df_binned.groupBy(bin_column).count().orderBy(F.desc('count')).collect()
            total_count = df_binned.count()
        
            logger.info(f"\nBinning Results:")
            for row in bin_dist:
                bin_value = row[bin_column]
                count = row['count']
                percentage = (count / total_count * 100)
                logger.info(f"Bin {bin_value}: {count} ({percentage:.2f}%)")
        
            # Drop original column if requested
            df_binned = df_binned.drop(column)

            ProjectLogger.log_success_header(logger, "BUCKETIZER BINNING COMPLETED")

            return df_binned
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN FEATURE BINNING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
