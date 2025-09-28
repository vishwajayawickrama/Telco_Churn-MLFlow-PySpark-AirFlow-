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
