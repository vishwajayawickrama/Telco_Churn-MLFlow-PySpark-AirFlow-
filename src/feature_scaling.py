"""
Feature scaling module for PySpark DataFrame operations.
Provides various scaling strategies including Min-Max and Standard scaling.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Optional
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

from utils.logger import ProjectLogger
from utils.config import log_exceptions
from utils.spark_utils import get_spark_session

# Configure logger
logger = logging.getLogger(__name__)


class ScalingType(Enum):
    """Enumeration for different scaling types."""
    MINMAX = 'minmax'
    STANDARD = 'standard'


class FeatureScalingStrategy(ABC):
    """
    Abstract base class for feature scaling strategies using PySpark.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_spark_session()

    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Scale features in the DataFrame.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            DataFrame: DataFrame with scaled features
        """
        pass


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """
    Min-Max scaling strategy using PySpark ML MinMaxScaler.
    """
    
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0, spark: Optional[SparkSession] = None):
        """
        Initialize Min-Max scaling strategy.
        
        Args:
            min_value (float): Minimum value for scaled features (default: 0.0)
            max_value (float): Maximum value for scaled features (default: 1.0)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.min_value = min_value
        self.max_value = max_value
        self.fitted_scalers = {}
        self.scaling_stats = {}
        
        ProjectLogger.log_section_header(logger, "INITIALIZING MIN-MAX SCALING STRATEGY")
        logger.info(f"Feature range: ({self.min_value}, {self.max_value})")
        logger.info("Method: (X - min) / (max - min) * (max_value - min_value) + min_value")

    @log_exceptions(logger)
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            DataFrame: DataFrame with scaled features
        """
        ProjectLogger.log_step_header(logger, "STEP", "APPLYING MIN-MAX SCALING")
        
        try:
            # Validate input
            if df.count() == 0:
                raise ValueError("Input DataFrame is empty")
            
            if not columns_to_scale:
                logger.warning("No columns specified for scaling")
                return df
            
            logger.info(f"Scaling {len(columns_to_scale)} columns: {columns_to_scale}")
            logger.info(f"Initial DataFrame row count: {df.count()}")
            
            # Validate columns exist
            available_columns = df.columns
            missing_columns = [col for col in columns_to_scale if col not in available_columns]
            if missing_columns:
                logger.warning(f"Columns not found: {missing_columns}")
                existing_columns = [col for col in columns_to_scale if col in available_columns]
                logger.info(f"Using existing columns: {existing_columns}")
            else:
                existing_columns = columns_to_scale
            
            if not existing_columns:
                logger.warning("No valid columns for scaling")
                return df
            
            # Handle TotalCharges conversion if present
            df_result = df
            if "TotalCharges" in existing_columns:
                logger.info("Converting TotalCharges to numeric (handling potential string values)")
                df_result = df_result.withColumn("TotalCharges", 
                    F.when(F.col("TotalCharges").rlike("^[0-9.]+$"), 
                           F.col("TotalCharges").cast(DoubleType()))
                    .otherwise(F.lit(None).cast(DoubleType())))
            
            # Apply scaling to each column
            for column in existing_columns:
                logger.info(f"Scaling column: {column}")
                
                # Ensure column is numeric
                df_result = df_result.withColumn(column, F.col(column).cast(DoubleType()))
                
                # Create vector assembler for single column
                assembler = VectorAssembler(
                    inputCols=[column],
                    outputCol=f"{column}_vector",
                    handleInvalid="keep"
                )
                
                # Create MinMax scaler
                scaler = MinMaxScaler(
                    inputCol=f"{column}_vector",
                    outputCol=f"{column}_scaled_vector",
                    min=self.min_value,
                    max=self.max_value
                )
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler])
                pipeline_model = pipeline.fit(df_result)
                
                # Transform data
                df_scaled = pipeline_model.transform(df_result)
                
                # Extract scaled values from vector
                df_result = df_scaled.withColumn(
                    column,
                    F.col(f"{column}_scaled_vector").getItem(0)
                ).drop(f"{column}_vector", f"{column}_scaled_vector")
                
                # Store scaler for future use
                self.fitted_scalers[column] = pipeline_model
                
                # Calculate and store statistics
                stats = df_result.select(
                    F.min(column).alias("min"),
                    F.max(column).alias("max"),
                    F.mean(column).alias("mean"),
                    F.stddev(column).alias("std")
                ).collect()[0]
                
                self.scaling_stats[column] = {
                    "min": stats["min"],
                    "max": stats["max"], 
                    "mean": stats["mean"],
                    "std": stats["std"]
                }
                
                logger.info(f"Column {column} scaled successfully")
                logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            logger.info(f"Min-Max scaling completed for {len(existing_columns)} columns")
            ProjectLogger.log_success_header(logger, "MIN-MAX SCALING COMPLETED")
            
            return df_result
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "MIN-MAX SCALING FAILED")
            logger.error(f"Error during Min-Max scaling: {str(e)}")
            raise


class StandardScalingStrategy(FeatureScalingStrategy):
    """
    Standard scaling (Z-score normalization) strategy using PySpark ML StandardScaler.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, spark: Optional[SparkSession] = None):
        """
        Initialize Standard scaling strategy.
        
        Args:
            with_mean (bool): Whether to center data at mean (default: True)
            with_std (bool): Whether to scale to unit variance (default: True)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.fitted_scalers = {}
        self.scaling_stats = {}
        
        ProjectLogger.log_section_header(logger, "INITIALIZING STANDARD SCALING STRATEGY")
        logger.info(f"Center data (subtract mean): {self.with_mean}")
        logger.info(f"Scale to unit variance: {self.with_std}")
        logger.info("Method: (X - mean) / std")

    @log_exceptions(logger)
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Standard scaling to specified columns.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            DataFrame: DataFrame with scaled features
        """
        ProjectLogger.log_step_header(logger, "STEP", "APPLYING STANDARD SCALING")
        
        try:
            # Validate input
            if df.count() == 0:
                raise ValueError("Input DataFrame is empty")
            
            if not columns_to_scale:
                logger.warning("No columns specified for scaling")
                return df
            
            logger.info(f"Scaling {len(columns_to_scale)} columns: {columns_to_scale}")
            
            # Validate columns exist
            available_columns = df.columns
            existing_columns = [col for col in columns_to_scale if col in available_columns]
            missing_columns = [col for col in columns_to_scale if col not in available_columns]
            
            if missing_columns:
                logger.warning(f"Columns not found: {missing_columns}")
            
            if not existing_columns:
                logger.warning("No valid columns for scaling")
                return df
            
            # Handle TotalCharges conversion if present
            df_result = df
            if "TotalCharges" in existing_columns:
                logger.info("Converting TotalCharges to numeric")
                df_result = df_result.withColumn("TotalCharges", 
                    F.when(F.col("TotalCharges").rlike("^[0-9.]+$"), 
                           F.col("TotalCharges").cast(DoubleType()))
                    .otherwise(F.lit(None).cast(DoubleType())))
            
            # Apply scaling to each column
            for column in existing_columns:
                logger.info(f"Scaling column: {column}")
                
                # Ensure column is numeric
                df_result = df_result.withColumn(column, F.col(column).cast(DoubleType()))
                
                # Create vector assembler for single column
                assembler = VectorAssembler(
                    inputCols=[column],
                    outputCol=f"{column}_vector",
                    handleInvalid="keep"
                )
                
                # Create Standard scaler
                scaler = StandardScaler(
                    inputCol=f"{column}_vector",
                    outputCol=f"{column}_scaled_vector",
                    withMean=self.with_mean,
                    withStd=self.with_std
                )
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler])
                pipeline_model = pipeline.fit(df_result)
                
                # Transform data
                df_scaled = pipeline_model.transform(df_result)
                
                # Extract scaled values from vector
                df_result = df_scaled.withColumn(
                    column,
                    F.col(f"{column}_scaled_vector").getItem(0)
                ).drop(f"{column}_vector", f"{column}_scaled_vector")
                
                # Store scaler for future use
                self.fitted_scalers[column] = pipeline_model
                
                # Calculate and store statistics
                stats = df_result.select(
                    F.min(column).alias("min"),
                    F.max(column).alias("max"),
                    F.mean(column).alias("mean"),
                    F.stddev(column).alias("std")
                ).collect()[0]
                
                self.scaling_stats[column] = {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "std": stats["std"]
                }
                
                logger.info(f"Column {column} scaled successfully")
                logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            logger.info(f"Standard scaling completed for {len(existing_columns)} columns")
            ProjectLogger.log_success_header(logger, "STANDARD SCALING COMPLETED")
            
            return df_result
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "STANDARD SCALING FAILED")
            logger.error(f"Error during Standard scaling: {str(e)}")
            raise


class FeatureScaler:
    """
    Main feature scaling class that uses different scaling strategies.
    """
    
    def __init__(self, scaling_type: Union[str, ScalingType] = ScalingType.MINMAX, spark: Optional[SparkSession] = None):
        """
        Initialize feature scaler with specified strategy.
        
        Args:
            scaling_type: Type of scaling to apply ('minmax' or 'standard')
            spark: Optional SparkSession
        """
        self.spark = spark or get_spark_session()
        self.scaling_type = ScalingType(scaling_type) if isinstance(scaling_type, str) else scaling_type
        self.strategy = self._create_strategy()
        
        ProjectLogger.log_section_header(logger, f"INITIALIZING FEATURE SCALER ({self.scaling_type.value.upper()})")

    def _create_strategy(self) -> FeatureScalingStrategy:
        """Create scaling strategy based on type."""
        if self.scaling_type == ScalingType.MINMAX:
            return MinMaxScalingStrategy(spark=self.spark)
        elif self.scaling_type == ScalingType.STANDARD:
            return StandardScalingStrategy(spark=self.spark)
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

    @log_exceptions(logger)
    def scale_features(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Scale features using the configured strategy.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            DataFrame: DataFrame with scaled features
        """
        return self.strategy.scale(df, columns_to_scale)
    
    def get_scaling_stats(self) -> dict:
        """Get scaling statistics."""
        return getattr(self.strategy, 'scaling_stats', {})
    
    def get_fitted_scalers(self) -> dict:
        """Get fitted scaler models."""
        return getattr(self.strategy, 'fitted_scalers', {})