import os
import sys
import json
from enum import Enum
from typing import Dict, List, Union, Any, Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

logger = get_logger(__name__)


class FeatureEncodingStrategy(ABC):
    """
    Abstract base class for feature encoding strategies.
    """

    @abstractmethod
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Encode features in the DataFrame.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: DataFrame with encoded features
        """
        pass

class VariableType(str, Enum):
    """Enumeration of variable types."""
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """
    Nominal encoding strategy using StringIndexer.
    Creates numeric indices for categorical values.
    """

    def __init__(self, nominal_columns: List[str], one_hot: bool = False, spark: Optional[SparkSession] = None):
        """
        Initialize nominal encoding strategy.
        
        Args:
            nominal_columns: List of column names to encode
            one_hot: Whether to apply one-hot encoding after indexing
            spark: Optional SparkSession
        """
        from spark_utils import get_spark_session
        self.spark = spark or get_spark_session()
        self.nominal_columns = nominal_columns
        self.one_hot = one_hot
        self.encoder_dicts = {}
        self.indexers = {}
        self.encoders = {}
        os.makedirs('artifacts/encode', exist_ok=True)
        
        ProjectLogger.log_section_header(logger, "INITIALIZING NOMINAL ENCODING STRATEGY")
        logger.info(f"Nominal columns to encode: {len(self.nominal_columns)}")
        logger.info(f"One-hot encoding: {one_hot}")

    @log_exceptions(logger)
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Encode nominal variables using label encoding.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: DataFrame with encoded nominal variables
        """
        ProjectLogger.log_step_header(logger, "STEP", "ENCODING NOMINAL VARIABLES")
        
        try:
            df_encoded = df
            stages = []
            # Validate input


            
            # Create pipeline stages for transformations
            df_result = df_encoded
            
            # Track encoding results
            encoding_summary = {}
            
            for column in self.nominal_columns:
                logger.info(f"\n--- Processing column: {column} ---")

                # Validate column existence
                missing_count = df_encoded.filter(F.col(column).isNull()).count()
                if missing_count > 0:
                    logger.warning(f"Column has {missing_count} missing values before encoding")

                # Fill missing values with a placeholder
                df_encoded = df_encoded.fillna({column: "MISSING"})

                # Get unique values
                unique_values = df_encoded.select(column).distinct().count()
                
                logger.info(f"  Unique values: {unique_values}")

                """
                    StringIndexer: Creates a new column with indexed values. The most frequent value gets index 0.
                    Parameters:
                        inputCol: Name of the input column
                        outputCol: Name of the output indexed column
                        handleInvalid: How to handle invalid data (e.g., unseen labels)
                """
            
                # Create StringIndexer
                indexer = StringIndexer(
                    inputCol=column,
                    outputCol=f"{column}_index",
                    handleInvalid="keep"  # Keeps unseen labels as index = numLabels
                )

                # Fit the indexer
                indexer_model = indexer.fit(df_encoded)

                self.indexers[column] = indexer_model

                # Get the mapping
                labels = indexer_model.labels
                encoder_dict = {label: idx for idx, label in enumerate(labels)}
                self.encoder_dicts[column] = encoder_dict

                

                

                
            
            ProjectLogger.log_success_header(logger, "NOMINAL ENCODING COMPLETED")
            
            return df_result
            
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN NOMINAL ENCODING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """
    Ordinal encoding strategy using custom mappings for ordered categorical variables.
    """

    def __init__(self, ordinal_mappings: Dict[str, Dict[str, int]]):
        """
        Initialize ordinal encoding strategy.
        
        Args:
            ordinal_mappings (Dict): Dictionary mapping column names to their ordinal mappings
                Example: {'education': {'High School': 1, 'Bachelor': 2, 'Master': 3}}
        """
        self.ordinal_mappings = ordinal_mappings
        
        ProjectLogger.log_section_header(logger, "INITIALIZING ORDINAL ENCODING STRATEGY")
        logger.info(f"Ordinal columns to encode: {len(self.ordinal_mappings)}")

    @log_exceptions(logger)
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Encode ordinal variables using custom mappings.
        
        Args:
            df (DataFrame): Input PySpark DataFrame
            
        Returns:
            DataFrame: DataFrame with encoded ordinal variables

        """
        ProjectLogger.log_step_header(logger, "STEP", "ENCODING ORDINAL VARIABLES")
        
        try:
            # Validate input
            if df.count() == 0:
                raise ValueError("Input DataFrame is empty")
            
            if not self.ordinal_mappings:
                logger.warning("No ordinal mappings specified for encoding")
                return df.copy()
            
            logger.info(f"Starting ordinal encoding for {len(self.ordinal_mappings)} columns")

            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            for column, mapping in self.ordinal_mappings.items():
                logger.info(f"Processing column: {column}")
        
                df[column] = df[column].map(mapping)
                logger.info(f"  - Successfully encoded '{column}'")

            logger.info(f"Final DataFrame shape: {df.shape}")
            logger.info(f"Encoded {len(self.ordinal_mappings)} ordinal columns successfully")

            ProjectLogger.log_success_header(logger, "ORDINAL ENCODING COMPLETED")

            return df

        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN ORDINAL ENCODING")
            logger.error(f"Unexpected error: {str(e)}")
            raise



